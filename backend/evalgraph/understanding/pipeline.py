"""Semantic Code Map pipeline: INGEST -> PARSE -> UNDERSTAND -> MAP -> RECOMMEND (deck slide 6).

Works on any Python / JavaScript / TypeScript LLM application. Every LLM stage has a deterministic fallback,
so an analysis never fails because the model is unreachable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..analyzer.extract import extract_source
from ..analyzer.facts import FileFacts
from ..core.schemas import ApplicationGraph, Hotspot, Recommendation
from ..hotspots.engine import recommend
from ..llm import dial_reachable
from ..settings import get_settings
from .aimap import render as render_aimap
from .assemble import Assembler, reduce_with_llm
from .buffer import RepoBuffer, load
from .cards import run_cards
from .components import build_components
from .flow import FlowResult, analyze_flows, annotate_graph
from .investigate import investigate
from .jsfacts import extract_js
from .miners import Finding, Miner
from .review import _dedupe, review
from .scan_eval import has_eval_suite
from .signals import tag_symbols
from .slices import select_slices

logger = logging.getLogger(__name__)
Report = Callable[..., None]

LANG_NAMES = {".py": "python", ".ipynb": "python (notebook)", ".ts": "typescript", ".tsx": "typescript",
              ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript"}


@dataclass
class Understanding:
    buffer: RepoBuffer
    graph: ApplicationGraph
    hotspots: list[Hotspot]
    recommendations: list[Recommendation]
    frameworks: list[str]
    languages: list[str]
    evalgraph_yaml: dict | None
    files_scanned: int
    llm_calls: int
    stats: dict = field(default_factory=dict)
    runtime: dict | None = None
    flow: FlowResult | None = None
    aimap: str = ""


def _recover_notebook(path: str, text: str) -> FileFacts:
    """Blank out cells that do not parse (line numbers stay valid) and parse the rest."""
    import ast
    import re as _re

    parts = _re.split(r"(?m)^(?=# %% \[cell \d+\])", text)
    fixed = []
    for cell in parts:
        try:
            ast.parse(cell)
            fixed.append(cell)
        except SyntaxError:
            fixed.append("\n".join("# (unparseable cell) " + ln for ln in cell.split("\n")))
    return extract_source(path, "".join(fixed))


def parse(buffer: RepoBuffer) -> list[FileFacts]:
    files = []
    for path, text in buffer.code_files().items():
        ext = Path(path).suffix
        if ext in (".py", ".ipynb"):
            facts = extract_source(path, text)
            if facts.error and ext == ".ipynb":
                facts = _recover_notebook(path, text)
            files.append(facts)
        else:
            files.append(extract_js(path, text, ext))
    return files


def frameworks_of(files: list[FileFacts], buffer: RepoBuffer) -> tuple[list[str], bool]:
    known = {
        "langgraph": "langgraph", "langchain": "langchain", "langchain_core": "langchain", "langchain_community": "langchain",
        "langchain_openai": "langchain", "llama_index": "llama-index", "llamaindex": "llama-index", "haystack": "haystack",
        "dspy": "dspy", "crewai": "crewai", "autogen": "autogen", "semantic_kernel": "semantic-kernel",
        "qdrant_client": "qdrant", "chromadb": "chromadb", "pinecone": "pinecone", "faiss": "faiss", "weaviate": "weaviate",
        "fastembed": "fastembed", "sentence_transformers": "sentence-transformers", "openai": "openai",
        "anthropic": "anthropic", "litellm": "litellm", "ollama": "ollama", "google": "google-genai", "ai": "vercel-ai-sdk",
        "fastapi": "fastapi", "flask": "flask", "streamlit": "streamlit", "gradio": "gradio", "chainlit": "chainlit",
        "express": "express", "next": "next.js", "groq": "groq", "mistralai": "mistral", "cohere": "cohere",
    }
    found, azure = set(), False
    for f in files:
        for imp in f.imports:
            mod = imp.module.lstrip("@").replace("/", ".").split(".")[0]
            if mod in known:
                found.add(known[mod])
            azure |= "AzureOpenAI" in imp.names or "AzureChatOpenAI" in imp.names
    dial = azure and any("ai-proxy" in t for t in buffer.files.values())
    if dial:
        found.add("openai-azure(dial)")
    return sorted(found), dial


def understand(source: dict, report: Report) -> Understanding:
    use_llm = not get_settings().mock_llm

    report("INGEST · cloning into memory" if source.get("repo_url") else "INGEST · reading folder into memory", 0.04)
    buffer = load(source.get("repo_url"), source.get("local_path"))
    if use_llm and not dial_reachable():  # don't wait out LLM timeouts: analyze statically and say so
        use_llm = False
        report("UNDERSTAND · DIAL unreachable", 0.1, "LLM steps skipped: DIAL is unreachable (VPN off?); static analysis only")
    report("PARSE · building the code graph", 0.12,
           f"INGEST: {buffer.name}{' @ ' + buffer.commit[:7] if buffer.commit else ''} · {len(buffer.files)} files "
           f"({buffer.size_bytes // 1024} KB) held in memory" + (", clone deleted" if buffer.repo_url else ""))

    files = parse(buffer)
    languages = sorted({LANG_NAMES[Path(f.path).suffix] for f in files if Path(f.path).suffix in LANG_NAMES})
    n_defs = sum(len(f.defs) for f in files)
    frameworks, dial = frameworks_of(files, buffer)
    report("UNDERSTAND · tagging AI components", 0.22,
           f"PARSE: {len(files)} code files ({', '.join(languages) or 'none'}) · {n_defs} symbols · "
           f"frameworks: {', '.join(frameworks) or 'none detected'}")

    symbols = tag_symbols(files, dial)
    comps, g, cstats = build_components(files, symbols)
    slices = select_slices(comps, symbols, cstats["scope"], cstats["ranks"], buffer, files)
    calls = 0
    report("UNDERSTAND · semantic mapping", 0.35,
           f"UNDERSTAND: {len(comps)} AI components from {cstats['ai_symbols']} tagged symbols · "
           f"{len(slices)} code slices selected by PageRank ({sum(s.size for s in slices) // 1024} KB)")
    card_stats = {}
    if use_llm and slices:
        comps, card_stats = run_cards(slices, comps, symbols, g.out)
        calls += card_stats["calls"]
        report("MAP · assembling the pipeline", 0.55,
               f"UNDERSTAND (LLM): {card_stats['cards']} component cards in {card_stats['calls']} calls · "
               f"{card_stats['discovered']} components discovered · {card_stats['wrappers']} wrappers of known components skipped · "
               f"{card_stats['rejected']} unverifiable claims dropped")
    else:
        report("MAP · assembling the pipeline", 0.55, "UNDERSTAND (LLM): skipped (mock/offline) · static cards used")

    asm = Assembler(comps, symbols, g, files)
    graph = asm.build()
    red_stats = {}
    if use_llm and graph.nodes:
        graph, red_stats = reduce_with_llm(graph, asm)
        calls += red_stats.get("calls", 0)
    flow = analyze_flows(files, symbols, comps, cstats["scope"])
    flow_edges = annotate_graph(graph, flow)
    report("RECOMMEND · mining bottlenecks", 0.7,
           f"MAP: {len(graph.nodes)} components · {len(graph.edges)} connections"
           + (f" · LLM added {red_stats.get('added', 0)}, removed {red_stats.get('removed', 0)}" if red_stats else "")
           + f" · data flow: {len(flow.recipes)} prompt recipes, {len(flow.flows)} flows ({flow_edges} new edges)")

    eval_suite = has_eval_suite(buffer)
    variants: list[str] = []
    ev_yaml = None
    if "evalgraph.yaml" in buffer.files:
        try:
            ev_yaml = yaml.safe_load(buffer.files["evalgraph.yaml"])
        except yaml.YAMLError:
            ev_yaml = None
    if isinstance(ev_yaml, dict):
        from ..runner.adapters import get_adapter

        adapter = get_adapter(ev_yaml.get("adapter"))
        variants = adapter.variants() if adapter else []
    findings = Miner(buffer, files, symbols, comps, g, eval_suite, cstats["scope"], flow=flow).run()
    investigation = None
    if use_llm and get_settings().investigator and graph.nodes:
        aimap = render_aimap(buffer.name, frameworks, languages, graph, flow, findings, buffer.commit)
        report("RECOMMEND · investigating", 0.75, f"RECOMMEND: AI map of {len(aimap) // 1024} KB "
               f"(~{len(aimap) // 4000}k tokens) handed to the investigator")
        investigation = investigate(aimap, _dedupe(findings), graph, buffer, files, symbols, comps, g, flow)
        calls += investigation.stats["calls"] if investigation else 0
    hotspots, rstats = review(findings, graph, slices, buffer, variants, use_llm=use_llm, investigation=investigation)
    from ..hotspots.patches import attach_patches

    rstats["patched"] = attach_patches(hotspots, buffer)
    calls += rstats["calls"]
    proposals = investigation.evals if investigation else []
    runtime = plan_runtime(buffer, symbols, comps, g, cstats, graph, use_llm, hotspots, proposals)
    # Labels exist when a registered adapter ships a golden set. A sandbox run uses generated questions (no labels);
    # a plan-only repo's plan is for the user's own harness, where labelled evaluators make sense.
    labelled = bool(variants) or runtime["plan"].get("mode") == "unsupported"
    recommendations = recommend(graph, hotspots, labelled=labelled, flow=flow, proposals=proposals)
    # the stored map shows the final hotspots (after review), not the raw static candidates
    aimap = render_aimap(buffer.name, frameworks, languages, graph, flow,
                         [Finding(h.rule_id, h.node_id, h.evidence) for h in hotspots], buffer.commit)
    report("Done", 0.96, f"RECOMMEND: {len(findings)} static signals → {len(hotspots)} hotspots"
           + (f" (LLM rejected {rstats['rejected']}, added {rstats['added']})" if use_llm else "")
           + (f" · investigator: {investigation.stats['tool_calls']} tool calls in {investigation.stats['turns']} turns"
              if investigation else "")
           + f" · {len(recommendations)} evaluator recommendations · {calls} LLM calls")

    return Understanding(buffer=buffer, graph=graph, hotspots=hotspots, recommendations=recommendations,
                         frameworks=frameworks, languages=languages, evalgraph_yaml=ev_yaml if isinstance(ev_yaml, dict) else None,
                         files_scanned=len(files), llm_calls=calls,
                         stats={"components": {k: v for k, v in cstats.items() if k not in ("scope", "ranks", "probe_targets")},
                                "cards": card_stats, "reduce": red_stats, "review": rstats,
                                "flow": {**flow.stats(), "edges_added": flow_edges},
                                "investigation": investigation.stats if investigation else None},
                         runtime=runtime, flow=flow, aimap=aimap)


def plan_runtime(buffer, symbols, comps, g, cstats, graph, use_llm, hotspots=(), proposals=()) -> dict:
    """Run plan + probe manifest + synthetic dataset for the Docker sandbox (PHASE-06B), plus probes aimed at the
    hotspots found and the investigator's proposed evaluations (PHASE-11)."""
    from ..runtime.dataset import synthesize, targeted
    from ..runtime.plan import infer_plan

    entries = [c.key for c in comps if c.type == "entrypoint"]
    plan = infer_plan(buffer, symbols, comps, g, cstats["scope"], entries)
    node_ids = {n.id for n in graph.nodes}
    probes, seen = [], set()
    for c in comps:
        if c.id not in node_ids:
            continue
        for path, qual in cstats.get("probe_targets", {}).get(c.key, [(c.path, c.qualname)]):
            if path.endswith(".py") and qual != "<module>" and (path, qual) not in seen:
                seen.add((path, qual))
                probes.append({"file": path, "qualname": qual, "node_id": c.id, "kind": c.type})
    summary = [{"id": n.id, "type": n.type, "label": n.label, "purpose": n.description} for n in graph.nodes][:30]
    dataset = synthesize(buffer, summary, use_llm=use_llm) + targeted(list(hotspots), list(proposals))         if plan.mode != "unsupported" else []
    return {"plan": plan.model_dump(), "probes": probes, "dataset": [s.model_dump() for s in dataset], "sandbox": True}
