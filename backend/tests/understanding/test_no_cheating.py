"""CUDA Archer gets no special treatment: its hints can be removed and its fixes are recognised."""

import re
import shutil

from .conftest import ARCHER, run

CORE = {"entrypoint", "agent", "guardrail", "memory", "router", "parser", "retriever", "reranker", "vector_db",
        "embedding", "llm", "tool", "external_api"}


def _copy(tmp_path):
    shutil.copytree(ARCHER / "cuda_archer", tmp_path / "cuda_archer", ignore=shutil.ignore_patterns("__pycache__", "static"))
    shutil.copytree(ARCHER / "eval", tmp_path / "eval")
    return tmp_path


def _strip_traced(root):
    for p in (root / "cuda_archer").rglob("*.py"):
        out, lines, i = [], p.read_text(encoding="utf-8").splitlines(), 0
        while i < len(lines):
            if lines[i].lstrip().startswith("@traced("):
                depth = 0
                while True:
                    depth += lines[i].count("(") - lines[i].count(")")
                    i += 1
                    if depth <= 0:
                        break
                continue
            out.append(lines[i])
            i += 1
        p.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_core_components(archer):
    assert CORE <= {n.type for n in archer.graph.nodes}


def test_decorators_do_not_matter(archer, tmp_path):
    root = _copy(tmp_path)
    _strip_traced(root)
    assert not any(ln.lstrip().startswith("@traced(") for p in (root / "cuda_archer").rglob("*.py")
                   for ln in p.read_text(encoding="utf-8").splitlines())
    stripped = run(root)
    assert {n.id for n in stripped.graph.nodes} == {n.id for n in archer.graph.nodes}
    assert {h.rule_id for h in stripped.hotspots} == {h.rule_id for h in archer.hotspots}


IMPROVED = {"DEFAULT_TOP_K": "12", "USE_RERANKER": "True", "DEFAULT_CHUNK_SIZE": "400", "DEFAULT_CHUNK_OVERLAP": "80",
            "DEFAULT_TEMPERATURE": "0.1", "STRICT_GROUNDING": "True", "MAX_CONTEXT_CHARS": "2400",
            "ENABLE_QUERY_REWRITE": "False", "ROUTER_MODEL": '"small"', "ROUTER_FEW_SHOT": "True",
            "ROBUST_PARSING": "True", "FUZZY_MATCH": "True", "VALIDATE_ARGS": "True", "REPAIR_TOOL_CALLS": "True",
            "HTTP_TIMEOUT": "3", "CACHE_MATRIX": "True", "USE_LINTER": "True", "GUARDRAIL_ENABLED": "True",
            "MEMORY_WINDOW": "3"}


def test_fixed_copy_loses_config_hotspots(tmp_path):
    root = _copy(tmp_path)
    for py in (root / "cuda_archer").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for name, value in IMPROVED.items():
            text = re.sub(rf"^({name}(?:\s*:[^=\n]+)?\s*=\s*)[^#\n]+", rf"\g<1>{value}  ", text, flags=re.M)
        py.write_text(text, encoding="utf-8")
    fired = {h.rule_id for h in run(root).hotspots}
    config_rules = {"RAG_TOPK_NO_RERANK", "RAG_BIG_CHUNKS", "LLM_HIGH_TEMP", "LLM_UNGROUNDED", "LLM_UNBOUNDED_CONTEXT",
                    "EXTRA_LLM_HOP", "BIG_MODEL_FOR_CLASSIFY", "ROUTER_NO_SCHEMA", "TOOL_BRITTLE_LOOKUP",
                    "TOOL_UNVALIDATED", "EXT_NO_TIMEOUT", "NO_GUARDRAIL", "MEMORY_UNBOUNDED"}
    assert not fired & config_rules, fired & config_rules
