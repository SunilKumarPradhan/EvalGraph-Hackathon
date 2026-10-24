"""Run an analyzed repo in an isolated Docker container and collect per-sample traces.

Isolation (DockerSandbox): fresh container per run (--rm), clone + install + run happen inside it, only an output
folder is mounted, all capabilities dropped except what the entrypoint needs to switch users, no-new-privileges,
memory/CPU/PID limits. The app runs as an unprivileged user and never sees the DIAL key (only the in-container
gateway does).

LocalSandbox runs the same harness on the host. It exists only to test the harness with EvalGraph's own fixtures
and is never reachable from the API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from ..core.schemas import AppOutput, GoldenSample, GraphNode, Span
from ..llm import estimate_cost
from ..settings import Settings, get_settings
from .plan import RunPlan

logger = logging.getLogger(__name__)
CONTAINER_DIR = Path(__file__).parent / "container"
HARNESS_DIR = CONTAINER_DIR / "harness"
IMAGE_PREFIX = "evalgraph-sandbox"


class SandboxError(RuntimeError):
    pass


@dataclass
class RawRun:
    out_dir: Path
    status: dict
    results: list[dict]
    spans: list[dict]
    llm_calls: list[dict]
    logs: dict[str, str]
    egress: list[dict] = field(default_factory=list)  # proxy decisions: {host, port, allowed, method}


# where an analyzed repo may connect in allowlist mode: package registries, git hosts, model files, and DIAL (added)
EGRESS_DEFAULT = ["pypi.org", "files.pythonhosted.org", "registry.npmjs.org", ".npmjs.org", "github.com",
                  "codeload.github.com", ".githubusercontent.com", "gitlab.com", "bitbucket.org", "huggingface.co",
                  ".huggingface.co", ".hf.co"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _collect(out: Path) -> RawRun:
    status = json.loads((out / "status.json").read_text(encoding="utf-8")) if (out / "status.json").exists() else {"ok": False, "error": "no status written"}
    logs = {p.stem: p.read_text(encoding="utf-8", errors="replace")[-4000:] for p in out.glob("*.log")}
    return RawRun(out, status, _read_jsonl(out / "results.jsonl"), _read_jsonl(out / "spans.jsonl"),
                  _read_jsonl(out / "llm_calls.jsonl"), logs, _read_jsonl(out / "egress.jsonl"))


def _owner(ts: float, windows: list[tuple[str, float, float]]) -> str | None:
    """Exactly one sample owns each event: the window containing it, else the closest window that started before it."""
    inside = [sid for sid, s, e in windows if s <= ts <= e]
    if inside:
        return inside[0]
    before = [(ts - e, sid) for sid, s, e in windows if s <= ts]
    return min(before)[1] if before else None


def to_outputs(raw: RawRun, samples: list[GoldenSample], nodes: list[GraphNode]) -> list[AppOutput]:
    """Assign spans and LLM calls to samples by time; attribute LLM tokens to the innermost component."""
    by_id = {s.id: s for s in samples}
    node_by_id = {n.id: n for n in nodes}
    results = [r for r in raw.results if r.get("sample_id") in by_id]
    windows = [(r["sample_id"], r["start"], r["end"]) for r in results]
    # ids beat clocks: the probe tags spans with the sample id (function mode); an LLM call belongs to the sample of
    # the span it happens inside; time windows are only the fallback (e.g. HTTP mode, calls outside any span)
    span_owner = [(s.get("sample_id") or _owner(s["start"], windows), s) for s in raw.spans]

    questions = {r["sample_id"]: by_id[r["sample_id"]].question.strip()[:60] for r in results}

    def call_sample(c: dict) -> str | None:
        # 1. content: a call whose messages contain exactly one sample's question belongs to it
        preview = c.get("preview", "")
        named = [sid for sid, q in questions.items() if q and (q in preview or json.dumps(q)[1:-1] in preview)]
        if len(named) == 1:
            return named[0]
        # 2. containment in a span already tagged with its sample, 3. time window
        hosts = [(s["start"], owner) for owner, s in span_owner if owner and s["start"] <= c["ts"] <= s["end"]]
        return max(hosts)[1] if hosts else _owner(c["ts"], windows)

    call_owner = [(call_sample(c), c) for c in raw.llm_calls]
    outputs: list[AppOutput] = []
    for r in results:
        sid = r["sample_id"]
        window = [s for owner, s in span_owner if owner == sid]
        spans: list[Span] = []
        for s in window:
            node = node_by_id.get(s["node_id"])
            output = {"text": s.get("output", "")}
            if s.get("items"):  # a returned list of chunks/documents, item by item
                output["chunks"] = [{"text": t} for t in s["items"]]
            spans.append(Span(component=s["component"], file=s["file"], kind=s["kind"] if node is None else node.type,
                              input=s.get("input") or {}, output=output,
                              latency_ms=round((s["end"] - s["start"]) * 1000, 2)))
        unattributed: list[Span] = []
        for owner, call in call_owner:
            if owner != sid:
                continue
            cost = estimate_cost(call.get("deployment", ""), call.get("tokens_in", 0), call.get("tokens_out", 0))
            inside = [i for i, s in enumerate(window) if s["start"] <= call["ts"] <= s["end"]]
            if inside:
                sp = spans[max(inside, key=lambda j: window[j]["start"])]  # innermost = latest started
                sp.tokens_in += call.get("tokens_in", 0)
                sp.tokens_out += call.get("tokens_out", 0)
                sp.cost_usd = round(sp.cost_usd + cost, 8)
                sp.model = call.get("deployment")
            else:
                unattributed.append(Span(component="llm_gateway", file="", kind="llm", input={"preview": call.get("preview", "")},
                                         output={"status": call.get("status")}, model=call.get("deployment"),
                                         latency_ms=round((call["end"] - call["ts"]) * 1000, 2),
                                         tokens_in=call.get("tokens_in", 0), tokens_out=call.get("tokens_out", 0), cost_usd=cost))
        entry = Span(component="sandbox_request", file="", kind="entrypoint", input={"question": by_id[sid].question},
                     output={"answer": r.get("answer", "")}, latency_ms=round((r["end"] - r["start"]) * 1000, 2),
                     error=r.get("error"))
        outputs.append(AppOutput(answer=r.get("answer") or "", spans=[*spans, *unattributed, entry]))
    return outputs


def _job(source: dict, plan: RunPlan, samples: list[GoldenSample], probes: list[dict], gateway: str, work: str,
         patches: list[dict] | None = None, isolated: bool = False) -> dict:
    return {"source": source, "plan": plan.model_dump(), "samples": [{"id": s.id, "question": s.question} for s in samples],
            "probes": probes, "gateway": gateway, "work": work, "patches": patches or [], "isolated": isolated,
            "timeouts": {"install": 900, "run": 1800, "start": 180, "sample": 180}}


class DockerSandbox:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @staticmethod
    def available() -> bool:
        """Docker Desktop's Resource Saver pauses the engine when idle; the first call then waits for it to resume.
        Two tries with a generous timeout, so a waking engine isn't reported as "not running".
        `docker info --format` exits 0 even when Docker Desktop reports "unable to start", so require a server version."""
        for _ in range(2):
            try:
                probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True,
                                       text=True, timeout=60)
                if probe.returncode == 0 and probe.stdout.strip() and "error" not in probe.stdout.lower():
                    return True
            except subprocess.TimeoutExpired:
                continue
            except OSError:  # no docker CLI at all
                return False
        return False

    @staticmethod
    def image_tag() -> str:
        h = hashlib.sha256()
        for p in sorted(CONTAINER_DIR.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.relative_to(CONTAINER_DIR).as_posix().encode())
                h.update(p.read_bytes())
        return f"{IMAGE_PREFIX}:{h.hexdigest()[:12]}"

    def ensure_image(self) -> str:
        tag = self.image_tag()
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
            logger.info("building sandbox image %s", tag)
            build = subprocess.run(["docker", "build", "-t", tag, str(CONTAINER_DIR)], capture_output=True, text=True, timeout=1800)
            if build.returncode != 0:
                raise SandboxError(f"docker build failed: {build.stderr[-800:]}")
            self.prune_images(keep=tag)
        return tag

    @staticmethod
    def prune_images(keep: str) -> list[str]:
        """Old sandbox images (one per harness version, ~1 GB each) are removed once a new one is built (conf_05 Q-58)."""
        listed = subprocess.run(["docker", "images", IMAGE_PREFIX, "--format", "{{.Repository}}:{{.Tag}}"],
                                capture_output=True, text=True)
        removed = []
        for ref in listed.stdout.split():
            if ref != keep and subprocess.run(["docker", "rmi", ref], capture_output=True).returncode == 0:
                removed.append(ref)  # an image still used by a running container is simply kept
        if removed:
            logger.info("removed old sandbox images: %s", removed)
        return removed

    @staticmethod
    def cache_volume(source: dict) -> str:
        """Download cache per repo: a shared cache would let one analyzed repo poison another's packages."""
        key = source.get("repo_url") or Path(source.get("local_path", "")).name.lower()
        key += f"#{source.get('ref', '')}/{source.get('subdir', '')}" if source.get("ref") or source.get("subdir") else ""
        return f"evalgraph-cache-{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def allowlist(self) -> list[str]:
        s = self.settings
        dial = urlsplit(s.dial_endpoint).hostname or ""
        extra = [h.strip() for h in s.egress_allow.split(",") if h.strip()]
        return [*EGRESS_DEFAULT, dial, *extra]

    def start_egress(self, tag: str, out: Path, name: str) -> tuple[str, str]:
        """Internal network (no route out) + an allowlist proxy that is on it and on the default bridge."""
        net, proxy = f"{name}-net", f"{name}-egress"
        if subprocess.run(["docker", "network", "create", "--internal", net], capture_output=True).returncode != 0:
            raise SandboxError("could not create the sandbox network")
        started = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", proxy, "--entrypoint", "python3", "--user", "10001",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "256m", "--pids-limit", "128",
             "-v", f"{out}:/out", tag, "/harness/egress.py", "--port", "3128", "--allow", ",".join(self.allowlist()),
             "--log", "/out/egress.jsonl"], capture_output=True, text=True)
        if started.returncode != 0 or subprocess.run(["docker", "network", "connect", "--alias", "egress", net, proxy],
                                                     capture_output=True).returncode != 0:
            self.stop_egress(net, proxy)
            raise SandboxError(f"could not start the egress proxy: {started.stderr[-300:]}")
        return net, proxy

    @staticmethod
    def stop_egress(net: str, proxy: str) -> None:
        subprocess.run(["docker", "rm", "-f", proxy], capture_output=True)
        subprocess.run(["docker", "network", "rm", net], capture_output=True)

    def command(self, tag: str, out: Path, name: str, local_path: str | None, cache: str | None = None,
                network: str | None = None) -> list[str]:
        s = self.settings
        cmd = ["docker", "run", "--rm", "--name", name,
               "--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "SETUID", "--cap-add", "SETGID",
               "--cap-add", "DAC_OVERRIDE", "--security-opt", "no-new-privileges",
               "--memory", "4g", "--cpus", "2", "--pids-limit", "512",
               "-v", f"{out}:/out",
               "-e", f"DIAL_ENDPOINT={s.dial_endpoint}", "-e", f"DIAL_API_VERSION={s.dial_api_version}",
               "-e", "DIAL_API_KEY", "-e", f"SANDBOX_CHAT_MODEL={s.sandbox_model}",
               "-e", f"SANDBOX_EMBED_MODEL={s.sandbox_embed_model}",
               "-e", f"SANDBOX_MOCK_LLM={'1' if s.mock_llm else '0'}",
               "-e", f"SANDBOX_HONOR_MODEL={'1' if s.sandbox_honor_model else '0'}"]
        if local_path:
            cmd += ["-v", f"{local_path}:/src:ro"]
        if cache:
            cmd += ["-v", f"{cache}:/cache"]
        if network:  # allowlist mode: only the internal network; the proxy is the way out
            cmd += ["--network", network, "-e", "EVALGRAPH_PROXY=http://egress:3128"]
        return cmd + [tag]

    def run(self, source: dict, plan: RunPlan, samples: list[GoldenSample], probes: list[dict], timeout: int = 3600,
            patches: list[dict] | None = None, secrets: dict[str, str] | None = None, isolated: bool = False) -> RawRun:
        if not self.available():
            raise SandboxError("Docker daemon is not running; start Docker to run repositories in the sandbox.")
        tag = self.ensure_image()
        out = Path(tempfile.mkdtemp(prefix="evalgraph-run-"))
        local = source.get("local_path")
        job_source = {"path": "/src"} if local else {k: source[k] for k in ("repo_url", "ref", "subdir") if source.get(k)}
        (out / "job.json").write_text(json.dumps(_job(job_source, plan, samples, probes, "http://127.0.0.1:8919", "/work", patches, isolated)),
                                      encoding="utf-8")
        name = f"evalgraph-{out.name[-8:].lower()}"
        env = {**os.environ, "DIAL_API_KEY": self.settings.dial_api_key or ""}  # passed by name, not on the command line
        env.update({f"APPSECRET_{k}": v for k, v in (secrets or {}).items()})
        net = proxy = None
        if self.settings.sandbox_egress == "allowlist":
            net, proxy = self.start_egress(tag, out, name)
        try:
            cache = self.cache_volume(source) if self.settings.sandbox_cache else None
            cmd = self.command(tag, out, name, local, cache, net)
            for k in secrets or {}:  # by name only: values never appear on the command line
                cmd[-1:-1] = ["-e", f"APPSECRET_{k}"]
            subprocess.run(cmd, env=env, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            raise SandboxError(f"sandbox run exceeded {timeout}s and was killed")
        finally:
            if net:
                self.stop_egress(net, proxy)
        return _collect(out)


class LocalSandbox:
    """Runs the harness on the host. For EvalGraph's own trusted test fixtures only."""

    def run(self, source: dict, plan: RunPlan, samples: list[GoldenSample], probes: list[dict],
            upstream: str = "http://127.0.0.1:9", mock: bool = True, patches: list[dict] | None = None,
            secrets: dict[str, str] | None = None, isolated: bool = False) -> RawRun:
        sys.path.insert(0, str(HARNESS_DIR))
        import gateway  # type: ignore  # the same stdlib gateway that runs in the container

        out = Path(tempfile.mkdtemp(prefix="evalgraph-local-"))
        cfg = gateway.Config(upstream, "test-key", "2024-02-01", "gpt-4o-mini", "text-embedding-3-small",
                             str(out / "llm_calls.jsonl"), mock=mock)
        server = gateway.serve(cfg, 0)
        port = server.server_address[1]
        try:
            work = out / "work"
            job = _job({"path": source["local_path"]}, plan, samples, probes, f"http://127.0.0.1:{port}", str(work), patches,
                       isolated)
            (out / "job.json").write_text(json.dumps(job), encoding="utf-8")
            env = dict(os.environ)
            if secrets:  # the container entrypoint writes this file; on the host the test writes it
                (out / "app_secrets.json").write_text(json.dumps(secrets), encoding="utf-8")
                env["EVALGRAPH_APP_SECRETS"] = str(out / "app_secrets.json")
            subprocess.run([sys.executable, str(HARNESS_DIR / "harness.py"), str(out / "job.json")], timeout=900,
                           capture_output=True, env=env)
            return _collect(out)
        finally:
            server.shutdown()


def cleanup(raw: RawRun) -> None:
    shutil.rmtree(raw.out_dir, ignore_errors=True)
