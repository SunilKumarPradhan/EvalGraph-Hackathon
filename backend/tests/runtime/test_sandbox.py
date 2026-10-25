"""Sandbox runtime: plans, harness + probe + gateway end to end (host run of EvalGraph's own fixtures), Docker hardening."""

import subprocess
from pathlib import Path

import pytest

from evalgraph.core.schemas import AnalysisResult, GoldenSample
from evalgraph.runtime.adapter import SandboxAdapter
from evalgraph.runtime.plan import RunPlan
from evalgraph.runtime.sandbox import DockerSandbox, LocalSandbox, cleanup, to_outputs
from evalgraph.settings import Settings
from evalgraph.understanding.pipeline import understand

ZOO = Path(__file__).resolve().parents[1] / "understanding" / "zoo"
FIX = Path(__file__).parent / "fixtures"
ARCHER = Path(__file__).resolve().parents[3].parent / "cuda-archer"


def _u(path: Path):
    return understand({"local_path": str(path.resolve())}, lambda *a, **k: None)


@pytest.mark.parametrize("app,mode,target", [
    ("openai_agent", "python_function", ("agent", "run_agent", "task")),
    ("langchain_rag", "python_function", ("app", "answer", "question")),
    ("ts_nextjs_rag", "http", "/api/chat"),
    ("vercel_ai_sdk", "http", "/chat"),
    ("notebook_rag", "unsupported", None),
])
def test_plans_for_any_repo(app, mode, target):
    plan = RunPlan(**_u(ZOO / app).runtime["plan"])
    assert plan.mode == mode
    if mode == "python_function":
        assert (plan.module, plan.function, plan.question_param) == target
    elif mode == "http":
        assert plan.path == target and plan.start_cmd


def test_plan_for_cuda_archer_matches_its_own_entry():
    plan = RunPlan(**_u(ARCHER).runtime["plan"])
    assert (plan.module, plan.function, plan.question_param) == ("cuda_archer.app", "run", "question")


def test_end_to_end_on_host_harness():
    u = _u(ZOO / "openai_agent")
    samples = [GoldenSample(**s) for s in u.runtime["dataset"]]
    assert any(s.expected_behavior == "refuse" for s in samples)  # safety probes are part of every synthetic set
    raw = LocalSandbox().run({"local_path": str((ZOO / "openai_agent").resolve())},
                             RunPlan(**{**u.runtime["plan"], "install": "none"}), samples, u.runtime["probes"])
    assert raw.status["ok"], raw.logs
    outs = to_outputs(raw, samples, u.graph.nodes)
    assert len(outs) == len(samples)
    for o in outs:
        traced = [s for s in o.spans if s.component == "run_agent"]
        assert len(traced) == 1 and traced[0].tokens_in > 0 and traced[0].kind in ("llm", "agent")
        assert o.answer.startswith("[MOCK]") and not any(s.component == "llm_gateway" for s in o.spans)
    cleanup(raw)


def test_app_never_sees_dial_key(monkeypatch):
    monkeypatch.setenv("DIAL_API_KEY", "SUPER-SECRET")
    plan = RunPlan(mode="python_function", module="app", function="answer", question_param="question", install="none")
    raw = LocalSandbox().run({"local_path": str((FIX / "env_app").resolve())}, plan,
                             [GoldenSample(id="s1", category="t", question="hi")], [])
    assert raw.status["ok"], raw.logs
    import json

    seen = json.loads(raw.results[0]["answer"])
    assert "DIAL_API_KEY" not in seen["visible"] and "SUPER-SECRET" not in raw.results[0]["answer"]
    assert seen["openai_key"].startswith("sk-evalgraph-sandbox") and seen["base_url"].endswith("/v1")
    assert seen["reply"].startswith("[MOCK]")
    cleanup(raw)


def test_docker_command_is_hardened(tmp_path):
    s = Settings(EVALGRAPH_MOCK_LLM=True, DIAL_API_KEY="SECRET-VALUE")
    cmd = DockerSandbox(s).command("evalgraph-sandbox:abc", tmp_path, "evalgraph-x", None)
    joined = " ".join(cmd)
    assert "--rm" in cmd and "--cap-drop ALL" in joined and "no-new-privileges" in joined
    assert "--memory 4g" in joined and "--pids-limit 512" in joined
    assert "SECRET-VALUE" not in joined                      # key passed by env name, never on the command line
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert mounts == [f"{tmp_path}:/out"]                     # only the output folder is shared with the host
    with_src = DockerSandbox(s).command("t", tmp_path, "n", "C:/repo")
    assert "C:/repo:/src:ro" in with_src                      # a local repo is mounted read-only


def test_image_tag_tracks_container_sources():
    assert DockerSandbox.image_tag().startswith("evalgraph-sandbox:") and DockerSandbox.image_tag() == DockerSandbox.image_tag()


def test_adapter_requires_runnable_plan():
    from evalgraph import storage
    from evalgraph.analysis import run_analysis

    aid = run_analysis({"local_path": str((ZOO / "notebook_rag").resolve())}, lambda *a, **k: None)
    a = storage.load("analyses", aid, AnalysisResult)
    assert a.summary.runnable is False
    with pytest.raises(Exception):
        SandboxAdapter(a)
    aid = run_analysis({"local_path": str((ZOO / "openai_agent").resolve())}, lambda *a, **k: None)
    b = storage.load("analyses", aid, AnalysisResult)
    assert b.summary.runnable and b.summary.adapter == "sandbox" and b.summary.variants == ["as-is"]


DOCKER = DockerSandbox.available()


@pytest.mark.skipif(not DOCKER, reason="Docker daemon not running: sandbox end-to-end test skipped (run on a Docker host)")
def test_docker_end_to_end_isolated():
    from evalgraph import storage
    from evalgraph.analysis import run_analysis

    aid = run_analysis({"local_path": str((ZOO / "openai_agent").resolve())}, lambda *a, **k: None)
    adapter = SandboxAdapter(storage.load("analyses", aid, AnalysisResult))
    outs = adapter.run_batch(adapter.load_dataset()[:2])
    assert len(outs) == 2 and all(o.answer for o in outs)
    assert any(s.component == "run_agent" and s.tokens_in > 0 for o in outs for s in o.spans)
    # nothing of the run is left behind on the host and no container keeps running
    ps = subprocess.run(["docker", "ps", "--filter", "name=evalgraph-", "-q"], capture_output=True, text=True).stdout.strip()
    assert ps == ""


def test_app_specific_llm_env_and_imports_are_inferred():
    from evalgraph.runtime.plan import infer_llm_env, infer_requirements
    from evalgraph.understanding.buffer import RepoBuffer

    buf = RepoBuffer(name="x", files={
        "app.py": 'import os, json\nimport requests\nfrom sklearn import svm\nimport helpers\n'
                  'KEY = os.getenv("MYAPP_LLM_KEY")\nURL = os.environ["MYAPP_OPENAI_BASE_URL"]\nDB = os.getenv("QDRANT_URL")\n',
        "helpers.py": "X = 1\n",
        ".env.example": "DIAL_ENDPOINT=https://example\nDIAL_API_VERSION=2024-02-01\n",
    })
    assert infer_llm_env(buf) == {"DIAL_API_VERSION": "version", "DIAL_ENDPOINT": "endpoint",
                                  "MYAPP_LLM_KEY": "key", "MYAPP_OPENAI_BASE_URL": "base"}  # QDRANT_URL untouched
    assert infer_requirements(buf, {}) == ["requests", "scikit-learn"]  # stdlib and local modules excluded


def test_harness_patch_safety(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("harness_mod", Path(__file__).resolve().parents[2]
                                                  / "evalgraph" / "runtime" / "container" / "harness" / "harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    repo, out = tmp_path / "repo", tmp_path / "out"
    repo.mkdir(), out.mkdir()
    (repo / "app.py").write_bytes(b"A = 1\r\nTOP_K = 10\r\n")
    harness.apply_patches([{"file": "app.py", "line": 2, "before": "TOP_K = 10", "after": "TOP_K = 4"}], str(repo), str(out))
    assert (repo / "app.py").read_bytes() == b"A = 1\r\nTOP_K = 4\r\n"  # line endings kept
    with pytest.raises(RuntimeError, match="changed since the analysis"):
        harness.apply_patches([{"file": "app.py", "line": 2, "before": "TOP_K = 10", "after": "TOP_K = 3"}], str(repo), str(out))
    with pytest.raises(RuntimeError, match="escapes the repo"):
        harness.apply_patches([{"file": "../x.py", "line": 1, "before": "", "after": ""}], str(repo), str(out))


def test_patched_local_run_returns_item_level_chunks():
    """Returned chunk lists reach the evaluators item by item, and a top-k patch visibly changes them."""
    app = Path(__file__).resolve().parent / "fixtures" / "rag_app"
    u = understand({"local_path": str(app)}, lambda *a, **k: None)
    plan = RunPlan(**{**u.runtime["plan"], "install": "none"})
    samples = [GoldenSample(id="s1", category="domain", question="How often should I descale?")]
    topk = next(h for h in u.hotspots if h.rule_id == "RAG_TOPK_NO_RERANK")
    counts = []
    for patches in (None, [e.model_dump() for e in topk.patch]):
        raw = LocalSandbox().run({"local_path": str(app)}, plan, samples, u.runtime["probes"], patches=patches)
        assert raw.status["ok"], raw.logs
        out = to_outputs(raw, samples, u.graph.nodes)[0]
        retr = next(s for s in out.spans if s.component == "retrieve")
        counts.append(len(retr.output.get("chunks", [])))
        cleanup(raw)
    assert counts == [10, 4]


def test_cache_volume_is_per_repo(tmp_path):
    a = DockerSandbox.cache_volume({"repo_url": "https://github.com/x/app"})
    b = DockerSandbox.cache_volume({"repo_url": "https://github.com/y/app"})
    assert a.startswith("evalgraph-cache-") and a != b and a == DockerSandbox.cache_volume({"repo_url": "https://github.com/x/app"})
    s = Settings(EVALGRAPH_MOCK_LLM=True)
    cmd = DockerSandbox(s).command("t", tmp_path, "n", None, a)
    assert f"{a}:/cache" in cmd  # a named volume, not a host folder


def test_app_secrets_reach_the_app_but_never_the_dial_key(monkeypatch):
    monkeypatch.setenv("DIAL_API_KEY", "SUPER-SECRET")
    plan = RunPlan(mode="python_function", module="app", function="answer", question_param="question", install="none")
    raw = LocalSandbox().run({"local_path": str((FIX / "env_app").resolve())}, plan,
                             [GoldenSample(id="s1", category="t", question="hi")], [],
                             secrets={"SEARCH_API_KEY": "search-123", "OPENAI_API_KEY": "user-tried-to-override"})
    assert raw.status["ok"], raw.logs
    import json

    seen = json.loads(raw.results[0]["answer"])
    assert "SEARCH_API_KEY" in seen["visible"] and "DIAL_API_KEY" not in seen["visible"]
    assert seen["openai_key"].startswith("sk-evalgraph-sandbox")  # an app secret can't replace the gateway's dummy key
    cleanup(raw)


def test_app_env_and_secret_storage(tmp_path):
    from evalgraph.runtime.plan import infer_app_env
    from evalgraph.runtime.secrets import load_secrets, save_secrets
    from evalgraph.understanding.buffer import RepoBuffer

    buf = RepoBuffer(name="x", files={"a.py": 'import os\nT = os.getenv("TAVILY_API_KEY")\nP = os.environ["PORT"]\nK = os.getenv("OPENAI_API_KEY")\n',
                                      "s.ts": "const u = process.env.PINECONE_URL;"})
    assert infer_app_env(buf, {}) == ["PINECONE_URL", "TAVILY_API_KEY"]  # LLM settings and PORT are not app secrets
    assert save_secrets("a_secret01", {"TAVILY_API_KEY": "t"}) == ["TAVILY_API_KEY"]
    assert save_secrets("a_secret01", {"TAVILY_API_KEY": None}) == [] and load_secrets("a_secret01") == {}
    with pytest.raises(ValueError):
        save_secrets("a_secret01", {"DIAL_API_KEY": "nope"})


@pytest.mark.skipif(not DOCKER, reason="Docker daemon not running")
def test_docker_app_secrets_isolated():
    import json

    plan = RunPlan(mode="python_function", module="app", function="answer", question_param="question",
                   inferred_requirements=["openai"])
    raw = DockerSandbox(Settings(EVALGRAPH_MOCK_LLM=True, DIAL_API_KEY="SUPER-SECRET", EVALGRAPH_SANDBOX_CACHE=False)).run(
        {"local_path": str((FIX / "env_app").resolve())}, plan, [GoldenSample(id="s1", category="t", question="hi")], [],
        secrets={"SEARCH_API_KEY": "search-123"})
    try:
        assert raw.status["ok"], raw.logs
        seen = json.loads(raw.results[0]["answer"])
        assert "SEARCH_API_KEY" in seen["visible"] and not any(k.startswith(("DIAL", "APPSECRET")) for k in seen["visible"])
    finally:
        cleanup(raw)


def test_isolated_samples_do_not_share_memory():
    plan = RunPlan(mode="python_function", module="app", function="chat", question_param="message", install="none")
    samples = [GoldenSample(id=f"s{i}", category="t", question=f"hello {i}") for i in range(3)]
    answers = {}
    for isolated in (False, True):
        raw = LocalSandbox().run({"local_path": str((FIX / "memory_app").resolve())}, plan, samples, [], isolated=isolated)
        assert raw.status["ok"], raw.logs
        answers[isolated] = [r["answer"] for r in sorted(raw.results, key=lambda r: r["sample_id"])]
        cleanup(raw)
    assert answers[False] == ["turn 1", "turn 2", "turn 3"]  # shared process: memory carries over
    assert answers[True] == ["turn 1", "turn 1", "turn 1"]   # isolated: every sample starts fresh


@pytest.mark.skipif(not DOCKER, reason="Docker daemon not running")
def test_docker_egress_allowlist():
    import json

    plan = RunPlan(mode="python_function", module="app", function="answer", question_param="question", install="none")
    s = Settings(EVALGRAPH_MOCK_LLM=True, EVALGRAPH_SANDBOX_CACHE=False)
    raw = DockerSandbox(s).run({"local_path": str((FIX / "net_app").resolve())}, plan,
                               [GoldenSample(id="s1", category="t", question="hi")], [])
    try:
        assert raw.status["ok"], raw.logs
        seen = json.loads(raw.results[0]["answer"])
        assert seen["pypi"].startswith("ok") and seen["example"].startswith("blocked"), seen
        assert {"example.com"} <= {e["host"] for e in raw.egress if not e["allowed"]}
    finally:
        cleanup(raw)
    nets = subprocess.run(["docker", "network", "ls", "--filter", "name=evalgraph-", "-q"], capture_output=True, text=True)
    assert nets.stdout.strip() == ""  # the run's network and proxy are removed afterwards
