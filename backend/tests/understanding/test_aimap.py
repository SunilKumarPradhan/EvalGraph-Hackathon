"""AI Map (PHASE-11): the whole analysis as compact, deterministic, line-referenced text."""

import re

from fastapi.testclient import TestClient

from evalgraph.api.main import app
from evalgraph.understanding.aimap import render

from .conftest import ZOO
from .test_api_analyze import _wait


def test_archer_map_is_small_and_complete(archer):
    text = archer.aimap
    assert len(text) / 4 <= 8000, f"~{len(text) // 4} tokens"     # the old slice budget alone was ~22k tokens
    for fact in ("DEFAULT_TOP_K = 10", "USE_RERANKER = False", "GUARDRAIL_ENABLED = False"):
        assert fact in text
    for section in ("## COMPONENTS", "## PIPELINE", "## PROMPT RECIPES", "## DATA FLOWS", "## STATIC FINDINGS"):
        assert section in text
    assert all(f"[{n.id}]" in text for n in archer.graph.nodes)


def test_map_references_are_real_lines(archer):
    for path, line in re.findall(r"@?([\w./-]+\.(?:py|yaml|ts|js)):(\d+)", archer.aimap):
        assert path in archer.buffer.files and int(line) <= len(archer.buffer.files[path].splitlines()), (path, line)


def test_map_is_deterministic(archer):
    from evalgraph.understanding.miners import Finding

    findings = [Finding(h.rule_id, h.node_id, h.evidence) for h in archer.hotspots]
    again = render(archer.buffer.name, archer.frameworks, archer.languages, archer.graph, archer.flow, findings,
                   archer.buffer.commit)
    assert again == archer.aimap


def test_aimap_endpoint():
    client = TestClient(app)
    job = _wait(client.post("/api/analyze", json={"local_path": str(ZOO / "openai_agent")}).json()["job_id"])
    r = client.get(f"/api/analyses/{job['result_id']}/aimap")
    assert r.status_code == 200 and r.text.startswith("# AI MAP") and "INDIRECT_INJECTION" in r.text
    assert client.get("/api/analyses/nope/aimap").status_code == 404
