"""Shared contracts. Mirror of docs/02-CONTRACTS.md and frontend/src/types/contracts.ts.

Change all three together. Add fields as optional; never rename.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- §1 enums

NodeType = Literal[
    "entrypoint", "agent", "router", "llm", "prompt", "retriever", "vector_db",
    "embedding", "reranker", "tool", "memory", "guardrail", "external_api",
    "parser", "dataset", "other",
]
EdgeKind = Literal["calls", "uses", "reads", "writes", "feeds", "routes_to"]
Severity = Literal["high", "medium", "low"]
Status = Literal["green", "amber", "red", "error", "na"]
JobStatus = Literal["queued", "running", "done", "error"]
JobKind = Literal["analyze", "run"]
EvalCategory = Literal["rag", "agent", "llm", "safety", "ops"]
EvalMethod = Literal["llm_judge", "deterministic"]
Unit = Literal["score", "ms", "usd", "tokens"]

ConfigValue = str | int | float | bool | None

# ---------------------------------------------------------------- §2 analysis


class Evidence(BaseModel):
    file: str
    line: int | None = None
    snippet: str = ""
    fact: str


class GraphNode(BaseModel):
    id: str
    type: NodeType
    label: str
    file: str | None = None
    symbol: str | None = None
    provider: str | None = None
    config: dict[str, ConfigValue] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 1.0
    description: str | None = None          # one-line purpose (Component Card)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str | None = None
    flow_labels: list[str] | None = None    # data proven to flow along this edge: retrieved, memory, llm_output, ...


class ApplicationGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class PatchEdit(BaseModel):
    """One whole-line edit of a rule-based fix: the line as analyzed and the same line with one literal changed."""
    file: str
    line: int
    before: str
    after: str
    summary: str = ""                       # "DEFAULT_TOP_K 10 → 4"
    span: list[int] | None = None           # [start, end) of the changed literal in `after` (single-literal edits)
    sweep: list[str] = Field(default_factory=list)  # values worth sweeping for this literal, e.g. ["2", "4", "6", "8"]


class Hotspot(BaseModel):
    id: str
    rule_id: str
    node_id: str
    severity: Severity
    title: str
    why: str
    evidence: list[Evidence]
    impacts: list[str] = Field(default_factory=list)
    recommended_evaluators: list[str] = Field(default_factory=list)
    suggested_variant: str | None = None
    source: Literal["static", "llm", "static+llm"] = "static"
    patch: list[PatchEdit] = Field(default_factory=list)  # rule-based literal fix, if one exists (sandbox runs try it)


class Recommendation(BaseModel):
    node_id: str
    evaluator_id: str
    priority: Severity
    rationale: str


class AnalysisSource(BaseModel):
    repo_url: str | None = None
    local_path: str | None = None
    commit: str | None = None
    name: str


class AnalysisSummary(BaseModel):
    language: str
    files_scanned: int
    frameworks: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    eval_gaps: int = 0
    runnable: bool = False
    adapter: str | None = None
    variants: list[str] = Field(default_factory=list)
    headline: str = ""
    languages: list[str] = Field(default_factory=list)
    llm_calls: int = 0                       # LLM calls spent on this analysis
    truncated: bool = False                  # the file-count limit cut the repo short (Q-05)
    skipped_files: int = 0                   # files over the per-file / total size limits, not analyzed


class RuntimeInfo(BaseModel):
    """How the sandbox will run this repo (PHASE-06B). Stored with the analysis; editable later."""
    plan: dict[str, Any] = Field(default_factory=dict)          # runtime.plan.RunPlan
    probes: list[dict[str, Any]] = Field(default_factory=list)  # [{file, qualname, node_id, kind}]
    dataset: list["GoldenSample"] = Field(default_factory=list) # synthetic samples when the repo ships none
    sandbox: bool = True                                         # runs inside the Docker sandbox


class AnalysisResult(BaseModel):
    id: str
    created_at: str
    source: AnalysisSource
    summary: AnalysisSummary
    graph: ApplicationGraph
    hotspots: list[Hotspot] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    runtime: RuntimeInfo | None = None
    mock: bool = False
    replayed: bool = False
    aimap: str | None = None                 # compact line-referenced AI Map (PHASE-11), GET .../aimap serves it
    attachments: list["Attachment"] = Field(default_factory=list)   # the evaluation setup, shared by everyone (Q-19)
    dismissed_hotspots: list[str] = Field(default_factory=list)     # hotspot ids a user marked as not relevant (Q-15)


# ---------------------------------------------------------------- §3 evaluators


class Thresholds(BaseModel):
    green: float
    amber: float


class EvaluatorSpec(BaseModel):
    id: str
    name: str
    category: EvalCategory
    description: str
    applies_to: list[NodeType]
    method: EvalMethod
    unit: Unit
    higher_is_better: bool = True
    thresholds: Thresholds
    needs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- §4 runs


class Attachment(BaseModel):
    node_id: str
    evaluator_id: str


class RunRequest(BaseModel):
    analysis_id: str
    attachments: list[Attachment]
    variant: str = "baseline"
    limit: int | None = None
    categories: list[str] | None = None
    patches: list[str] | None = None        # hotspot ids whose patches are applied in the sandbox before running
    isolated: bool = False                  # sandbox function runs: one fresh process per sample (no shared memory)
    patch_values: dict[str, str] | None = None  # hotspot id -> value for its sweepable literal (that edit only)


class SampleResult(BaseModel):
    sample_id: str
    node_id: str
    evaluator_id: str
    value: float | None = None
    reason: str = ""
    error: str | None = None


class NodeEvalResult(BaseModel):
    node_id: str
    evaluator_id: str
    value: float | None = None
    unit: Unit
    status: Status
    n: int = 0
    worst_samples: list[str] = Field(default_factory=list)
    reason: str = ""                         # why na/error, or what was excluded


class Bottleneck(BaseModel):
    node_id: str
    evaluator_id: str
    value: float | None = None
    status: Status
    contributors: list[Evidence] = Field(default_factory=list)
    explanation: str = ""
    suggested_experiment: str | None = None
    suggested_variant: str | None = None
    suggested_patch: list[str] = Field(default_factory=list)  # hotspot ids with a literal fix (repos without variants)


class RunResult(BaseModel):
    id: str
    analysis_id: str
    variant: str
    created_at: str
    dataset: dict[str, Any] = Field(default_factory=dict)
    results: list[NodeEvalResult] = Field(default_factory=list)
    samples: list[SampleResult] = Field(default_factory=list)
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    overall: float | None = None
    duration_s: float = 0.0
    models: dict[str, str] = Field(default_factory=dict)
    mock: bool = False
    replayed: bool = False
    patches: list[str] = Field(default_factory=list)          # hotspot ids applied in this run


class SweepRequest(BaseModel):
    analysis_id: str
    hotspot_id: str
    values: list[str] | None = None         # default: the edit's suggested sweep values
    attachments: list[Attachment]
    limit: int | None = None
    isolated: bool = False
    base_run_id: str | None = None          # reuse an existing as-is run as the baseline


class SweepResult(BaseModel):
    id: str
    analysis_id: str
    hotspot_id: str
    key: str                                # the swept literal, e.g. "TOP_K"
    original: str                           # its value in the code
    base_run_id: str
    runs: dict[str, str] = Field(default_factory=dict)   # value -> run id
    created_at: str


class RunComparison(BaseModel):
    base_run_id: str
    other_run_id: str
    rows: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------- §5 jobs


class Job(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus = "queued"
    stage: str = "Queued"
    progress: float = 0.0
    log: list[str] = Field(default_factory=list)
    result_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------- §7 runtime spans


class Span(BaseModel):
    component: str
    file: str
    kind: NodeType
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class AppOutput(BaseModel):
    answer: str
    route: str | None = None
    blocked: bool = False
    nodes: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)


# ---------------------------------------------------------------- §9 golden samples


class GoldenSample(BaseModel):
    id: str
    category: str
    question: str
    reference_answer: str = ""
    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_route: str | None = None
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] | None = None
    expected_numeric: dict[str, float] | None = None
    expected_findings: list[str] | None = None
    expected_behavior: str = "answer"
    history: list[dict[str, str]] = Field(default_factory=list)
    probes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- API helpers


class RuntimeUpdate(BaseModel):
    """User edits to how the sandbox runs a repo (Run setup drawer). Plan keys are merged into the inferred plan."""
    plan: dict[str, Any] | None = None
    dataset: list[GoldenSample] | None = None


class AnalyzeRequest(BaseModel):
    repo_url: str | None = None
    local_path: str | None = None


class JobRef(BaseModel):
    job_id: str


class Health(BaseModel):
    ok: bool = True
    mock: bool
    replay: bool
    model: str
    judge_model: str


RuntimeInfo.model_rebuild()
AnalysisResult.model_rebuild()
