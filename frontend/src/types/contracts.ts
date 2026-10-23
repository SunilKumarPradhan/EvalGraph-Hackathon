// Mirror of backend/evalgraph/core/schemas.py and docs/02-CONTRACTS.md. Change all three together.

// §1 enums
export type NodeType =
  | "entrypoint" | "agent" | "router" | "llm" | "prompt" | "retriever" | "vector_db"
  | "embedding" | "reranker" | "tool" | "memory" | "guardrail" | "external_api"
  | "parser" | "dataset" | "other";
export type EdgeKind = "calls" | "uses" | "reads" | "writes" | "feeds" | "routes_to";
export type Severity = "high" | "medium" | "low";
export type Status = "green" | "amber" | "red" | "error" | "na";
export type JobStatus = "queued" | "running" | "done" | "error";
export type JobKind = "analyze" | "run";
export type EvalCategory = "rag" | "agent" | "llm" | "safety" | "ops";
export type EvalMethod = "llm_judge" | "deterministic";
export type Unit = "score" | "ms" | "usd" | "tokens";
export type ConfigValue = string | number | boolean | null;

// §2 analysis
export interface Evidence {
  file: string;
  line: number | null;
  snippet: string;
  fact: string;
}

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  file: string | null;
  symbol: string | null;
  provider: string | null;
  config: Record<string, ConfigValue>;
  evidence: Evidence[];
  confidence: number;
  description?: string | null;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
  label: string | null;
  flow_labels?: string[] | null; // PHASE-11: data proven to flow along this edge (retrieved, memory, llm_output, ...)
}

export interface ApplicationGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface PatchEdit {
  file: string;
  line: number;
  before: string;
  after: string;
  summary: string;
  span?: number[] | null;
  sweep?: string[];
}

export interface Hotspot {
  id: string;
  rule_id: string;
  node_id: string;
  severity: Severity;
  title: string;
  why: string;
  evidence: Evidence[];
  impacts: string[];
  recommended_evaluators: string[];
  suggested_variant: string | null;
  patch?: PatchEdit[];
  source?: "static" | "llm" | "static+llm";
}

export interface Recommendation {
  node_id: string;
  evaluator_id: string;
  priority: Severity;
  rationale: string;
}

export interface AnalysisSource {
  repo_url: string | null;
  local_path: string | null;
  commit: string | null;
  name: string;
}

export interface AnalysisSummary {
  truncated?: boolean;
  skipped_files?: number;
  language: string;
  files_scanned: number;
  frameworks: string[];
  counts: Record<string, number>;
  eval_gaps: number;
  runnable: boolean;
  adapter: string | null;
  variants: string[];
  headline: string;
  languages?: string[];
  llm_calls?: number;
}

export interface RuntimeInfo {
  plan: Record<string, unknown>;
  probes: Record<string, unknown>[];
  dataset: GoldenSample[];
  sandbox: boolean;
}

/** §9 golden sample (the fields the UI reads and edits; the rest are optional labels). */
export interface GoldenSample {
  id: string;
  category: string;
  question: string;
  expected_behavior?: string;
  reference_answer?: string;
  probes?: string[];
}

export interface RuntimeUpdate {
  plan?: Record<string, unknown> | null;
  dataset?: GoldenSample[] | null;
}

export interface AnalysisResult {
  id: string;
  created_at: string;
  source: AnalysisSource;
  summary: AnalysisSummary;
  graph: ApplicationGraph;
  hotspots: Hotspot[];
  recommendations: Recommendation[];
  runtime?: RuntimeInfo | null;
  mock: boolean;
  replayed: boolean;
  attachments?: Attachment[];
  dismissed_hotspots?: string[];
  aimap?: string | null; // PHASE-11: compact line-referenced AI Map (also GET /analyses/{id}/aimap)
}

// §3 evaluators
export interface Thresholds {
  green: number;
  amber: number;
}

export interface EvaluatorSpec {
  id: string;
  name: string;
  category: EvalCategory;
  description: string;
  applies_to: NodeType[];
  method: EvalMethod;
  unit: Unit;
  higher_is_better: boolean;
  thresholds: Thresholds;
  needs: string[];
}

// §4 runs
export interface Attachment {
  node_id: string;
  evaluator_id: string;
}

export interface RunRequest {
  analysis_id: string;
  attachments: Attachment[];
  variant?: string;
  limit?: number | null;
  categories?: string[] | null;
  patches?: string[] | null;
  isolated?: boolean;
  patch_values?: Record<string, string> | null;
}

export interface SampleResult {
  sample_id: string;
  node_id: string;
  evaluator_id: string;
  value: number | null;
  reason: string;
  error: string | null;
}

export interface NodeEvalResult {
  node_id: string;
  evaluator_id: string;
  value: number | null;
  unit: Unit;
  status: Status;
  n: number;
  worst_samples: string[];
  reason?: string; // why the status is na/error, or which samples were excluded
}

export interface Bottleneck {
  node_id: string;
  evaluator_id: string;
  value: number | null;
  status: Status;
  contributors: Evidence[];
  explanation: string;
  suggested_experiment: string | null;
  suggested_variant: string | null;
  suggested_patch?: string[];
  source?: "static" | "llm" | "static+llm";
}

export interface RunResult {
  id: string;
  analysis_id: string;
  variant: string;
  created_at: string;
  dataset: Record<string, unknown>;
  results: NodeEvalResult[];
  samples: SampleResult[];
  bottlenecks: Bottleneck[];
  overall: number | null;
  duration_s: number;
  models: Record<string, string>;
  mock: boolean;
  replayed: boolean;
  patches?: string[];
}

export interface SweepRequest {
  analysis_id: string;
  hotspot_id: string;
  values?: string[] | null;
  attachments: Attachment[];
  limit?: number | null;
  isolated?: boolean;
  base_run_id?: string | null;
}

export interface SweepResult {
  id: string;
  analysis_id: string;
  hotspot_id: string;
  key: string;
  original: string;
  base_run_id: string;
  runs: Record<string, string>;
  created_at: string;
}

export interface CompareRow {
  node_id: string;
  evaluator_id: string;
  unit: Unit;
  base: number | null;
  other: number | null;
  delta: number | null;
  better: boolean | null;
  base_status?: Status;
  other_status?: Status;
}

export interface RunComparison {
  base_run_id: string;
  other_run_id: string;
  rows: CompareRow[];
}

// §5 jobs
export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  stage: string;
  progress: number;
  log: string[];
  result_id: string | null;
  error: string | null;
}

// API helpers
export interface AnalyzeRequest {
  repo_url?: string;
  local_path?: string;
}

export interface JobRef {
  job_id: string;
}

export interface Health {
  ok: boolean;
  mock: boolean;
  replay: boolean;
  model: string;
  judge_model: string;
}

export interface AnalysisListItem {
  id: string;
  name: string;
  created_at: string;
  runnable: boolean;
}

export interface RunListItem {
  id: string;
  variant: string;
  created_at: string;
  overall: number | null;
  mock?: boolean;
}

/** RunResult.dataset.samples: per-sample previews for drill-down (written by the runner). */
export interface SamplePreview {
  question: string;
  category: string;
  answer: string;
  error: string | null;
}
