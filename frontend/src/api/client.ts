import type {
  AnalysisListItem,
  AnalysisResult,
  AnalyzeRequest,
  EvaluatorSpec,
  Health,
  Job,
  JobRef,
  RunComparison,
  RunListItem,
  RunRequest,
  RunResult,
  RuntimeUpdate,
  SweepRequest,
  SweepResult,
  Attachment,
} from "../types/contracts";

/** `?fixtures=1` in the URL: serve everything from /fixtures/*.json, no backend needed. */
export const useFixtures = new URLSearchParams(window.location.search).has("fixtures");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

async function fixture<T>(name: string): Promise<T> {
  const res = await fetch(`/fixtures/${name}.json`);
  if (!res.ok) throw new ApiError(res.status, `fixture ${name} missing`);
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () =>
    useFixtures
      ? Promise.resolve<Health>({ ok: true, mock: true, replay: false, model: "fixtures", judge_model: "fixtures" })
      : request<Health>("/health"),
  analyze: (body: AnalyzeRequest) => post<JobRef>("/analyze", body),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  analyses: () => (useFixtures ? Promise.resolve<AnalysisListItem[]>([]) : request<AnalysisListItem[]>("/analyses")),
  analysis: (id: string) => (useFixtures ? fixture<AnalysisResult>("sample_analysis") : request<AnalysisResult>(`/analyses/${id}`)),
  evaluators: () => (useFixtures ? fixture<EvaluatorSpec[]>("evaluators") : request<EvaluatorSpec[]>("/evaluators")),
  startRun: (body: RunRequest) => post<JobRef>("/runs", body),
  run: (id: string) => (useFixtures ? fixture<RunResult>("sample_run") : request<RunResult>(`/runs/${id}`)),
  runs: (analysisId: string) => request<RunListItem[]>(`/runs?analysis_id=${encodeURIComponent(analysisId)}`),
  compare: (a: string, b: string) => request<RunComparison>(`/runs/${a}/compare/${b}`),
  startSweep: (body: SweepRequest) => post<JobRef>("/runs/sweep", body),
  sweep: (id: string) => request<SweepResult>(`/sweeps/${id}`),
  updateRuntime: (id: string, body: RuntimeUpdate) => request<AnalysisResult>(`/analyses/${id}/runtime`, { method: "PUT", body: JSON.stringify(body) }),
  saveAttachments: (id: string, body: Attachment[]) => request<AnalysisResult>(`/analyses/${id}/attachments`, { method: "PUT", body: JSON.stringify(body) }),
  secrets: (id: string) => request<{ stored: string[]; wanted: string[] }>(`/analyses/${id}/secrets`),
  saveSecrets: (id: string, changes: Record<string, string | null>) =>
    request<{ stored: string[]; wanted: string[] }>(`/analyses/${id}/secrets`, { method: "PUT", body: JSON.stringify(changes) }),
  saveDismissed: (id: string, ids: string[]) => request<AnalysisResult>(`/analyses/${id}/dismissed`, { method: "PUT", body: JSON.stringify(ids) }),
};

/** Poll a job every `intervalMs` until done/error. Resolves with the final job; rejects on error status. */
export async function pollJob(jobId: string, onUpdate: (job: Job) => void, intervalMs = 750): Promise<Job> {
  for (;;) {
    const job = await api.job(jobId);
    onUpdate(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new ApiError(500, job.error ?? "job failed");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
