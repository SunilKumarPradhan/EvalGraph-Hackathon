import { create } from "zustand";
import { ApiError, api, pollJob, useFixtures } from "../api/client";
import type { AnalysisResult, Attachment, EvaluatorSpec, Job } from "../types/contracts";
import { useRunStore } from "./useRunStore";
import { toast } from "./useToast";

const storageKey = (analysisId: string) => `evalgraph.attachments.${analysisId}`;

function loadSaved(analysisId: string): Attachment[] {
  try {
    const raw = localStorage.getItem(storageKey(analysisId));
    const parsed = raw ? (JSON.parse(raw) as Attachment[]) : [];
    return Array.isArray(parsed) ? parsed.filter((a) => a && a.node_id && a.evaluator_id) : [];
  } catch {
    return [];
  }
}

let serverTimer: ReturnType<typeof setTimeout> | undefined;

/** Attachments live with the analysis on the server (any laptop sees them); localStorage is the offline fallback. */
function save(analysisId: string | null, attachments: Attachment[]) {
  if (!analysisId) return;
  if (!useFixtures) {
    clearTimeout(serverTimer);
    serverTimer = setTimeout(() => void api.saveAttachments(analysisId, attachments).catch(() => undefined), 400);
  }
  try {
    localStorage.setItem(storageKey(analysisId), JSON.stringify(attachments));
  } catch {
    /* storage unavailable: attachments just won't survive a reload */
  }
}

async function fakeRun(onUpdate: (job: Job) => void, total: number): Promise<Job> {
  const job: Job = { id: "j_fixture_run", kind: "run", status: "running", stage: "Loading dataset", progress: 0, log: [], result_id: null, error: null };
  const steps = [`Loading dataset (${total} samples)`, `Running CUDA Archer ${total}/${total}`, "Mapping spans to architecture", "Scoring evaluators", "Diagnosing bottlenecks"];
  for (let i = 0; i < steps.length; i++) {
    await new Promise((r) => setTimeout(r, 450));
    job.log = [...job.log, steps[i]];
    job.stage = steps[i];
    job.progress = (i + 1) / steps.length;
    onUpdate({ ...job });
  }
  return { ...job, status: "done", result_id: "r_fixture1" };
}

interface EvalState {
  analysisId: string | null;
  evaluators: EvaluatorSpec[];
  attachments: Attachment[];
  dragging: string | null;
  variant: string;
  quick: boolean;
  isolated: boolean;
  running: boolean;
  runJob: Job | null;
  runId: string | null;

  init: (analysis: AnalysisResult, evaluators: EvaluatorSpec[]) => void;
  spec: (evaluatorId: string) => EvaluatorSpec | undefined;
  attach: (analysis: AnalysisResult, nodeId: string, evaluatorId: string, quiet?: boolean) => boolean;
  detach: (nodeId: string, evaluatorId: string) => void;
  clear: () => void;
  applyRecommended: (analysis: AnalysisResult) => Promise<void>;
  setDragging: (evaluatorId: string | null) => void;
  setVariant: (variant: string) => void;
  setQuick: (quick: boolean) => void;
  setIsolated: (isolated: boolean) => void;
  /** Plain run of the current attachments, or a fix re-run: the base run's attachments with another variant
   *  (adapter repos) or with rule-based patches applied in the sandbox (any other repo), then compare. */
  startRun: (analysis: AnalysisResult, fix?: { variant?: string; patches?: string[]; baseRunId: string }) => Promise<void>;
  /** Sweep one hotspot's numeric literal over its suggested values against the shown (unpatched) run. */
  startSweep: (analysis: AnalysisResult, hotspotId: string, baseRunId: string) => Promise<void>;
}

export const useEvalStore = create<EvalState>()((set, get) => ({
  analysisId: null,
  evaluators: [],
  attachments: [],
  dragging: null,
  variant: "baseline",
  quick: true,
  isolated: false,
  running: false,
  runJob: null,
  runId: null,

  init: (analysis, evaluators) => {
    if (get().analysisId === analysis.id && get().evaluators.length === evaluators.length) return;
    const valid = new Set(analysis.graph.nodes.map((n) => n.id));
    set({
      analysisId: analysis.id,
      evaluators,
      // the server copy wins; a setup only saved in this browser is uploaded once
      attachments: (analysis.attachments?.length ? analysis.attachments : loadSaved(analysis.id)).filter((a) => valid.has(a.node_id)),
      variant: analysis.summary.variants.includes("baseline") ? "baseline" : analysis.summary.variants[0] ?? "baseline",
      runJob: null,
      runId: null,
    });
  },

  spec: (evaluatorId) => get().evaluators.find((e) => e.id === evaluatorId),

  attach: (analysis, nodeId, evaluatorId, quiet = false) => {
    const node = analysis.graph.nodes.find((n) => n.id === nodeId);
    const spec = get().spec(evaluatorId);
    if (!node || !spec) return false;
    if (!spec.applies_to.includes(node.type)) {
      if (!quiet) toast(`${spec.name} applies to: ${spec.applies_to.join(", ")}. "${node.label}" is a ${node.type}.`, "error");
      return false;
    }
    if (get().attachments.some((a) => a.node_id === nodeId && a.evaluator_id === evaluatorId)) return false;
    const attachments = [...get().attachments, { node_id: nodeId, evaluator_id: evaluatorId }];
    set({ attachments });
    save(get().analysisId, attachments);
    if (!quiet) toast(`${spec.name} → ${node.label}`, "success");
    return true;
  },

  detach: (nodeId, evaluatorId) => {
    const attachments = get().attachments.filter((a) => !(a.node_id === nodeId && a.evaluator_id === evaluatorId));
    set({ attachments });
    save(get().analysisId, attachments);
  },

  clear: () => {
    set({ attachments: [] });
    save(get().analysisId, []);
  },

  applyRecommended: async (analysis) => {
    let added = 0;
    for (const r of analysis.recommendations) {
      if (get().attach(analysis, r.node_id, r.evaluator_id, true)) {
        added += 1;
        await new Promise((res) => setTimeout(res, 70)); // chips pop in one by one
      }
    }
    toast(added ? `Added ${added} recommended evaluators` : "All recommended evaluators are already attached", added ? "success" : "info");
  },

  setDragging: (evaluatorId) => set({ dragging: evaluatorId }),
  setVariant: (variant) => set({ variant }),
  setQuick: (quick) => set({ quick }),
  setIsolated: (isolated) => set({ isolated }),

  startSweep: async (analysis, hotspotId, baseRunId) => {
    const runs = useRunStore.getState();
    const base = runs.runs[baseRunId];
    if (!base || get().running) return;
    const attachments = base.results.map((r) => ({ node_id: r.node_id, evaluator_id: r.evaluator_id }));
    set({ running: true, runJob: null });
    try {
      const { job_id } = await api.startSweep({
        analysis_id: analysis.id, hotspot_id: hotspotId, attachments, limit: Number(base.dataset.size ?? 0) || null,
        isolated: Boolean(base.dataset.isolated), base_run_id: (base.patches ?? []).length ? null : base.id,
      });
      const job = await pollJob(job_id, (j) => set({ runJob: j }));
      if (job.result_id) await runs.openSweep(job.result_id);
      void runs.loadHistory();
      toast("Sweep finished", "success");
    } catch (e) {
      toast(`Sweep failed: ${(e as ApiError).message}`, "error");
    } finally {
      set({ running: false });
    }
  },

  startRun: async (analysis, fix) => {
    const runs = useRunStore.getState();
    const base = fix ? runs.runs[fix.baseRunId] : undefined;
    // a fix re-run repeats the base run exactly (same attachments, same sample count), only the variant changes
    const attachments = base ? base.results.map((r) => ({ node_id: r.node_id, evaluator_id: r.evaluator_id })) : get().attachments;
    // patches stack on the base run's own patches and keep its underlying variant (sandbox: "as-is")
    const patches = fix?.patches ? [...new Set([...(base?.patches ?? []), ...fix.patches])] : null;
    const plainVariant = base && analysis.summary.variants.includes(base.variant) ? base.variant
      : analysis.summary.variants[0] ?? get().variant;
    const variant = fix?.variant ?? (patches ? plainVariant : get().variant);
    const label = patches ? `${patches.length} patch${patches.length > 1 ? "es" : ""}` : variant;
    const limit = base ? Number(base.dataset.size ?? 0) || null : get().quick ? 10 : null;
    if (!attachments.length || get().running) return;
    set({ running: true, runJob: null, runId: null });
    try {
      let job: Job;
      if (useFixtures) {
        job = await fakeRun((j) => set({ runJob: j }), limit ?? 41);
      } else {
        // a fix re-run keeps the base run's isolation so before/after stay comparable
        const isolated = base ? Boolean(base.dataset.isolated) : get().isolated;
        const { job_id } = await api.startRun({ analysis_id: analysis.id, attachments, variant, limit, patches, isolated });
        job = await pollJob(job_id, (j) => set({ runJob: j }));
      }
      set({ runJob: job, runId: job.result_id });
      if (job.result_id) {
        if (fix && !useFixtures) await runs.compareWith(fix.baseRunId, job.result_id);
        else await runs.openRun(job.result_id);
        void runs.loadHistory();
      }
      toast(fix ? `Fix run finished (${label}): before/after ready` : `Run finished (${variant})`, "success");
    } catch (e) {
      toast(`Run failed: ${(e as ApiError).message}`, "error");
    } finally {
      set({ running: false });
    }
  },
}));
