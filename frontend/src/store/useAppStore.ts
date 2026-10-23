import { create } from "zustand";
import { api, pollJob, useFixtures } from "../api/client";
import type { AnalysisListItem, AnalysisResult, AnalyzeRequest, Health, Job } from "../types/contracts";

export type Screen = "home" | "analyzing" | "workspace";
export type Mode = "architecture" | "evaluation";

const FIXTURE_LOG = [
  "Fetched cuda-archer",
  "Scanning 39 Python files",
  "Extracted 160 functions/classes; detected 21 AI components",
  "Built architecture graph: 21 components, 31 connections",
  "LLM refined 21 labels, added 0 edges, rejected 0",
  "Found 17 evaluation hotspots",
];

interface AppState {
  screen: Screen;
  mode: Mode;
  health: Health | null;
  healthError: string | null;
  source: AnalyzeRequest | null;
  job: Job | null;
  error: string | null;
  analysis: AnalysisResult | null;
  recent: AnalysisListItem[];
  selectedNodeId: string | null;
  focusRequest: { nodeId: string; at: number } | null;

  loadHealth: () => Promise<void>;
  loadRecent: () => Promise<void>;
  analyze: (source: AnalyzeRequest) => Promise<void>;
  openAnalysis: (id: string) => Promise<void>;
  goHome: () => void;
  setMode: (mode: Mode) => void;
  select: (nodeId: string | null) => void;
  focusNode: (nodeId: string) => void;
  /** Replace the open analysis after a server-side edit (run setup, dismissed hotspots). */
  setAnalysis: (analysis: AnalysisResult) => void;
}

function setHash(id: string | null) {
  const target = id ? `#/a/${id}` : "#/";
  if (window.location.hash !== target) window.history.replaceState(null, "", target + "");
}

async function fakeJob(onUpdate: (job: Job) => void): Promise<Job> {
  const job: Job = { id: "j_fixture", kind: "analyze", status: "running", stage: "Opening repository", progress: 0, log: [], result_id: null, error: null };
  for (let i = 0; i < FIXTURE_LOG.length; i++) {
    await new Promise((r) => setTimeout(r, 550));
    job.log = [...job.log, FIXTURE_LOG[i]];
    job.progress = (i + 1) / (FIXTURE_LOG.length + 1);
    job.stage = FIXTURE_LOG[i];
    onUpdate({ ...job });
  }
  return { ...job, status: "done", progress: 1, result_id: "a_fixture1" };
}

export const useAppStore = create<AppState>()((set, get) => ({
  screen: "home",
  mode: "architecture",
  health: null,
  healthError: null,
  source: null,
  job: null,
  error: null,
  analysis: null,
  recent: [],
  selectedNodeId: null,
  focusRequest: null,

  loadHealth: async () => {
    try {
      set({ health: await api.health(), healthError: null });
    } catch (e) {
      set({ healthError: (e as Error).message });
    }
  },

  loadRecent: async () => {
    try {
      set({ recent: await api.analyses() });
    } catch {
      set({ recent: [] });
    }
  },

  analyze: async (source) => {
    set({ screen: "analyzing", source, job: null, error: null, analysis: null, selectedNodeId: null, mode: "architecture" });
    try {
      let job: Job;
      if (useFixtures) {
        job = await fakeJob((j) => set({ job: j }));
      } else {
        const { job_id } = await api.analyze(source);
        job = await pollJob(job_id, (j) => set({ job: j }));
      }
      set({ job });
      if (job.result_id) await get().openAnalysis(job.result_id);
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  openAnalysis: async (id) => {
    try {
      const analysis = await api.analysis(id);
      set({ analysis, screen: "workspace", selectedNodeId: null, error: null });
      setHash(analysis.id);
    } catch (e) {
      set({ error: (e as Error).message, screen: "home" });
      setHash(null);
    }
  },

  goHome: () => {
    set({ screen: "home", analysis: null, job: null, error: null, selectedNodeId: null, mode: "architecture" });
    setHash(null);
    void get().loadRecent();
  },

  setMode: (mode) => set({ mode }),
  select: (nodeId) => set({ selectedNodeId: nodeId }),
  focusNode: (nodeId) => set({ selectedNodeId: nodeId, focusRequest: { nodeId, at: Date.now() } }),
  setAnalysis: (analysis) => set({ analysis }),
}));
