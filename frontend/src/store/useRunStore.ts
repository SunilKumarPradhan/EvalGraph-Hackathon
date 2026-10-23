import { create } from "zustand";
import { api, useFixtures } from "../api/client";
import type { RunComparison, RunListItem, RunResult, SweepResult } from "../types/contracts";
import { toast } from "./useToast";

export type Side = "base" | "other";

interface RunState {
  analysisId: string | null;
  runs: Record<string, RunResult>;
  activeRunId: string | null;
  history: RunListItem[];
  /** Before/after pair after "Try suggested fix"; `side` picks which run the graph shows. */
  comparison: RunComparison | null;
  side: Side;
  compareOpen: boolean;
  reportOpen: boolean;
  sweep: SweepResult | null;
  sweepOpen: boolean;

  reset: (analysisId: string) => void;
  loadHistory: () => Promise<void>;
  openRun: (runId: string) => Promise<RunResult | null>;
  compareWith: (baseRunId: string, otherRunId: string) => Promise<void>;
  setSide: (side: Side) => void;
  setCompareOpen: (open: boolean) => void;
  setReportOpen: (open: boolean) => void;
  shownRun: () => RunResult | null;
  /** Load a finished sweep and every run in it, then open the sweep table. */
  openSweep: (sweepId: string) => Promise<void>;
  setSweepOpen: (open: boolean) => void;
}

export const useRunStore = create<RunState>()((set, get) => ({
  analysisId: null,
  runs: {},
  activeRunId: null,
  history: [],
  comparison: null,
  side: "other",
  compareOpen: false,
  reportOpen: false,
  sweep: null,
  sweepOpen: false,

  reset: (analysisId) => {
    if (get().analysisId === analysisId) return;
    set({ analysisId, runs: {}, activeRunId: null, history: [], comparison: null, side: "other", compareOpen: false, reportOpen: false,
          sweep: null, sweepOpen: false });
    void get().loadHistory();
  },

  loadHistory: async () => {
    const id = get().analysisId;
    if (!id || useFixtures) return;
    try {
      set({ history: await api.runs(id) });
    } catch {
      set({ history: [] });
    }
  },

  openRun: async (runId) => {
    try {
      const run = get().runs[runId] ?? (await api.run(runId));
      set((s) => ({ runs: { ...s.runs, [run.id]: run }, activeRunId: run.id, reportOpen: true }));
      return run;
    } catch (e) {
      toast(`Could not load run ${runId}: ${(e as Error).message}`, "error");
      return null;
    }
  },

  compareWith: async (baseRunId, otherRunId) => {
    try {
      await Promise.all([baseRunId, otherRunId].map((id) => (get().runs[id] ? null : get().openRun(id))));
      const comparison = await api.compare(baseRunId, otherRunId);
      set({ comparison, side: "other", activeRunId: otherRunId, compareOpen: true });
    } catch (e) {
      toast(`Compare failed: ${(e as Error).message}`, "error");
    }
  },

  setSide: (side) => {
    const c = get().comparison;
    set({ side, ...(c ? { activeRunId: side === "base" ? c.base_run_id : c.other_run_id } : {}) });
  },
  setCompareOpen: (compareOpen) => set({ compareOpen }),
  setReportOpen: (reportOpen) => set({ reportOpen }),
  openSweep: async (sweepId) => {
    try {
      const sweep = await api.sweep(sweepId);
      const ids = [sweep.base_run_id, ...Object.values(sweep.runs)];
      const loaded = await Promise.all(ids.map((id) => get().runs[id] ?? api.run(id)));
      set((s) => ({ runs: { ...s.runs, ...Object.fromEntries(loaded.map((r) => [r.id, r])) }, sweep, sweepOpen: true }));
    } catch (e) {
      toast(`Could not load sweep: ${(e as Error).message}`, "error");
    }
  },
  setSweepOpen: (sweepOpen) => set({ sweepOpen }),
  shownRun: () => {
    const { activeRunId, runs } = get();
    return activeRunId ? runs[activeRunId] ?? null : null;
  },
}));
