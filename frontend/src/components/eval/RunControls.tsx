import { Loader2, Play } from "lucide-react";
import { useEvalStore } from "../../store/useEvalStore";
import type { AnalysisResult } from "../../types/contracts";

export function RunControls({ analysis }: { analysis: AnalysisResult }) {
  const { attachments, variant, setVariant, quick, setQuick, isolated, setIsolated, running, startRun } = useEvalStore();
  const canIsolate = analysis.summary.adapter === "sandbox" && analysis.runtime?.plan?.mode === "python_function";
  const { runnable, variants } = analysis.summary;
  const disabledReason = !runnable
    ? `Plan mode: ${String(analysis.runtime?.plan?.reason ?? "this repo cannot be run")}.`
    : attachments.length === 0
      ? "Attach at least one evaluator first."
      : running ? "Run in progress…" : "";

  return (
    <div className="flex items-center gap-2">
      {runnable && variants.length > 0 && (
        <select
          value={variant}
          onChange={(e) => setVariant(e.target.value)}
          className="rounded-lg border border-line bg-canvas px-2 py-1.5 text-[12px] outline-none"
          title="Config variant of the target app to run"
        >
          {variants.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      )}
      {runnable && (
        <label className="flex items-center gap-1.5 text-[12px] text-muted" title="Run 10 samples spread across all question categories">
          <input type="checkbox" checked={quick} onChange={(e) => setQuick(e.target.checked)} className="accent-teal-400" />
          quick
        </label>
      )}
      {canIsolate && (
        <label className="flex items-center gap-1.5 text-[12px] text-muted"
               title="Run every question in a fresh process, so conversation memory can't leak between samples">
          <input type="checkbox" checked={isolated} onChange={(e) => setIsolated(e.target.checked)} className="accent-teal-400" />
          isolated
        </label>
      )}
      <span title={disabledReason}>
        <button
          onClick={() => void startRun(analysis)}
          disabled={Boolean(disabledReason)}
          className="flex items-center gap-1.5 rounded-lg bg-good px-3.5 py-1.5 whitespace-nowrap text-[13px] font-bold text-canvas hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {running ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} fill="currentColor" />}
          Run Evaluation
        </button>
      </span>
    </div>
  );
}

export function RunProgress() {
  const { running, runJob } = useEvalStore();
  if (!running) return null;
  const pct = Math.round((runJob?.progress ?? 0.02) * 100);
  return (
    <div className="eg-pop absolute bottom-4 left-4 z-10 w-80 rounded-xl border border-line bg-panel/95 p-3 shadow-2xl backdrop-blur">
      <div className="flex items-center gap-2 text-[13px] font-semibold">
        <Loader2 size={15} className="animate-spin text-accent" /> {runJob?.stage ?? "Starting run"}
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full rounded-full bg-accent transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <ul className="mt-2 flex flex-col gap-0.5 text-[11px] text-muted">
        {(runJob?.log ?? []).slice(-4).map((l, i) => <li key={i}>✓ {l}</li>)}
      </ul>
    </div>
  );
}
