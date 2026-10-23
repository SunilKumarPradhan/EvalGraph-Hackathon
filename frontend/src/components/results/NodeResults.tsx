import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { formatValue } from "../../lib/format";
import { useEvalStore } from "../../store/useEvalStore";
import type { NodeEvalResult, RunResult, SamplePreview, SampleResult } from "../../types/contracts";
import { MockPill } from "./RunReport";
import { STATUS_COLOR } from "./status";

function SampleRow({ s, unit, preview }: { s: SampleResult; unit: NodeEvalResult["unit"]; preview?: SamplePreview }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-md bg-canvas">
      <button className="flex w-full items-start gap-2 px-2 py-1.5 text-left" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={13} className="mt-0.5 shrink-0" /> : <ChevronRight size={13} className="mt-0.5 shrink-0" />}
        <span className="font-mono text-[11px] text-muted">{s.sample_id}</span>
        <span className="min-w-0 flex-1 text-[11.5px] leading-snug">{s.error ? `error: ${s.error}` : s.reason}</span>
        <span className="text-[11.5px] font-bold tabular-nums">{s.error ? "—" : formatValue(s.value, unit)}</span>
      </button>
      {open && preview && (
        <div className="flex flex-col gap-1.5 border-t border-line/60 px-2.5 py-2 text-[11.5px]">
          <div><span className="text-muted">Question · {preview.category}: </span>{preview.question}</div>
          {preview.error
            ? <div className="text-bad">App error: {preview.error}</div>
            : <div className="whitespace-pre-wrap"><span className="text-muted">Answer: </span>{preview.answer || "(empty)"}</div>}
        </div>
      )}
    </li>
  );
}

function ResultCard({ r, run, open, onToggle }: { r: NodeEvalResult; run: RunResult; open: boolean; onToggle: () => void }) {
  const spec = useEvalStore((s) => s.spec(r.evaluator_id));
  const color = STATUS_COLOR[r.status];
  const previews = (run.dataset.samples ?? {}) as Record<string, SamplePreview>;
  const samples = run.samples.filter((s) => s.node_id === r.node_id && s.evaluator_id === r.evaluator_id);
  // worst first (direction-aware), errors last
  const sign = spec?.higher_is_better === false ? -1 : 1;
  const ordered = [...samples].sort((a, b) => (a.value === null ? 1 : b.value === null ? -1 : sign * (a.value - b.value)));
  return (
    <div className="rounded-lg border bg-panel-2" style={{ borderColor: `${color}66` }}>
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left" onClick={onToggle}>
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
        <span className="flex-1 text-[13px] font-semibold">{spec?.name ?? r.evaluator_id}</span>
        <span className="text-[11px] text-muted">n={r.n}</span>
        <span className="text-[14px] font-bold tabular-nums" style={{ color }}>{formatValue(r.value, r.unit)}</span>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open && (
        <div className="flex flex-col gap-2 px-3 pb-3">
          {spec && (
            <div className="text-[11px] text-muted">
              {spec.description} Green {spec.higher_is_better ? "≥" : "≤"} {formatValue(spec.thresholds.green, spec.unit)}
              {r.unit === "ms" ? " (p95)" : ""}.
            </div>
          )}
          {r.reason && <div className="rounded-md border border-line px-2 py-1 text-[11.5px] text-muted">{r.reason}</div>}
          {ordered.length > 0 && (
            <ul className="flex max-h-72 flex-col gap-1 overflow-y-auto">
              {ordered.map((s) => <SampleRow key={s.sample_id} s={s} unit={r.unit} preview={previews[s.sample_id]} />)}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

/** "Results" section of the node drawer when a run is shown. */
export function NodeResults({ nodeId, run }: { nodeId: string; run: RunResult }) {
  const results = run.results.filter((r) => r.node_id === nodeId);
  const [openIdx, setOpenIdx] = useState(0);
  if (!results.length) return null;
  return (
    <section className="flex flex-col gap-2">
      <h3 className="flex items-center gap-2 text-[11px] font-semibold tracking-wider text-muted uppercase">
        Results · {run.variant} <MockPill run={run} />
      </h3>
      {results.map((r, i) => (
        <ResultCard key={r.evaluator_id} r={r} run={run} open={openIdx === i} onToggle={() => setOpenIdx(openIdx === i ? -1 : i)} />
      ))}
    </section>
  );
}
