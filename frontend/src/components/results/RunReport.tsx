import { ChevronDown, ChevronRight, FlaskConical, GitCompare, Sparkles, Wand2, X } from "lucide-react";
import { useState } from "react";
import { fileLine, formatValue } from "../../lib/format";
import { useAppStore } from "../../store/useAppStore";
import { useEvalStore } from "../../store/useEvalStore";
import { useRunStore } from "../../store/useRunStore";
import type { AnalysisResult, Bottleneck, RunResult } from "../../types/contracts";
import { STATUS_COLOR } from "./status";

export function MockPill({ run }: { run: RunResult }) {
  if (run.replayed) {
    return (
      <span className="rounded-full border border-warn px-1.5 text-[10px] font-semibold text-warn"
            title={run.mock ? "Replay of a recorded MOCK run: not quality numbers" : `Replay of a real run recorded ${run.created_at}`}>
        {run.mock ? "REPLAY · MOCK" : "REPLAY"}
      </span>
    );
  }
  if (!run.mock) return null;
  return (
    <span className="rounded-full border border-warn px-1.5 text-[10px] font-semibold text-warn" title="Canned LLM output and pseudo judge scores: not quality numbers">
      MOCK
    </span>
  );
}

function BottleneckCard({ b, run, analysis, open, onToggle }: {
  b: Bottleneck; run: RunResult; analysis: AnalysisResult; open: boolean; onToggle: () => void;
}) {
  const { spec, startRun, running } = useEvalStore();
  const focusNode = useAppStore((s) => s.focusNode);
  const node = analysis.graph.nodes.find((n) => n.id === b.node_id);
  const s = spec(b.evaluator_id);
  const result = run.results.find((r) => r.node_id === b.node_id && r.evaluator_id === b.evaluator_id);
  const color = STATUS_COLOR[b.status];
  return (
    <div className="rounded-lg border bg-panel-2" style={{ borderColor: `${color}66` }}>
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left" onClick={onToggle}>
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-semibold">{node?.label ?? b.node_id}</span>
          <span className="block text-[11px] text-muted">{s?.name ?? b.evaluator_id} · n={result?.n ?? 0}</span>
        </span>
        <span className="text-[14px] font-bold tabular-nums" style={{ color }}>{formatValue(b.value, result?.unit ?? "score")}</span>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open && (
        <div className="flex flex-col gap-3 px-3 pb-3">
          <p className="text-[12.5px] leading-relaxed">{b.explanation}</p>
          {b.contributors.length > 0 && (
            <div>
              <div className="mb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">Likely contributors</div>
              <ul className="flex flex-col gap-1">
                {b.contributors.slice(0, 5).map((c, i) => (
                  <li key={i} className="rounded-md bg-canvas px-2 py-1">
                    <div className="font-mono text-[10.5px] text-accent">{fileLine(c.file, c.line)}</div>
                    <div className="text-[12px]">{c.fact}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {b.suggested_experiment && <div className="text-[12px] text-muted"><FlaskConical size={12} className="mr-1 inline" />{b.suggested_experiment}</div>}
          <div className="flex gap-2">
            <button onClick={() => focusNode(b.node_id)} className="rounded-md border border-line px-2.5 py-1 text-[12px] hover:border-accent">
              Show on graph
            </button>
            {((b.suggested_variant && b.suggested_variant !== run.variant)
              || (!!b.suggested_patch?.length && !b.suggested_patch.every((id) => (run.patches ?? []).includes(id)))) && (
              <button
                disabled={running}
                onClick={() => void startRun(analysis, b.suggested_variant
                  ? { variant: b.suggested_variant, baseRunId: run.id }
                  : { patches: b.suggested_patch ?? [], baseRunId: run.id })}
                className="flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 text-[12px] font-bold text-canvas hover:brightness-110 disabled:opacity-40"
              >
                <Wand2 size={13} /> Try suggested fix
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function RunReport({ analysis }: { analysis: AnalysisResult }) {
  const run = useRunStore((s) => (s.activeRunId ? s.runs[s.activeRunId] : null));
  const { comparison, setReportOpen, setCompareOpen } = useRunStore();
  const { startRun, running } = useEvalStore();
  const [openIdx, setOpenIdx] = useState(0);
  if (!run) return null;
  const scored = run.results.filter((r) => r.status !== "na" && r.status !== "error").length;
  const canFixAll = analysis.summary.variants.includes("improved") && run.variant !== "improved";

  return (
    <aside className="eg-slide flex w-[380px] shrink-0 flex-col overflow-hidden border-l border-line bg-panel">
      <div className="flex items-start gap-2 border-b border-line p-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[15px] font-semibold">
            Run report <MockPill run={run} />
          </div>
          <div className="mt-0.5 text-[12px] text-muted">
            variant <span className="font-mono text-ink">{run.variant}</span> · {String(run.dataset.size)} samples
            {run.dataset.name === "synthetic" ? " (generated)" : ""} · {run.duration_s.toFixed(1)} s
          </div>
          <div className="mt-0.5 text-[11px] text-muted">
            {Array.isArray(run.dataset.egress_blocked) && run.dataset.egress_blocked.length > 0 && (
              <span className="mr-1 text-warn" title="The sandbox only reaches package registries, git hosts, Hugging Face and DIAL">
                egress blocked: {(run.dataset.egress_blocked as string[]).join(", ")} ·{" "}
              </span>
            )}
            {scored}/{run.results.length} evaluations scored
            {run.overall !== null ? ` · mean score ${formatValue(run.overall, "score")}` : ""} · judge {run.models.judge ?? "?"}
          </div>
        </div>
        <button onClick={() => setReportOpen(false)} className="text-muted hover:text-ink" aria-label="Close report"><X size={18} /></button>
      </div>
      <div className="flex gap-2 border-b border-line px-4 py-2">
        {comparison && (
          <button onClick={() => setCompareOpen(true)} className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1 text-[12px] hover:border-accent">
            <GitCompare size={13} /> Before / after
          </button>
        )}
        {canFixAll && (
          <button
            disabled={running}
            onClick={() => void startRun(analysis, { variant: "improved", baseRunId: run.id })}
            className="flex items-center gap-1.5 rounded-md border border-accent/60 px-2.5 py-1 text-[12px] text-accent hover:bg-accent/10 disabled:opacity-40"
            title="Re-run with every fix applied (variant 'improved') and compare"
          >
            <Sparkles size={13} /> Fix everything
          </button>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-2 overflow-y-auto p-4">
        <h3 className="text-[11px] font-semibold tracking-wider text-muted uppercase">Bottlenecks ({run.bottlenecks.length})</h3>
        {run.bottlenecks.length === 0 && (
          <p className="text-[12.5px] text-muted">No evaluation is amber or red. Everything scored is within its green threshold.</p>
        )}
        {run.bottlenecks.map((b, i) => (
          <BottleneckCard key={`${b.node_id}:${b.evaluator_id}`} b={b} run={run} analysis={analysis} open={openIdx === i}
                          onToggle={() => setOpenIdx(openIdx === i ? -1 : i)} />
        ))}
      </div>
    </aside>
  );
}
