import { X } from "lucide-react";
import { formatValue } from "../../lib/format";
import { useEvalStore } from "../../store/useEvalStore";
import { useRunStore } from "../../store/useRunStore";
import type { AnalysisResult, NodeEvalResult } from "../../types/contracts";
import { MockPill } from "./RunReport";
import { STATUS_COLOR } from "./status";

/** One column per swept value (plus the code's own value); the best value per metric is marked, nothing is hidden. */
export function SweepView({ analysis }: { analysis: AnalysisResult }) {
  const { sweep, sweepOpen, setSweepOpen, runs } = useRunStore();
  const spec = useEvalStore((s) => s.spec);
  if (!sweep || !sweepOpen) return null;
  const columns = [
    { label: `${sweep.key} = ${sweep.original} (code)`, run: runs[sweep.base_run_id] },
    ...Object.entries(sweep.runs).map(([v, id]) => ({ label: `${sweep.key} = ${v}`, run: runs[id] })),
  ];
  const keys = [...new Map(columns.flatMap((c) => c.run?.results ?? []).map((r) => [`${r.node_id}|${r.evaluator_id}`, r])).values()];
  const label = (id: string) => analysis.graph.nodes.find((n) => n.id === id)?.label ?? id;
  const cell = (c: (typeof columns)[number], k: NodeEvalResult) =>
    c.run?.results.find((r) => r.node_id === k.node_id && r.evaluator_id === k.evaluator_id);

  return (
    <div className="absolute inset-0 z-20 flex items-start justify-center bg-canvas/70 p-6 backdrop-blur-sm" onClick={() => setSweepOpen(false)}>
      <div className="eg-pop max-w-full overflow-hidden rounded-xl border border-line bg-panel shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 text-[15px] font-semibold">
              Sweep of <span className="font-mono text-accent">{sweep.key}</span>
              {columns[0].run && <MockPill run={columns[0].run} />}
            </div>
            <div className="text-[12px] text-muted">
              Same {String(columns[0].run?.dataset.size ?? "?")} samples per column · only {sweep.key} changes · ★ best per row
            </div>
          </div>
          <button onClick={() => setSweepOpen(false)} className="text-muted hover:text-ink" aria-label="Close sweep"><X size={18} /></button>
        </div>
        <div className="max-h-[70vh] overflow-auto">
          <table className="text-[12.5px]">
            <thead className="sticky top-0 bg-panel-2 text-[11px] tracking-wider text-muted uppercase">
              <tr>
                <th className="px-4 py-2 text-left">Node · evaluator</th>
                {columns.map((c) => <th key={c.label} className="px-3 py-2 text-right font-mono normal-case">{c.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => {
                const s = spec(k.evaluator_id);
                const vals = columns.map((c) => cell(c, k)?.value ?? null);
                const scored = vals.filter((v): v is number => v !== null);
                // a star marks a real winner: none when every value ties
                const best = scored.length > 1 && Math.min(...scored) !== Math.max(...scored)
                  ? (s?.higher_is_better === false ? Math.min(...scored) : Math.max(...scored)) : null;
                return (
                  <tr key={`${k.node_id}|${k.evaluator_id}`} className="border-t border-line/60">
                    <td className="px-4 py-1.5">{label(k.node_id)} <span className="text-muted">· {s?.name ?? k.evaluator_id}</span></td>
                    {columns.map((c, i) => {
                      const r = cell(c, k);
                      return (
                        <td key={c.label} className="px-3 py-1.5 text-right tabular-nums" style={{ color: r ? STATUS_COLOR[r.status] : undefined }}
                            title={r?.reason || undefined}>
                          {formatValue(vals[i], k.unit)}{best !== null && vals[i] === best ? " ★" : ""}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
