import { X } from "lucide-react";
import { formatValue } from "../../lib/format";
import { useAppStore } from "../../store/useAppStore";
import { useEvalStore } from "../../store/useEvalStore";
import { useRunStore } from "../../store/useRunStore";
import type { AnalysisResult, CompareRow } from "../../types/contracts";
import { MockPill } from "./RunReport";
import { STATUS_COLOR } from "./status";

function formatDelta(row: CompareRow): string {
  if (row.delta === null) return "—";
  if (row.unit === "score" && row.base !== null && row.other !== null) {
    // Same rounding as the Before/After cells, so 13% → 15% reads +2 pts, not the raw +1.5 rounded down.
    let pts = Math.round(row.other * 100) - Math.round(row.base * 100);
    let text = String(Math.abs(pts));
    if (pts === 0 && row.delta !== 0) { pts = row.delta; text = Math.abs(row.delta * 100).toFixed(1); }
    return `${pts > 0 ? "+" : pts < 0 ? "−" : "±"}${text} pts`;
  }
  const sign = row.delta > 0 ? "+" : row.delta < 0 ? "−" : "±";
  const abs = Math.abs(row.delta);
  if (row.unit === "usd" && abs > 0 && abs < 0.00005) return `${sign}<$0.0001`; // would print as $0.0000
  return row.unit === "score" ? `${sign}${Math.round(abs * 100)} pts` : `${sign}${formatValue(abs, row.unit)}`;
}

/** Before/after table for a fix re-run. Rows that moved come first; nothing is hidden or rounded into a verdict. */
export function CompareView({ analysis }: { analysis: AnalysisResult }) {
  const { comparison, runs, compareOpen, setCompareOpen } = useRunStore();
  const spec = useEvalStore((s) => s.spec);
  const focusNode = useAppStore((s) => s.focusNode);
  if (!comparison || !compareOpen) return null;
  const base = runs[comparison.base_run_id];
  const other = runs[comparison.other_run_id];
  const label = (id: string) => analysis.graph.nodes.find((n) => n.id === id)?.label ?? id;
  const rows = [...comparison.rows].sort((a, b) => Number(b.better !== null) - Number(a.better !== null));
  const improved = rows.filter((r) => r.better === true).length;
  const worse = rows.filter((r) => r.better === false).length;

  return (
    <div className="absolute inset-0 z-20 flex items-start justify-center bg-canvas/70 p-6 backdrop-blur-sm" onClick={() => setCompareOpen(false)}>
      <div className="eg-pop w-[760px] max-w-full overflow-hidden rounded-xl border border-line bg-panel shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 text-[15px] font-semibold">
              <span className="font-mono">{base?.variant ?? "base"}</span> vs <span className="font-mono text-accent">{other?.variant ?? "other"}</span>
              {other && <MockPill run={other} />}
            </div>
            <div className="text-[12px] text-muted">
              {improved} better · {worse} worse · {rows.length - improved - worse} unchanged or not comparable · same samples ({String(other?.dataset.size ?? "?")})
            </div>
          </div>
          <button onClick={() => setCompareOpen(false)} className="text-muted hover:text-ink" aria-label="Close compare"><X size={18} /></button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto">
          <table className="w-full text-[12.5px]">
            <thead className="sticky top-0 bg-panel-2 text-[11px] tracking-wider text-muted uppercase">
              <tr>
                <th className="px-4 py-2 text-left">Node</th>
                <th className="px-2 py-2 text-left">Evaluator</th>
                <th className="px-2 py-2 text-right">Before</th>
                <th className="px-2 py-2 text-right">After</th>
                <th className="px-4 py-2 text-right">Δ</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.node_id}:${r.evaluator_id}`} className="cursor-pointer border-t border-line/60 hover:bg-panel-2"
                    onClick={() => { setCompareOpen(false); focusNode(r.node_id); }}>
                  <td className="px-4 py-1.5">{label(r.node_id)}</td>
                  <td className="px-2 py-1.5 text-muted">{spec(r.evaluator_id)?.name ?? r.evaluator_id}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums" style={{ color: r.base_status ? STATUS_COLOR[r.base_status] : undefined }}>{formatValue(r.base, r.unit)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums" style={{ color: r.other_status ? STATUS_COLOR[r.other_status] : undefined }}>{formatValue(r.other, r.unit)}</td>
                  <td className={`px-4 py-1.5 text-right font-bold tabular-nums ${r.better === true ? "text-good" : r.better === false ? "text-bad" : "text-muted"}`}>
                    {formatDelta(r)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
