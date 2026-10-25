import { History } from "lucide-react";
import { formatValue } from "../../lib/format";
import { useRunStore } from "../../store/useRunStore";

/** Past runs of this analysis, newest first; picking one paints it on the graph. */
export function RunHistory() {
  const { history, activeRunId, openRun, comparison, side, setSide } = useRunStore();
  if (!history.length && !comparison) return null;
  return (
    <div className="flex items-center gap-2">
      {comparison && (
        <div className="flex rounded-lg border border-line bg-canvas p-0.5 text-[11px] font-semibold" title="Which run the graph shows">
          {(["base", "other"] as const).map((s) => (
            <button key={s} onClick={() => setSide(s)}
                    className={`rounded-md px-2 py-1 ${side === s ? "bg-panel-2 text-ink" : "text-muted hover:text-ink"}`}>
              {s === "base" ? "Before" : "After"}
            </button>
          ))}
        </div>
      )}
      {history.length > 0 && (
        <label className="flex items-center gap-1 text-muted" title="Past runs">
          <History size={14} />
          <select
            value={activeRunId ?? ""}
            onChange={(e) => e.target.value && void openRun(e.target.value)}
            className="max-w-[150px] rounded-lg border border-line bg-canvas px-2 py-1.5 text-[12px] text-ink outline-none"
          >
            {!activeRunId && <option value="">Past runs…</option>}
            {history.map((r) => (
              <option key={r.id} value={r.id}>
                {r.variant} · {new Date(r.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                {r.overall !== null ? ` · ${formatValue(r.overall, "score")}` : ""}{r.mock ? " · mock" : ""}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
