import { useState } from "react";
import { useAppStore } from "../store/useAppStore";
import type { AnalysisResult } from "../types/contracts";
import { SEVERITY_COLOR, SEVERITY_RANK } from "./canvas/typeStyle";

export function HotspotBar({ analysis }: { analysis: AnalysisResult }) {
  const focusNode = useAppStore((s) => s.focusNode);
  const selected = useAppStore((s) => s.selectedNodeId);
  const labels = Object.fromEntries(analysis.graph.nodes.map((n) => [n.id, n.label]));
  const [showAll, setShowAll] = useState(false);
  const dismissed = new Set(analysis.dismissed_hotspots ?? []);
  const active = analysis.hotspots.filter((h) => !dismissed.has(h.id));
  const hidden = active.filter((h) => h.severity === "low").length;
  const hotspots = active.filter((h) => showAll || h.severity !== "low")
    .sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
  if (active.length === 0) {
    return <div className="border-t border-line bg-panel px-4 py-2.5 text-[12px] text-muted">No evaluation gaps found. Nice.</div>;
  }
  return (
    <div className="relative shrink-0 border-t border-line bg-panel">
      <div className="flex items-center gap-2 overflow-x-auto px-4 py-2.5">
        <span className="shrink-0 text-[11px] font-semibold tracking-wider text-muted uppercase">⚠ Hotspots</span>
        {(hidden > 0 || dismissed.size > 0) && (
          <button onClick={() => setShowAll(!showAll)} className="shrink-0 text-[11px] text-muted underline hover:text-ink"
                  title="Low-severity hotspots are hidden by default; dismissed ones stay hidden">
            {showAll ? "hide low" : `+${hidden} low`}{dismissed.size ? ` · ${dismissed.size} dismissed` : ""}
          </button>
        )}
        {hotspots.map((h) => (
          <button
            key={h.id}
            onClick={() => focusNode(h.node_id)}
            className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1 text-[12px] transition hover:bg-panel-2 ${
              selected === h.node_id ? "bg-panel-2" : ""}`}
            style={{ borderColor: `${SEVERITY_COLOR[h.severity]}88` }}
            title={h.why}
          >
            <span className="h-2 w-2 rounded-full" style={{ background: SEVERITY_COLOR[h.severity] }} />
            <span className="font-semibold">{labels[h.node_id] ?? h.node_id}</span>
            <span className="text-muted">{h.title}</span>
          </button>
        ))}
      </div>
      <div className="pointer-events-none absolute top-0 right-0 bottom-0 w-10 bg-gradient-to-l from-panel to-transparent" />
    </div>
  );
}
