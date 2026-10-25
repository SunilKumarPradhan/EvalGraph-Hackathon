import { ClipboardList, X } from "lucide-react";
import { useAppStore } from "../../store/useAppStore";
import { useEvalStore } from "../../store/useEvalStore";
import { useRunStore } from "../../store/useRunStore";
import type { AnalysisResult } from "../../types/contracts";
import { SEVERITY_COLOR } from "../canvas/typeStyle";

/** For repos EvalGraph can't run: the evaluation plan (what to measure where, and why) is the deliverable. */
export function EvaluationPlan({ analysis }: { analysis: AnalysisResult }) {
  const spec = useEvalStore((s) => s.spec);
  const focusNode = useAppStore((s) => s.focusNode);
  const setReportOpen = useRunStore((s) => s.setReportOpen);
  const byNode = new Map<string, typeof analysis.recommendations>();
  for (const r of analysis.recommendations) byNode.set(r.node_id, [...(byNode.get(r.node_id) ?? []), r]);
  const label = (id: string) => analysis.graph.nodes.find((n) => n.id === id)?.label ?? id;

  return (
    <aside className="eg-slide flex w-[360px] shrink-0 flex-col overflow-hidden border-l border-line bg-panel">
      <div className="flex items-start gap-2 border-b border-line p-4">
        <ClipboardList size={18} className="mt-0.5 text-accent" />
        <div className="flex-1">
          <div className="text-[15px] font-semibold">Evaluation plan</div>
          <div className="mt-0.5 text-[12px] text-muted">
            This repo can't be run here ({String(analysis.runtime?.plan?.reason ?? "no run plan")}). Use this plan with your own harness.
          </div>
        </div>
        <button onClick={() => setReportOpen(false)} className="text-muted hover:text-ink" aria-label="Close plan"><X size={18} /></button>
      </div>
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-4">
        {[...byNode.entries()].map(([nodeId, recs]) => (
          <div key={nodeId} className="rounded-lg border border-line bg-panel-2 p-2.5">
            <button onClick={() => focusNode(nodeId)} className="text-[13px] font-semibold hover:text-accent">{label(nodeId)}</button>
            <ul className="mt-1 flex flex-col gap-1">
              {recs.map((r) => (
                <li key={r.evaluator_id} className="flex items-start gap-2 text-[12px]">
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: SEVERITY_COLOR[r.priority] }} />
                  <span><span className="font-medium">{spec(r.evaluator_id)?.name ?? r.evaluator_id}</span>
                    <span className="text-muted"> · {r.rationale}</span></span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </aside>
  );
}
