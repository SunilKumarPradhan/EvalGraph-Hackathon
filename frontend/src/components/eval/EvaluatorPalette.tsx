import { GripVertical, Trash2, Wand2 } from "lucide-react";
import { useAppStore } from "../../store/useAppStore";
import { useEvalStore } from "../../store/useEvalStore";
import { toast } from "../../store/useToast";
import type { AnalysisResult, EvalCategory, EvaluatorSpec } from "../../types/contracts";

export const DRAG_MIME = "application/evalgraph-evaluator";

// golden-set fields a generated dataset never has (it does carry refusal probes)
const LABEL_FIELDS = new Set(["relevant_doc_ids", "reference_answer", "expected_numeric", "expected_route", "expected_tool",
  "expected_findings", "history"]);

/** Why this evaluator can't score on this repo's dataset, or null. Sandbox runs use generated, unlabelled questions. */
function unavailableReason(spec: EvaluatorSpec, analysis: AnalysisResult): string | null {
  if (analysis.summary.adapter !== "sandbox") return null;
  const missing = spec.needs.filter((n) => LABEL_FIELDS.has(n));
  return missing.length ? `Needs labelled ${missing.join(", ")}; this repo runs on generated questions without labels.` : null;
}

const CATEGORIES: [EvalCategory, string][] = [
  ["rag", "RAG"], ["llm", "LLM"], ["agent", "Agent & Tools"], ["safety", "Safety"], ["ops", "Ops"],
];

function Card({ spec, analysis }: { spec: EvaluatorSpec; analysis: AnalysisResult }) {
  const { setDragging, attach, attachments } = useEvalStore();
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const used = attachments.filter((a) => a.evaluator_id === spec.id).length;
  const unavailable = unavailableReason(spec, analysis);
  if (unavailable) {
    return (
      <div title={unavailable} className="flex items-start gap-2 rounded-lg border border-line/60 bg-panel-2/40 px-2.5 py-2 opacity-45">
        <GripVertical size={14} className="mt-0.5 shrink-0 text-muted" />
        <div className="min-w-0 flex-1">
          <span className="text-[13px] font-semibold">{spec.name}</span>
          <div className="text-[11px] leading-snug text-muted">{unavailable}</div>
        </div>
      </div>
    );
  }
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_MIME, spec.id);
        e.dataTransfer.effectAllowed = "copy";
        setDragging(spec.id);
      }}
      onDragEnd={() => setDragging(null)}
      onClick={() => {
        if (selectedNodeId) attach(analysis, selectedNodeId, spec.id);
        else toast("Drag onto a component, or select a component first and click the evaluator.");
      }}
      title={`${spec.description}\nApplies to: ${spec.applies_to.join(", ")}`}
      className="group flex cursor-grab items-start gap-2 rounded-lg border border-line bg-panel-2 px-2.5 py-2 hover:border-accent active:cursor-grabbing"
    >
      <GripVertical size={14} className="mt-0.5 shrink-0 text-muted group-hover:text-accent" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="text-[13px] font-semibold">{spec.name}</span>
          {used > 0 && <span className="rounded-full bg-accent px-1.5 text-[10px] font-bold text-canvas">{used}</span>}
        </div>
        <div className="line-clamp-2 text-[11px] leading-snug text-muted">{spec.description}</div>
        <div className="mt-1 text-[10px] text-muted/80">{spec.method === "llm_judge" ? "LLM judge" : "deterministic"} · {spec.unit}</div>
      </div>
    </div>
  );
}

export function EvaluatorPalette({ analysis }: { analysis: AnalysisResult }) {
  const { evaluators, attachments, applyRecommended, clear } = useEvalStore();
  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <div className="text-[11px] font-semibold tracking-wider text-muted uppercase">Evaluators</div>
        <span className="text-[11px] text-muted">{attachments.length} attached</span>
      </div>
      <div className="flex gap-2">
        <button onClick={() => void applyRecommended(analysis)}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-accent px-2 py-2 text-[12px] font-semibold text-canvas hover:brightness-110">
          <Wand2 size={14} /> Apply recommended
        </button>
        <button onClick={clear} disabled={!attachments.length} title="Remove all attachments"
          className="rounded-lg border border-line px-2.5 text-muted hover:text-ink disabled:opacity-40">
          <Trash2 size={14} />
        </button>
      </div>
      <p className="text-[11px] leading-snug text-muted">
        Drag an evaluator onto a component. Valid targets light up. Or select a component and click an evaluator.
      </p>
      {CATEGORIES.map(([cat, label]) => {
        const specs = evaluators.filter((e) => e.category === cat);
        if (!specs.length) return null;
        return (
          <section key={cat} className="flex flex-col gap-1.5">
            <div className="text-[11px] font-semibold text-accent">{label}</div>
            {specs.map((s) => <Card key={s.id} spec={s} analysis={analysis} />)}
          </section>
        );
      })}
    </div>
  );
}
