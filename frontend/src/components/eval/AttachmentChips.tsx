import { X } from "lucide-react";
import { useEvalStore } from "../../store/useEvalStore";

export function AttachmentChips({ nodeId, evaluatorIds }: { nodeId: string; evaluatorIds: string[] }) {
  const { detach, spec } = useEvalStore();
  return (
    <div className="flex flex-wrap gap-1">
      {evaluatorIds.map((id) => (
        <span key={id} className="eg-pop nodrag flex items-center gap-1 rounded-md border border-accent/50 bg-accent/10 py-0.5 pr-1 pl-1.5 text-[10.5px] text-accent">
          {spec(id)?.name ?? id}
          <button
            className="nodrag rounded text-accent/70 hover:bg-accent/20 hover:text-ink"
            aria-label={`Remove ${spec(id)?.name ?? id}`}
            onClick={(e) => {
              e.stopPropagation();
              detach(nodeId, id);
            }}
          >
            <X size={11} />
          </button>
        </span>
      ))}
    </div>
  );
}
