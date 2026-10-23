import { X } from "lucide-react";
import { useToast } from "../store/useToast";

const KIND = { info: "border-line", error: "border-bad text-bad", success: "border-good text-good" };

export function Toasts() {
  const { toasts, dismiss } = useToast();
  return (
    <div className="pointer-events-none fixed right-4 bottom-16 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <div key={t.id} className={`eg-pop pointer-events-auto flex max-w-sm items-start gap-2 rounded-lg border bg-panel px-3 py-2 text-[13px] shadow-xl ${KIND[t.kind]}`}>
          <span className="flex-1">{t.text}</span>
          <button onClick={() => dismiss(t.id)} className="text-muted hover:text-ink" aria-label="Dismiss"><X size={14} /></button>
        </div>
      ))}
    </div>
  );
}
