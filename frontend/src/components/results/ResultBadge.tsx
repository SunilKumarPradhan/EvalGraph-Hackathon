import { AlertTriangle } from "lucide-react";
import { formatValue } from "../../lib/format";
import type { NodeEvalResult } from "../../types/contracts";
import { STATUS_COLOR } from "./status";

/** One evaluator result on a node card: short name + value, coloured by status. n/a and error carry the reason. */
export function ResultBadge({ result, name, delay = 0 }: { result: NodeEvalResult; name: string; delay?: number }) {
  const color = STATUS_COLOR[result.status];
  const muted = result.status === "na" || result.status === "error";
  const title = `${name}: ${formatValue(result.value, result.unit)} (${result.status}, n=${result.n})${result.reason ? `\n${result.reason}` : ""}`;
  return (
    <span
      className="eg-pop flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10.5px] whitespace-nowrap"
      style={{ borderColor: `${color}88`, background: `${color}1f`, color: muted ? "#94a3b8" : color, animationDelay: `${delay}ms` }}
      title={title}
    >
      {result.status === "error" && <AlertTriangle size={10} />}
      <span className="max-w-[92px] truncate opacity-85">{name}</span>
      <span className="font-bold tabular-nums">{formatValue(result.value, result.unit)}</span>
    </span>
  );
}

export function ResultBadges({ results, nameOf, baseDelay = 0 }: {
  results: NodeEvalResult[];
  nameOf: (id: string) => string;
  baseDelay?: number;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {results.map((r, i) => <ResultBadge key={r.evaluator_id} result={r} name={nameOf(r.evaluator_id)} delay={baseDelay + i * 90} />)}
    </div>
  );
}
