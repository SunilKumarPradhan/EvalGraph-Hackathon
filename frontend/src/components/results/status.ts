import type { NodeEvalResult, Status } from "../../types/contracts";

export const STATUS_COLOR: Record<Status, string> = {
  green: "#22c55e",
  amber: "#f59e0b",
  red: "#ef4444",
  error: "#64748b",
  na: "#64748b",
};

const STATUS_RANK: Record<Status, number> = { red: 0, amber: 1, error: 2, green: 3, na: 4 };

/** Worst status among a node's results; na only when nothing was scored. */
export function worstStatus(results: NodeEvalResult[]): Status | null {
  if (!results.length) return null;
  return results.reduce<Status>((w, r) => (STATUS_RANK[r.status] < STATUS_RANK[w] ? r.status : w), "na");
}

export function statusRank(s: Status): number {
  return STATUS_RANK[s];
}
