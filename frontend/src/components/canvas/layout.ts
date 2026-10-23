import { Graph, layout } from "@dagrejs/dagre";
import type { ApplicationGraph } from "../../types/contracts";

export const NODE_W = 232;
export const NODE_H = 84;

/** Top-to-bottom dagre layout. Returns positions and a reveal order (by rank, then left to right). */
export function layoutGraph(graph: ApplicationGraph, extraHeight = 0): { positions: Record<string, { x: number; y: number }>; order: string[] } {
  const h = NODE_H + extraHeight;
  const g = new Graph();
  g.setGraph({ rankdir: "TB", nodesep: 36, ranksep: 46, marginx: 16, marginy: 16 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of graph.nodes) g.setNode(n.id, { width: NODE_W, height: h });
  for (const e of graph.edges) g.setEdge(e.source, e.target);
  layout(g);
  const positions: Record<string, { x: number; y: number }> = {};
  for (const n of graph.nodes) {
    const p = g.node(n.id);
    positions[n.id] = { x: (p?.x ?? 0) - NODE_W / 2, y: (p?.y ?? 0) - h / 2 };
  }
  const order = [...graph.nodes]
    .map((n) => n.id)
    .sort((a, b) => positions[a].y - positions[b].y || positions[a].x - positions[b].x);
  return { positions, order };
}
