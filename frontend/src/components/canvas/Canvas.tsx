import {
  Background, BackgroundVariant, Controls, MarkerType, MiniMap, ReactFlow, useNodesInitialized, useNodesState, useReactFlow,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../../store/useAppStore";
import type { AnalysisResult, Hotspot, Severity } from "../../types/contracts";
import { ComponentNode, type ComponentFlowNode, type ComponentNodeData } from "./ComponentNode";
import { layoutGraph } from "./layout";
import { SEVERITY_RANK, TYPE_STYLE } from "./typeStyle";

const nodeTypes = { component: ComponentNode };
const REVEAL_MS = 110;

export interface CanvasDecorations {
  /** Extra per-node data (evaluation mode adds attachment chips, drop highlighting, results). Memoize it. */
  decorations?: Record<string, Partial<ComponentNodeData>>;
  onDragOver?: React.DragEventHandler;
  onDrop?: React.DragEventHandler;
  /** Changing this re-frames the whole graph (e.g. on mode switch). */
  fitKey?: string;
  /** Reserve vertical room under each node (evaluation mode shows attachment chips there). */
  roomy?: boolean;
}

function worstSeverity(hotspots: Hotspot[]): Severity | null {
  return hotspots.reduce<Severity | null>(
    (worst, h) => (worst === null || SEVERITY_RANK[h.severity] < SEVERITY_RANK[worst] ? h.severity : worst), null);
}

export function Canvas({ analysis, decorations, onDragOver, onDrop, fitKey, roomy = false }: { analysis: AnalysisResult } & CanvasDecorations) {
  const select = useAppStore((s) => s.select);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const focusRequest = useAppStore((s) => s.focusRequest);
  const flow = useReactFlow();
  const flowRef = useRef(flow);
  flowRef.current = flow;
  const { positions } = useMemo(() => layoutGraph(analysis.graph, roomy ? 48 : 0), [analysis.graph, roomy]);
  // reveal order must not change when only spacing changes
  const revealOrder = useMemo(() => layoutGraph(analysis.graph).order, [analysis.graph]);
  const lastPositions = useRef(positions);
  const [revealed, setRevealed] = useState(0);

  // The demo money shot: components appear one by one, top to bottom, then the camera frames them.
  useEffect(() => {
    setRevealed(0);
    let i = 0;
    const timer = setInterval(() => {
      i += 1;
      setRevealed(i);
      if (i >= revealOrder.length) clearInterval(timer);
    }, REVEAL_MS);
    return () => clearInterval(timer);
  }, [analysis.id, revealOrder.length]);

  // Frame the whole graph as soon as React Flow has measured every node (they render invisibly first).
  const initialized = useNodesInitialized();
  useEffect(() => {
    if (initialized) void flowRef.current.fitView({ padding: 0.08, duration: 400 });
  }, [initialized, analysis.id, fitKey]);

  useEffect(() => {
    if (focusRequest) void flowRef.current.fitView({ nodes: [{ id: focusRequest.nodeId }], padding: 1.6, duration: 500, maxZoom: 1.2 });
  }, [focusRequest]);

  const hotspotsByNode = useMemo(() => {
    const map: Record<string, Hotspot[]> = {};
    const dismissed = new Set(analysis.dismissed_hotspots ?? []);
    for (const h of analysis.hotspots) if (!dismissed.has(h.id)) (map[h.node_id] ??= []).push(h);
    return map;
  }, [analysis.hotspots, analysis.dismissed_hotspots]);

  const visible = useMemo(() => new Set(revealOrder.slice(0, revealed)), [revealOrder, revealed]);

  // Nodes live in React Flow state so measured sizes (and user drags) survive our data updates.
  const [nodes, setNodes, onNodesChange] = useNodesState<ComponentFlowNode>([]);
  useEffect(() => {
    const relayout = lastPositions.current !== positions;
    lastPositions.current = positions;
    setNodes((prev) => {
      const previous = new Map(prev.map((n) => [n.id, n]));
      return analysis.graph.nodes.map((n) => {
        const hs = hotspotsByNode[n.id] ?? [];
        const data: ComponentNodeData = {
          node: n, severity: worstSeverity(hs), hotspotCount: hs.length, revealed: visible.has(n.id),
          ...(decorations?.[n.id] ?? {}),
        };
        const old = previous.get(n.id);
        return old
          ? { ...old, data, selected: n.id === selectedNodeId, ...(relayout ? { position: positions[n.id] } : {}) }
          : { id: n.id, type: "component" as const, position: positions[n.id], data, selected: n.id === selectedNodeId };
      });
    });
  }, [analysis.graph.nodes, positions, hotspotsByNode, visible, decorations, selectedNodeId, setNodes]);

  const edges: Edge[] = analysis.graph.edges.map((e) => {
    const dataFlow = e.kind === "feeds" || e.kind === "routes_to";
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      type: "smoothstep",
      animated: dataFlow,
      hidden: !(visible.has(e.source) && visible.has(e.target)),
      label: e.kind === "routes_to" ? e.label ?? undefined : undefined,
      labelStyle: { fill: "#8b97ad", fontSize: 11 },
      labelBgStyle: { fill: "#0b0f17" },
      labelBgPadding: [4, 2] as [number, number],
      style: { stroke: e.kind === "uses" ? "#3a4760" : "#4b5b78", strokeWidth: 1.4, strokeDasharray: e.kind === "uses" ? "4 4" : undefined },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#4b5b78", width: 16, height: 16 },
    };
  });

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeClick={(_, node) => select(node.id)}
      onPaneClick={() => select(null)}
      onDragOver={onDragOver}
      onDrop={onDrop}
      nodesConnectable={false}
      minZoom={0.2}
      maxZoom={1.8}
      proOptions={{ hideAttribution: true }}
      colorMode="dark"
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#1f2a3c" />
      <Controls showInteractive={false} position="bottom-right" />
      <MiniMap
        pannable
        zoomable
        position="top-right"
        style={{ background: "#0e1420", width: 150, height: 100 }}
        nodeColor={(n) => TYPE_STYLE[(n.data as ComponentNodeData).node.type].color}
        maskColor="rgba(11,15,23,.7)"
      />
    </ReactFlow>
  );
}
