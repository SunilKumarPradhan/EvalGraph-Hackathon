import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { memo } from "react";
import type { GraphNode, Severity, Status } from "../../types/contracts";
import { STATUS_COLOR } from "../results/status";
import { NODE_H, NODE_W } from "./layout";
import { SEVERITY_COLOR, TYPE_STYLE } from "./typeStyle";

export type ComponentNodeData = {
  node: GraphNode;
  severity: Severity | null;
  hotspotCount: number;
  revealed?: boolean;
  dimmed?: boolean;
  dropTarget?: boolean;
  footer?: React.ReactNode;
  /** Worst result status of the shown run; replaces the hotspot ring once there are real scores. */
  resultStatus?: Status | null;
  /** The worst bottleneck of the run. */
  pulse?: boolean;
};

export type ComponentFlowNode = Node<ComponentNodeData, "component">;

function ComponentNodeView({ data, selected }: NodeProps<ComponentFlowNode>) {
  const { node, severity, hotspotCount, revealed = true, dimmed, dropTarget, footer, resultStatus, pulse } = data;
  const style = TYPE_STYLE[node.type];
  const Icon = style.icon;
  const scored = resultStatus && resultStatus !== "na";
  const ring = scored ? STATUS_COLOR[resultStatus] : severity ? SEVERITY_COLOR[severity] : null;
  const pulsing = scored ? Boolean(pulse) : severity === "high";
  const inferred = node.confidence < 0.9;

  return (
    <div
      className={`relative rounded-xl border bg-panel text-ink transition-[opacity,box-shadow,border-color] duration-200 ${revealed ? "eg-pop" : "invisible"}`}
      style={{
        width: NODE_W,
        minHeight: NODE_H,
        borderColor: selected ? "#e6ebf5" : dropTarget ? "#2dd4bf" : "#263146",
        borderStyle: inferred || dropTarget ? "dashed" : "solid",
        opacity: dimmed ? 0.35 : 1,
        boxShadow: dropTarget ? "0 0 0 3px rgba(45,212,191,.35), 0 0 22px rgba(45,212,191,.35)"
          : ring ? `0 0 0 2px ${ring}, 0 0 18px ${ring}66` : "0 4px 14px rgba(0,0,0,.35)",
      }}
      title={inferred ? "Inferred by LLM (lower confidence)" : undefined}
    >
      {pulsing && ring && (
        <span className="pointer-events-none absolute -inset-[3px] animate-pulse rounded-xl" style={{ boxShadow: `0 0 0 2px ${ring}` }} />
      )}
      <Handle type="target" position={Position.Top} className="!h-2 !w-2 !border-0 !bg-line" />
      <div className="flex items-start gap-2.5 px-3 pt-2.5 pb-2">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg" style={{ background: `${style.color}22` }}>
          <Icon size={17} color={style.color} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] leading-tight font-semibold">{node.label}</div>
          <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted">
            <span style={{ color: style.color }}>{style.name}</span>
            {node.provider && <span className="rounded bg-panel-2 px-1.5 py-px">{node.provider}</span>}
          </div>
          <div className="mt-0.5 truncate font-mono text-[10.5px] text-muted/80">{node.symbol ? `${node.symbol}()` : node.file}</div>
        </div>
        {hotspotCount > 0 && severity && (
          <span className="rounded-full px-1.5 text-[11px] font-bold text-canvas" style={{ background: SEVERITY_COLOR[severity] }} title={`${hotspotCount} hotspot(s)`}>
            {hotspotCount}
          </span>
        )}
      </div>
      {footer && <div className="border-t border-line px-2 py-1.5">{footer}</div>}
      <Handle type="source" position={Position.Bottom} className="!h-2 !w-2 !border-0 !bg-line" />
    </div>
  );
}

export const ComponentNode = memo(ComponentNodeView);
