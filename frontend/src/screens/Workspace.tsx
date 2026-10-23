import { ReactFlowProvider } from "@xyflow/react";
import { ClipboardList, FileBarChart, Settings2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { Canvas } from "../components/canvas/Canvas";
import type { ComponentNodeData } from "../components/canvas/ComponentNode";
import { AttachmentChips } from "../components/eval/AttachmentChips";
import { DRAG_MIME, EvaluatorPalette } from "../components/eval/EvaluatorPalette";
import { RunControls, RunProgress } from "../components/eval/RunControls";
import { RunSetup } from "../components/eval/RunSetup";
import { HotspotBar } from "../components/HotspotBar";
import { NodeDrawer } from "../components/NodeDrawer";
import { CompareView } from "../components/results/CompareView";
import { EvaluationPlan } from "../components/results/EvaluationPlan";
import { NodeResults } from "../components/results/NodeResults";
import { ResultBadges } from "../components/results/ResultBadge";
import { RunHistory } from "../components/results/RunHistory";
import { RunReport } from "../components/results/RunReport";
import { SweepView } from "../components/results/SweepView";
import { worstStatus } from "../components/results/status";
import { SummaryPanel } from "../components/SummaryPanel";
import { TopBar } from "../components/TopBar";
import { useAppStore } from "../store/useAppStore";
import { useEvalStore } from "../store/useEvalStore";
import { useRunStore } from "../store/useRunStore";
import { toast } from "../store/useToast";
import type { AnalysisResult, NodeEvalResult } from "../types/contracts";

function nodeIdAt(x: number, y: number): string | null {
  const el = document.elementFromPoint(x, y)?.closest(".react-flow__node");
  return el?.getAttribute("data-id") ?? null;
}

export function Workspace({ analysis }: { analysis: AnalysisResult }) {
  const mode = useAppStore((s) => s.mode);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const { evaluators, attachments, dragging, init, attach, setDragging, spec } = useEvalStore();
  const resetRuns = useRunStore((s) => s.reset);
  const run = useRunStore((s) => (s.activeRunId ? s.runs[s.activeRunId] ?? null : null));
  const { reportOpen, setReportOpen } = useRunStore();
  const [setupOpen, setSetupOpen] = useState(false);
  // the sandbox plan is editable for any repo without a hand-written adapter
  const setupEditable = analysis.summary.adapter === "sandbox" || !analysis.summary.runnable;

  useEffect(() => {
    api.evaluators().then((list) => init(analysis, list)).catch(() => init(analysis, []));
    resetRuns(analysis.id);
  }, [analysis, init, resetRuns]);

  // repos we can't run get their evaluation plan as soon as the user switches to evaluation mode
  useEffect(() => {
    if (mode === "evaluation" && !analysis.summary.runnable) setReportOpen(true);
  }, [mode, analysis.summary.runnable, setReportOpen]);

  const evaluatorName = (id: string) => evaluators.find((e) => e.id === id)?.name ?? id;
  const showResults = mode === "evaluation" && run !== null;

  const decorations = useMemo(() => {
    if (mode !== "evaluation") return undefined;
    const draggingSpec = dragging ? spec(dragging) : undefined;
    const chipsByNode: Record<string, string[]> = {};
    for (const a of attachments) (chipsByNode[a.node_id] ??= []).push(a.evaluator_id);
    const resultsByNode: Record<string, NodeEvalResult[]> = {};
    if (run) for (const r of run.results) (resultsByNode[r.node_id] ??= []).push(r);
    const worstNode = run?.bottlenecks.find((b) => b.status === "red")?.node_id;
    const nameOf = (id: string) => spec(id)?.name ?? id;
    const out: Record<string, Partial<ComponentNodeData>> = {};
    analysis.graph.nodes.forEach((n, i) => {
      const valid = draggingSpec ? draggingSpec.applies_to.includes(n.type) : undefined;
      const results = resultsByNode[n.id];
      const chips = chipsByNode[n.id];
      out[n.id] = {
        dropTarget: valid === true,
        dimmed: valid === false,
        resultStatus: results ? worstStatus(results) : null,
        pulse: n.id === worstNode,
        footer: results?.length ? <ResultBadges results={results} nameOf={nameOf} baseDelay={(i % 12) * 60} />
          : chips?.length ? <AttachmentChips nodeId={n.id} evaluatorIds={chips} /> : undefined,
      };
    });
    return out;
  }, [mode, dragging, attachments, analysis.graph.nodes, spec, run]);

  const onDragOver = (e: React.DragEvent) => {
    if (mode !== "evaluation" || !e.dataTransfer.types.includes(DRAG_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };

  const onDrop = (e: React.DragEvent) => {
    const evaluatorId = e.dataTransfer.getData(DRAG_MIME);
    setDragging(null);
    if (!evaluatorId) return;
    e.preventDefault();
    const nodeId = nodeIdAt(e.clientX, e.clientY);
    if (!nodeId) {
      toast("Drop the evaluator onto a component.");
      return;
    }
    attach(analysis, nodeId, evaluatorId);
  };

  const panelButton = mode === "evaluation" && !reportOpen && (run || !analysis.summary.runnable) ? (
    <button onClick={() => setReportOpen(true)} className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-[12px] hover:border-accent">
      {run ? <FileBarChart size={14} /> : <ClipboardList size={14} />} {run ? "Report" : "Plan"}
    </button>
  ) : null;

  const setupButton = mode === "evaluation" && setupEditable ? (
    <button onClick={() => { setSetupOpen(!setupOpen); useAppStore.getState().select(null); }}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[12px] ${setupOpen ? "border-accent text-accent" : "border-line hover:border-accent"}`}
            title="How the sandbox calls this app, and with which questions">
      <Settings2 size={14} /> Run setup{analysis.runtime?.plan?.edited ? " ·" : ""}
    </button>
  ) : null;

  const rightPanel = setupOpen && mode === "evaluation" && !selectedNodeId
    ? <RunSetup analysis={analysis} onClose={() => setSetupOpen(false)} />
    : selectedNodeId
    ? <NodeDrawer analysis={analysis} evaluatorName={evaluatorName}
                  extra={showResults ? <NodeResults nodeId={selectedNodeId} run={run} /> : undefined} />
    : mode === "evaluation" && reportOpen
      ? analysis.summary.runnable ? <RunReport analysis={analysis} /> : <EvaluationPlan analysis={analysis} />
      : null;

  return (
    <ReactFlowProvider>
      <div className="flex h-full flex-col">
        <TopBar right={mode === "evaluation" ? (
          <div className="flex items-center gap-2">
            <RunHistory />
            {setupButton}
            {panelButton}
            <RunControls analysis={analysis} />
          </div>
        ) : undefined} />
        <div className="flex min-h-0 flex-1">
          <aside className="w-[300px] shrink-0 overflow-y-auto border-r border-line bg-panel">
            {mode === "architecture" ? <SummaryPanel analysis={analysis} /> : <EvaluatorPalette analysis={analysis} />}
          </aside>
          <main className="relative min-w-0 flex-1">
            <Canvas analysis={analysis} decorations={decorations} onDragOver={onDragOver} onDrop={onDrop} fitKey={mode} roomy={mode === "evaluation"} />
            {analysis.graph.nodes.length === 0 && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-8">
                <div className="max-w-md rounded-xl border border-line bg-panel/95 p-5 text-center">
                  <div className="text-[15px] font-semibold">No AI components found</div>
                  <p className="mt-1.5 text-[13px] text-muted">
                    EvalGraph found no LLM calls, retrieval, agents or tools in this repo
                    ({analysis.summary.files_scanned} code files scanned). Is it an LLM application, or does it call models
                    through a library EvalGraph doesn&apos;t recognise yet?
                  </p>
                </div>
              </div>
            )}
            <RunProgress />
            <CompareView analysis={analysis} />
            <SweepView analysis={analysis} />
          </main>
          {rightPanel}
        </div>
        <HotspotBar analysis={analysis} />
      </div>
    </ReactFlowProvider>
  );
}
