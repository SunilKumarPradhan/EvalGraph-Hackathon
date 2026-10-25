import { ChevronDown, ChevronRight, Download, EyeOff, HelpCircle, Wand2, X } from "lucide-react";
import { api, useFixtures } from "../api/client";
import { useState } from "react";
import { fileLine, formatConfig } from "../lib/format";
import { useAppStore } from "../store/useAppStore";
import { useEvalStore } from "../store/useEvalStore";
import { useRunStore } from "../store/useRunStore";
import type { AnalysisResult, Evidence, Hotspot, PatchEdit } from "../types/contracts";
import { SEVERITY_COLOR, SEVERITY_RANK, TYPE_STYLE } from "./canvas/typeStyle";

function EvidenceList({ items }: { items: Evidence[] }) {
  return (
    <ul className="flex flex-col gap-1.5">
      {items.map((e, i) => (
        <li key={i} className="rounded-md bg-canvas px-2.5 py-1.5">
          <div className="font-mono text-[11px] text-accent">{fileLine(e.file, e.line)}</div>
          <div className="text-[12px] font-medium">{e.fact}</div>
          {e.snippet && <div className="mt-0.5 truncate font-mono text-[11px] text-muted" title={e.snippet}>{e.snippet}</div>}
        </li>
      ))}
    </ul>
  );
}

function PatchView({ h, analysis }: { h: Hotspot; analysis: AnalysisResult }) {
  const { startRun, startSweep, running } = useEvalStore();
  const run = useRunStore((s) => (s.activeRunId ? s.runs[s.activeRunId] ?? null : null));
  const edits: PatchEdit[] = h.patch ?? [];
  const sweepable = edits.find((e) => e.span && e.sweep?.length);
  if (!edits.length) return null;
  // patches are tried in the Docker sandbox; adapter repos (CUDA Archer) use their config variants instead
  const canTry = analysis.summary.adapter === "sandbox" && run !== null && !(run.patches ?? []).includes(h.id);
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">Suggested patch (rule-based)</div>
      <ul className="flex flex-col gap-1.5">
        {edits.map((e) => (
          <li key={`${e.file}:${e.line}`} className="overflow-hidden rounded-md bg-canvas font-mono text-[11px]">
            <div className="px-2 pt-1 text-accent">{fileLine(e.file, e.line)} <span className="text-muted">· {e.summary}</span></div>
            <div className="truncate bg-bad/10 px-2 text-bad/90" title={e.before}>- {e.before.trim()}</div>
            <div className="truncate bg-good/10 px-2 pb-1 text-good" title={e.after}>+ {e.after.trim()}</div>
          </li>
        ))}
      </ul>
      <div className="mt-2 flex gap-2">
        <a href={`/api/analyses/${analysis.id}/patch?hotspots=${encodeURIComponent(h.id)}`} download
           className="flex items-center gap-1 rounded-md border border-line px-2 py-1 text-[11px] hover:border-accent">
          <Download size={12} /> .patch
        </a>
        {canTry && (
          <button disabled={running} onClick={() => void startRun(analysis, { patches: [h.id], baseRunId: run!.id })}
                  className="flex items-center gap-1 rounded-md bg-accent px-2 py-1 text-[11px] font-bold text-canvas hover:brightness-110 disabled:opacity-40"
                  title="Re-run the shown run in the sandbox with this patch applied, then compare">
            <Wand2 size={12} /> Try this fix
          </button>
        )}
        {canTry && sweepable && (
          <button disabled={running} onClick={() => void startSweep(analysis, h.id, run!.id)}
                  className="flex items-center gap-1 rounded-md border border-accent/60 px-2 py-1 text-[11px] text-accent hover:bg-accent/10 disabled:opacity-40"
                  title={`One run per value, each changing only ${sweepable.summary.split(" ")[0]}`}>
            Sweep {sweepable.summary.split(" ")[0]}: {sweepable.sweep!.join(" · ")}
          </button>
        )}
      </div>
    </div>
  );
}

function DismissButton({ h, analysis }: { h: Hotspot; analysis: AnalysisResult }) {
  const setAnalysis = useAppStore((s) => s.setAnalysis);
  const dismissed = (analysis.dismissed_hotspots ?? []).includes(h.id);
  if (useFixtures) return null;
  const toggle = async () => {
    const ids = dismissed ? (analysis.dismissed_hotspots ?? []).filter((x) => x !== h.id) : [...(analysis.dismissed_hotspots ?? []), h.id];
    setAnalysis(await api.saveDismissed(analysis.id, ids));
  };
  return (
    <button onClick={() => void toggle()} className="flex items-center gap-1 text-[11px] text-muted hover:text-ink"
            title="Mark as not relevant for this repo (hidden from the graph and hotspot bar for everyone)">
      <EyeOff size={12} /> {dismissed ? "Restore" : "Not relevant"}
    </button>
  );
}

function HotspotCard({ h, defaultOpen, evaluatorName, analysis }: {
  h: Hotspot; defaultOpen: boolean; evaluatorName: (id: string) => string; analysis: AnalysisResult;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const color = SEVERITY_COLOR[h.severity];
  return (
    <div className="rounded-lg border bg-panel-2" style={{ borderColor: `${color}66` }}>
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left" onClick={() => setOpen(!open)}>
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
        <span className="flex-1 text-[13px] font-semibold">{h.title}</span>
        {h.source && h.source !== "static" && (
          <span className="rounded border border-accent/40 px-1 text-[10px] text-accent" title="Found or confirmed by LLM review">
            {h.source === "llm" ? "LLM" : "LLM ✓"}
          </span>
        )}
        <span className="flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 text-[11px] font-bold text-accent">
          <HelpCircle size={12} /> WHY?
        </span>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open && (
        <div className="flex flex-col gap-3 px-3 pb-3">
          <p className="text-[13px] leading-relaxed">{h.why}</p>
          <div>
            <div className="mb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">Evidence</div>
            <EvidenceList items={h.evidence} />
          </div>
          <div>
            <div className="mb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">Potential impact</div>
            <ul className="list-disc pl-5 text-[12px]">{h.impacts.map((i) => <li key={i}>{i}</li>)}</ul>
          </div>
          <div>
            <div className="mb-1 text-[11px] font-semibold tracking-wider text-muted uppercase">Recommended evaluation</div>
            <div className="flex flex-wrap gap-1.5">
              {h.recommended_evaluators.map((e) => (
                <span key={e} className="rounded-md border border-accent/40 bg-accent/10 px-2 py-0.5 text-[11px] text-accent">→ {evaluatorName(e)}</span>
              ))}
            </div>
          </div>
          <PatchView h={h} analysis={analysis} />
          {h.suggested_variant && (
            <div className="text-[11px] text-muted">
              Fix to try: config variant <span className="font-mono text-ink">{h.suggested_variant}</span>
            </div>
          )}
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] text-muted/70">{h.rule_id}</span>
            <DismissButton h={h} analysis={analysis} />
          </div>
        </div>
      )}
    </div>
  );
}

export function NodeDrawer({ analysis, evaluatorName, extra }: {
  analysis: AnalysisResult;
  evaluatorName: (id: string) => string;
  extra?: React.ReactNode;
}) {
  const { selectedNodeId, select } = useAppStore();
  const node = analysis.graph.nodes.find((n) => n.id === selectedNodeId);
  if (!node) return null;
  const style = TYPE_STYLE[node.type];
  const Icon = style.icon;
  const hotspots = analysis.hotspots.filter((h) => h.node_id === node.id).sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
  const recs = analysis.recommendations.filter((r) => r.node_id === node.id);
  const config = Object.entries(node.config);

  return (
    <aside className="eg-slide flex w-[400px] shrink-0 flex-col overflow-hidden border-l border-line bg-panel">
      <div className="flex items-start gap-3 border-b border-line p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg" style={{ background: `${style.color}22` }}>
          <Icon size={20} color={style.color} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[16px] leading-tight font-semibold">{node.label}</div>
          <div className="mt-0.5 text-[12px]" style={{ color: style.color }}>
            {style.name}{node.provider ? ` · ${node.provider}` : ""}{node.confidence < 0.9 ? " · inferred" : ""}
          </div>
          <div className="mt-1 truncate font-mono text-[11px] text-muted" title={node.file ?? ""}>
            {node.symbol ? `${node.symbol}() · ` : ""}{fileLine(node.file, node.evidence[0]?.line ?? null)}
          </div>
        </div>
        <button onClick={() => select(null)} className="text-muted hover:text-ink" aria-label="Close"><X size={18} /></button>
      </div>
      {node.description && <p className="border-b border-line px-4 py-2.5 text-[13px] leading-snug text-ink/90">{node.description}</p>}
      <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-4">
        {extra}
        {hotspots.length > 0 && (
          <section className="flex flex-col gap-2">
            <h3 className="text-[11px] font-semibold tracking-wider text-muted uppercase">Hotspots ({hotspots.length})</h3>
            {hotspots.map((h, i) => <HotspotCard key={h.id} h={h} defaultOpen={i === 0} evaluatorName={evaluatorName} analysis={analysis} />)}
          </section>
        )}
        {config.length > 0 && (
          <section>
            <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Configuration (from code)</h3>
            <table className="w-full font-mono text-[12px]">
              <tbody>
                {config.map(([k, v]) => (
                  <tr key={k} className="border-b border-line/60">
                    <td className="py-1 pr-2 text-muted">{k}</td>
                    <td className="py-1 text-right">{formatConfig(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
        {recs.length > 0 && (
          <section>
            <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Recommended evaluators</h3>
            <div className="flex flex-wrap gap-1.5">
              {recs.map((r) => (
                <span key={r.evaluator_id} title={r.rationale} className="rounded-md border border-line bg-panel-2 px-2 py-0.5 text-[11px]">
                  {evaluatorName(r.evaluator_id)} <span className="text-muted">· {r.priority}</span>
                </span>
              ))}
            </div>
          </section>
        )}
        <section>
          <h3 className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Detected from</h3>
          <EvidenceList items={node.evidence} />
        </section>
      </div>
    </aside>
  );
}
