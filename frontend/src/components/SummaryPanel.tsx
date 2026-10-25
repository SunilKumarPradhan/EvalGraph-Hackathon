import { AlertTriangle, CheckCircle2, FlaskConical } from "lucide-react";
import { useEffect, useState } from "react";
import type { AnalysisResult, NodeType } from "../types/contracts";
import { TYPE_STYLE } from "./canvas/typeStyle";

function CountUp({ to, ms = 600 }: { to: number; ms?: number }) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      setValue(Math.round(to * (1 - (1 - t) ** 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [to, ms]);
  return <>{value}</>;
}

export function SummaryPanel({ analysis }: { analysis: AnalysisResult }) {
  const { summary } = analysis;
  const counts = Object.entries(summary.counts).sort((a, b) => b[1] - a[1]) as [NodeType, number][];
  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <div className="text-[11px] font-semibold tracking-wider text-muted uppercase">Detected</div>
        <p className="mt-1 text-[13px] leading-snug">{summary.headline.split(" · ")[0]}</p>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {counts.map(([type, n]) => {
          const s = TYPE_STYLE[type] ?? TYPE_STYLE.other;
          const Icon = s.icon;
          return (
            <div key={type} className="flex items-center gap-2 rounded-lg border border-line bg-panel-2 px-2.5 py-2">
              <Icon size={15} color={s.color} />
              <span className="text-lg font-semibold tabular-nums"><CountUp to={n} /></span>
              <span className="truncate text-[11px] text-muted">{s.name}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2.5">
        <AlertTriangle size={18} className="text-bad" />
        <div>
          <div className="text-xl leading-none font-bold"><CountUp to={summary.eval_gaps} /></div>
          <div className="text-[11px] text-muted">evaluation gaps found</div>
        </div>
      </div>
      <div
        className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-[12px] ${summary.runnable ? "border-good/40 text-good" : "border-line text-muted"}`}
      >
        {summary.runnable ? <CheckCircle2 size={15} className="mt-px shrink-0" /> : <FlaskConical size={15} className="mt-px shrink-0" />}
        <span>
          {!summary.runnable
            ? `Plan mode: ${String(analysis.runtime?.plan?.reason ?? "no way to run this repo was found")}. You still get the graph, hotspots and an evaluation plan.`
            : summary.adapter === "sandbox"
              ? `Runnable in an isolated Docker sandbox · ${String(analysis.runtime?.plan?.reason ?? "")} · ${analysis.runtime?.dataset.length ?? 0} generated test questions`
              : `Runnable via the "${summary.adapter}" adapter · ${summary.variants.length} config variants`}
        </span>
      </div>
      <div>
        <div className="mb-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">Frameworks</div>
        <div className="flex flex-wrap gap-1.5">
          {summary.frameworks.map((f) => (
            <span key={f} className="rounded-md bg-panel-2 px-2 py-0.5 text-[11px]">{f}</span>
          ))}
        </div>
      </div>
      {summary.languages && summary.languages.length > 0 && (
        <div className="text-[11px] text-muted">Languages: {summary.languages.join(", ")}</div>
      )}
      {(summary.truncated || (summary.skipped_files ?? 0) > 0) && (
        <div className="rounded-lg border border-warn/50 bg-warn/10 px-3 py-2 text-[12px] text-warn">
          {summary.truncated ? "Large repo: only the first files within the limit were analyzed. " : ""}
          {summary.skipped_files ? `${summary.skipped_files} file(s) over the size limit were skipped. ` : ""}
          The graph may miss components; analyze a sub-folder (…/tree/&lt;branch&gt;/&lt;folder&gt;) for a closer look.
        </div>
      )}
      <div className="text-[11px] text-muted">
        {summary.llm_calls ? `${summary.llm_calls} LLM calls · ` : "static analysis only · "}
        {summary.files_scanned} code files scanned · {analysis.graph.nodes.length} components · {analysis.graph.edges.length} connections
        {analysis.aimap && (
          <>
            {" · "}
            <a href={`/api/analyses/${analysis.id}/aimap`} target="_blank" rel="noreferrer" className="underline hover:text-accent"
               title="The whole analysis as compact, line-referenced text: paste it into any LLM">
              AI map (~{Math.round(analysis.aimap.length / 4000)}k tokens)
            </a>
          </>
        )}
      </div>
    </div>
  );
}
