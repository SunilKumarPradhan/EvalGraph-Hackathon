import { GitBranch, Network } from "lucide-react";
import { useFixtures } from "../api/client";
import { useAppStore, type Mode } from "../store/useAppStore";
import { shortCommit } from "../lib/format";

export function ModeBadge() {
  const health = useAppStore((s) => s.health);
  const healthError = useAppStore((s) => s.healthError);
  const pill = "rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap";
  if (healthError) return <span className={`${pill} border-bad text-bad`}>API offline</span>;
  if (!health) return null;
  if (useFixtures) return <span className={`${pill} border-warn text-warn`}>FIXTURES</span>;
  if (health.replay) return <span className={`${pill} border-warn text-warn`} title="Replay mode: recorded runs of the demo are served when they match (labelled REPLAY); everything else runs live">REPLAY</span>;
  if (health.mock) return <span className={`${pill} border-warn text-warn`} title="No DIAL calls; canned LLM output">MOCK</span>;
  return <span className={`${pill} border-accent text-accent`}>DIAL · {health.model}</span>;
}

export function Logo({ onClick }: { onClick?: () => void }) {
  return (
    <button onClick={onClick} className="flex items-center gap-2 text-[17px] font-semibold tracking-wide">
      <Network size={20} className="text-accent" />
      <span>
        Eval<span className="text-accent">Graph</span>
      </span>
    </button>
  );
}

export function TopBar({ right }: { right?: React.ReactNode }) {
  const { analysis, mode, setMode, goHome } = useAppStore();
  const modes: [Mode, string][] = [["architecture", "Architecture"], ["evaluation", "Evaluation"]];
  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line bg-panel px-4">
      <Logo onClick={goHome} />
      {analysis && (
        <div className="flex shrink-0 items-center gap-2 text-sm">
          <span className="max-w-[14rem] truncate font-medium">{analysis.source.name}</span>
          {analysis.source.commit && (
            <span className="hidden items-center gap-1 font-mono text-xs text-muted 2xl:flex">
              <GitBranch size={12} />
              {shortCommit(analysis.source.commit)}
            </span>
          )}
        </div>
      )}
      <ModeBadge />
      <div className="flex-1" />
      {analysis && (
        <div className="flex rounded-lg border border-line bg-canvas p-0.5">
          {modes.map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3.5 py-1.5 text-xs font-semibold tracking-wider uppercase transition ${
                mode === m ? "bg-accent text-canvas" : "text-muted hover:text-ink"}`}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {right}
    </header>
  );
}
