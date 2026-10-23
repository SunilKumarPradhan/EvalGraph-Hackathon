import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { Logo } from "../components/TopBar";
import { useAppStore } from "../store/useAppStore";

export function Analyzing() {
  const { job, source, error, goHome, analyze } = useAppStore();
  const target = source?.repo_url ?? source?.local_path ?? "";
  const progress = Math.round((job?.progress ?? 0.03) * 100);
  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 items-center px-6"><Logo onClick={goHome} /></header>
      <main className="flex flex-1 items-center justify-center px-6 pb-24">
        <div className="w-full max-w-xl">
          <div className="text-[11px] font-semibold tracking-wider text-muted uppercase">Analyzing</div>
          <div className="mt-1 truncate font-mono text-[15px]">{target}</div>
          <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-panel-2">
            <div className="h-full rounded-full bg-accent transition-all duration-500" style={{ width: `${progress}%` }} />
          </div>
          <ul className="mt-6 flex flex-col gap-2.5">
            {(job?.log ?? []).map((line, i) => (
              <li key={i} className="eg-pop flex items-center gap-2.5 text-[14px]">
                <CheckCircle2 size={17} className="shrink-0 text-good" />
                {line}
              </li>
            ))}
            {!error && (
              <li className="flex items-center gap-2.5 text-[14px] text-muted">
                <Loader2 size={17} className="shrink-0 animate-spin text-accent" />
                {job?.stage ?? "Starting"}…
              </li>
            )}
          </ul>
          {error && (
            <div className="mt-6 rounded-lg border border-bad/50 bg-bad/10 p-3">
              <div className="flex items-center gap-2 font-semibold text-bad"><XCircle size={17} /> Analysis failed</div>
              <div className="mt-1 text-[13px]">{error}</div>
              <div className="mt-3 flex gap-2">
                <button className="rounded-md bg-accent px-3 py-1.5 text-[13px] font-semibold text-canvas" onClick={() => source && void analyze(source)}>Try again</button>
                <button className="rounded-md border border-line px-3 py-1.5 text-[13px]" onClick={goHome}>Back</button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
