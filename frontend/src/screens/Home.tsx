import { ArrowRight, FolderOpen, GitBranch, History } from "lucide-react";
import { useEffect, useState } from "react";
import { Logo, ModeBadge } from "../components/TopBar";
import { useAppStore } from "../store/useAppStore";

const DEMO_URL = "https://github.com/SunilKumarPradhan/cuda-archer";
const DEMO_LOCAL = "../../cuda-archer";

export function Home() {
  const { analyze, recent, loadRecent, openAnalysis, error } = useAppStore();
  const [value, setValue] = useState("");
  const [local, setLocal] = useState(false);

  useEffect(() => {
    void loadRecent();
  }, [loadRecent]);

  const submit = (v = value, isLocal = local) => {
    const text = v.trim();
    if (!text) return;
    void analyze(isLocal ? { local_path: text } : { repo_url: text });
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 items-center gap-4 px-6">
        <Logo />
        <ModeBadge />
      </header>
      <main className="flex flex-1 flex-col items-center justify-center px-6 pb-24">
        <h1 className="text-center text-4xl leading-tight font-bold">
          Paste your LLM app&apos;s repo.
          <br />
          <span className="text-accent">See where it breaks.</span>
        </h1>
        <p className="mt-3 max-w-xl text-center text-[15px] text-muted">
          EvalGraph reads the code without running it, draws the architecture, flags evaluation hotspots with evidence,
          and lets you drag evaluators onto the graph.
        </p>
        <form
          className="mt-8 flex w-full max-w-2xl items-center gap-2 rounded-xl border border-line bg-panel p-2 shadow-2xl focus-within:border-accent"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          {local ? <FolderOpen size={20} className="ml-2 text-muted" /> : <GitBranch size={20} className="ml-2 text-muted" />}
          <input
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={local ? "Local folder, e.g. ../../cuda-archer" : "https://github.com/owner/repo"}
            className="flex-1 bg-transparent px-2 py-2.5 text-[16px] outline-none placeholder:text-muted/60"
          />
          <button type="submit" className="flex items-center gap-1.5 rounded-lg bg-accent px-5 py-2.5 font-semibold text-canvas hover:brightness-110">
            Analyze <ArrowRight size={17} />
          </button>
        </form>
        <div className="mt-3 flex items-center gap-4 text-[13px]">
          <button className="text-accent hover:underline" onClick={() => { setValue(DEMO_URL); setLocal(false); submit(DEMO_URL, false); }}>
            Try demo: CUDA Archer
          </button>
          <span className="text-line">|</span>
          <button className="text-muted hover:text-ink" onClick={() => { setLocal(!local); setValue(local ? "" : DEMO_LOCAL); }}>
            {local ? "Use a GitHub URL instead" : "Use a local folder instead"}
          </button>
        </div>
        {error && <div className="mt-4 max-w-2xl rounded-lg border border-bad/50 bg-bad/10 px-3 py-2 text-[13px] text-bad">{error}</div>}
        {recent.length > 0 && (
          <div className="mt-10 w-full max-w-2xl">
            <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">
              <History size={13} /> Recent analyses
            </div>
            <div className="flex flex-col gap-1">
              {recent.slice(0, 6).map((a) => (
                <button key={a.id} onClick={() => void openAnalysis(a.id)}
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px] hover:bg-panel">
                  <span className="font-medium">{a.name}</span>
                  <span className={`text-[11px] ${a.runnable ? "text-good" : "text-muted"}`}>{a.runnable ? "runnable" : "plan mode"}</span>
                  <span className="flex-1" />
                  <span className="text-[11px] text-muted">{new Date(a.created_at).toLocaleString()}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
