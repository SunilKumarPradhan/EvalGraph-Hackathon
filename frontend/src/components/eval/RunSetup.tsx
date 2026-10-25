import { KeyRound, Plus, Save, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api } from "../../api/client";
import { useAppStore } from "../../store/useAppStore";
import { toast } from "../../store/useToast";
import type { AnalysisResult, GoldenSample } from "../../types/contracts";

type Plan = Record<string, unknown>;
const field = "w-full rounded-md border border-line bg-canvas px-2 py-1 text-[12px] outline-none focus:border-accent";

function Text({ label, value, onChange, mono = true }: { label: string; value: string; onChange: (v: string) => void; mono?: boolean }) {
  return (
    <label className="flex flex-col gap-0.5 text-[11px] text-muted">
      {label}
      <input value={value} onChange={(e) => onChange(e.target.value)} className={`${field} ${mono ? "font-mono" : ""}`} />
    </label>
  );
}

/** How the Docker sandbox calls the app, and with which questions. Inferred by the analysis; editable here. */
export function RunSetup({ analysis, onClose }: { analysis: AnalysisResult; onClose: () => void }) {
  const setAnalysis = useAppStore((s) => s.setAnalysis);
  const runtime = analysis.runtime;
  const [plan, setPlan] = useState<Plan>({ ...(runtime?.plan ?? { mode: "python_function" }) });
  const [args, setArgs] = useState(JSON.stringify(plan.fixed_args ?? {}, null, 1));
  const [questions, setQuestions] = useState<GoldenSample[]>(runtime?.dataset ?? []);
  const [saving, setSaving] = useState(false);
  const [secretNames, setSecretNames] = useState<{ stored: string[]; wanted: string[] }>({ stored: [], wanted: [] });
  const [secretValues, setSecretValues] = useState<Record<string, string>>({});
  const [newSecret, setNewSecret] = useState("");
  useEffect(() => {
    api.secrets(analysis.id).then(setSecretNames).catch(() => undefined);
  }, [analysis.id]);
  const mode = String(plan.mode ?? "python_function");
  const set = (k: string, v: unknown) => setPlan((p) => ({ ...p, [k]: v }));
  const str = (k: string) => (plan[k] === null || plan[k] === undefined ? "" : String(plan[k]));

  const save = async () => {
    let fixed: unknown;
    try {
      fixed = JSON.parse(args || "{}");
    } catch {
      toast("Fixed arguments must be a JSON object, e.g. {\"pdf_path\": \"docs/manual.pdf\"}", "error");
      return;
    }
    const keys = mode === "http" ? ["mode", "start_cmd", "path", "method", "port"] : ["mode", "module", "function", "question_param"];
    const edit: Plan = Object.fromEntries(keys.map((k) => [k, k === "port" ? Number(plan[k] ?? 8765) : plan[k] ?? null]));
    if (mode !== "http") edit.fixed_args = fixed;
    setSaving(true);
    try {
      setAnalysis(await api.updateRuntime(analysis.id, { plan: edit, dataset: questions }));
      const changed = Object.fromEntries(Object.entries(secretValues).filter(([, v]) => v !== ""));
      if (Object.keys(changed).length) setSecretNames(await api.saveSecrets(analysis.id, changed));
      toast("Run setup saved. The next run uses it.", "success");
      onClose();
    } catch (e) {
      toast(`Not saved: ${(e as ApiError).message}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const nextId = () => {
    let i = questions.length + 1;
    while (questions.some((q) => q.id === `u${i}`)) i += 1;
    return `u${i}`;
  };

  return (
    <aside className="eg-slide flex w-[400px] shrink-0 flex-col overflow-hidden border-l border-line bg-panel">
      <div className="flex items-start gap-2 border-b border-line p-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 text-[15px] font-semibold">
            Run setup {Boolean(runtime?.plan?.edited) && <span className="rounded-full border border-accent px-1.5 text-[10px] text-accent">edited</span>}
          </div>
          <div className="mt-0.5 text-[12px] text-muted">How the Docker sandbox calls this app. Inferred from the code; fix it if it guessed wrong.</div>
        </div>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close run setup"><X size={18} /></button>
      </div>
      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
        <section className="flex flex-col gap-2">
          <h3 className="text-[11px] font-semibold tracking-wider text-muted uppercase">Entry point</h3>
          <select value={mode} onChange={(e) => set("mode", e.target.value)} className={field}>
            <option value="python_function">Call a Python function in-process</option>
            <option value="http">Start a web server and POST to a route</option>
          </select>
          {mode === "http" ? (
            <>
              <Text label="Start command" value={str("start_cmd")} onChange={(v) => set("start_cmd", v)} />
              <div className="grid grid-cols-3 gap-2">
                <Text label="Method" value={str("method") || "POST"} onChange={(v) => set("method", v.toUpperCase())} />
                <Text label="Path" value={str("path")} onChange={(v) => set("path", v)} />
                <Text label="Port" value={str("port") || "8765"} onChange={(v) => set("port", v)} />
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                <Text label="Module" value={str("module")} onChange={(v) => set("module", v)} />
                <Text label="Function" value={str("function")} onChange={(v) => set("function", v)} />
              </div>
              <Text label="Question parameter" value={str("question_param")} onChange={(v) => set("question_param", v)} />
              <label className="flex flex-col gap-0.5 text-[11px] text-muted">
                Fixed arguments (JSON, passed on every call)
                <textarea value={args} onChange={(e) => setArgs(e.target.value)} rows={3} className={`${field} font-mono`} />
              </label>
            </>
          )}
          {runtime?.plan?.reason != null && <p className="text-[11px] text-muted">Analysis said: {String(runtime.plan.reason)}</p>}
        </section>
        <section className="flex flex-col gap-2">
          <h3 className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-muted uppercase">
            <KeyRound size={12} /> App secrets
          </h3>
          <p className="text-[11px] text-muted">
            Keys the app itself needs (search APIs, vector DB clouds). Stored on the EvalGraph server, never shown again,
            passed only into the sandbox. The LLM key is handled separately and never reaches the app.
          </p>
          {[...new Set([...secretNames.wanted, ...secretNames.stored, ...Object.keys(secretValues)])].map((name) => (
            <label key={name} className="flex items-center gap-2 text-[11px]">
              <span className="w-40 truncate font-mono" title={name}>{name}</span>
              <input type="password" placeholder={secretNames.stored.includes(name) ? "•••• stored" : "not set"}
                     value={secretValues[name] ?? ""} onChange={(e) => setSecretValues({ ...secretValues, [name]: e.target.value })}
                     className={field} autoComplete="off" />
              {secretNames.stored.includes(name) && (
                <button onClick={async () => setSecretNames(await api.saveSecrets(analysis.id, { [name]: null }))}
                        className="text-muted hover:text-bad" aria-label={`Delete ${name}`}><Trash2 size={12} /></button>
              )}
            </label>
          ))}
          <div className="flex gap-2">
            <input value={newSecret} onChange={(e) => setNewSecret(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                   placeholder="OTHER_VARIABLE" className={`${field} font-mono`} />
            <button onClick={() => { if (newSecret) { setSecretValues({ ...secretValues, [newSecret]: "" }); setNewSecret(""); } }}
                    className="text-[11px] whitespace-nowrap text-accent hover:underline">add variable</button>
          </div>
        </section>
        <section className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] font-semibold tracking-wider text-muted uppercase">Questions ({questions.length})</h3>
            <button onClick={() => setQuestions([...questions, { id: nextId(), category: "domain", question: "" }])}
                    className="flex items-center gap-1 text-[11px] text-accent hover:underline"><Plus size={12} /> add</button>
          </div>
          {questions.map((q, i) => (
            <div key={q.id} className="rounded-md border border-line bg-panel-2 p-2">
              <div className="mb-1 flex items-center gap-2 text-[10.5px] text-muted">
                <span className="font-mono">{q.id}</span>
                <input value={q.category} onChange={(e) => setQuestions(questions.map((x, j) => (j === i ? { ...x, category: e.target.value } : x)))}
                       className="w-24 rounded border border-line bg-canvas px-1 outline-none" title="Category" />
                <label className="flex items-center gap-1" title="The app should refuse this one (safety probe)">
                  <input type="checkbox" checked={q.expected_behavior === "refuse"} className="accent-teal-400"
                         onChange={(e) => setQuestions(questions.map((x, j) => (j === i ? { ...x, expected_behavior: e.target.checked ? "refuse" : "answer" } : x)))} />
                  must refuse
                </label>
                <button onClick={() => setQuestions(questions.filter((_, j) => j !== i))} className="ml-auto hover:text-bad" aria-label={`Remove ${q.id}`}>
                  <Trash2 size={12} />
                </button>
              </div>
              <textarea value={q.question} rows={2} className={field}
                        onChange={(e) => setQuestions(questions.map((x, j) => (j === i ? { ...x, question: e.target.value } : x)))} />
            </div>
          ))}
        </section>
      </div>
      <div className="border-t border-line p-3">
        <button onClick={() => void save()} disabled={saving}
                className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-accent py-2 text-[13px] font-bold text-canvas hover:brightness-110 disabled:opacity-40">
          <Save size={14} /> Save run setup
        </button>
      </div>
    </aside>
  );
}
