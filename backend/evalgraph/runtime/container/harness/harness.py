"""In-sandbox driver (stdlib only). Runs as the unprivileged `app` user inside the container.

    python harness.py /out/job.json

job.json: {"source": {"repo_url": ...} | {"path": "/src"}, "plan": RunPlan, "samples": [{"id", "question"}],
           "gateway": "http://127.0.0.1:8919", "timeouts": {...}}
Writes /out/results.jsonl ({sample_id, start, end, answer, error}), /out/spans.jsonl (probe), /out/harness.log.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def log(out: str, msg: str) -> None:
    with open(os.path.join(out, "harness.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def sh(cmd: list[str] | str, cwd: str, env: dict, timeout: int, out: str, label: str) -> int:
    log(out, f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    with open(os.path.join(out, f"{label}.log"), "a", encoding="utf-8") as f:
        try:
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                  shell=isinstance(cmd, str)).returncode
        except subprocess.TimeoutExpired:
            log(out, f"timeout after {timeout}s: {label}")
            return 124


def acquire(source: dict, work: str, out: str, env: dict) -> str:
    repo = os.path.join(work, "repo")
    if source.get("repo_url"):
        branch = ["--branch", source["ref"]] if source.get("ref") else []
        code = sh(["git", "clone", "--depth", "1", "--single-branch", *branch, source["repo_url"], repo], work, env, 300, out,
                  "clone")
        if code != 0:
            raise RuntimeError("git clone failed (see clone.log)")
        if source.get("subdir"):  # a monorepo folder: run that folder as the app
            sub = os.path.normpath(os.path.join(repo, source["subdir"]))
            if not sub.startswith(os.path.normpath(repo) + os.sep) or not os.path.isdir(sub):
                raise RuntimeError(f"folder {source['subdir']} not found in the repository")
            return sub
    else:
        shutil.copytree(source["path"], repo, ignore=shutil.ignore_patterns(".git", "node_modules", ".venv", "__pycache__"))
    return repo


def apply_patches(patches: list[dict], repo: str, out: str) -> None:
    """Whole-line edits from EvalGraph's rule-based fixes. Each line must still read exactly as analyzed."""
    for p in patches:
        path = os.path.normpath(os.path.join(repo, p["file"]))
        if not path.startswith(os.path.normpath(repo) + os.sep):
            raise RuntimeError(f"patch path escapes the repo: {p['file']}")
        with open(path, encoding="utf-8", newline="") as f:
            lines = f.read().split("\n")
        i = p["line"] - 1
        current = lines[i].rstrip("\r") if 0 <= i < len(lines) else None
        if current != p["before"].rstrip("\r"):
            raise RuntimeError(f"patch does not apply: {p['file']}:{p['line']} changed since the analysis")
        lines[i] = p["after"] + ("\r" if lines[i].endswith("\r") else "")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines))
        log(out, f"patched {p['file']}:{p['line']}: {p.get('summary', '')}")


def install(plan: dict, repo: str, env: dict, out: str, timeout: int) -> None:
    cmds = plan.get("install")
    if cmds == "none":
        return
    if not cmds or cmds == "auto":
        cmds = []
        if os.path.exists(os.path.join(repo, "requirements.txt")):
            cmds.append([sys.executable, "-m", "pip", "install", "--user", "-q", "-r", "requirements.txt"])
        elif os.path.exists(os.path.join(repo, "pyproject.toml")) or os.path.exists(os.path.join(repo, "setup.py")):
            cmds.append([sys.executable, "-m", "pip", "install", "--user", "-q", "."])
        elif plan.get("inferred_requirements"):  # no manifest: packages inferred from the code's imports
            cmds.append([sys.executable, "-m", "pip", "install", "--user", "-q", *plan["inferred_requirements"]])
        if plan.get("mode") == "http" and os.path.exists(os.path.join(repo, "package.json")):
            cmds.append(["npm", "install", "--no-audit", "--no-fund", "--loglevel=error"])
    for cmd in cmds:
        code = sh(cmd, repo, env, timeout, out, "install")
        if code != 0:
            log(out, f"install step failed with code {code}; continuing (the run may still work)")


def _app_secrets() -> dict:
    """User-provided secrets for the app (search keys, DB URLs), from the entrypoint's app-only file."""
    path = os.environ.get("EVALGRAPH_APP_SECRETS", "/run/evalgraph-app/secrets.json")
    try:
        with open(path, encoding="utf-8") as f:
            return {k: str(v) for k, v in json.load(f).items() if k and not k.startswith(("DIAL_", "EVALGRAPH"))}
    except (OSError, ValueError):
        return {}


def app_env(base_env: dict, gateway: str, out: str, repo: str, llm_env: dict | None = None) -> dict:
    env = dict(base_env)
    fake_key = "sk-evalgraph-sandbox-0000000000000000000000000000"
    # the app's own variable names for its LLM settings (found by the analysis), pointed at the gateway
    by_role = {"key": fake_key, "base": f"{gateway}/v1", "endpoint": gateway, "version": "2024-02-01"}
    env.update({name: by_role[role] for name, role in (llm_env or {}).items() if role in by_role})
    env.update(_app_secrets())
    env.update({
        # every common SDK points at the in-sandbox gateway with a dummy key
        "OPENAI_API_KEY": fake_key, "OPENAI_BASE_URL": f"{gateway}/v1", "OPENAI_API_BASE": f"{gateway}/v1",
        "AZURE_OPENAI_API_KEY": fake_key, "AZURE_OPENAI_ENDPOINT": gateway, "OPENAI_API_VERSION": "2024-02-01",
        "ANTHROPIC_API_KEY": fake_key, "ANTHROPIC_BASE_URL": gateway,
        "PYTHONPATH": os.pathsep.join([os.path.join(HERE, "site"), repo, os.path.join(repo, "src")]),
        "EVALGRAPH_PROBES": os.path.join(out, "probes.json"), "EVALGRAPH_SPANS": os.path.join(out, "spans.jsonl"),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def run_isolated(job: dict, repo: str, env: dict, out: str) -> None:
    """One fresh process per sample: nothing (conversation memory, caches) carries over between samples."""
    runner = os.path.join(HERE, "runner_fn.py")
    timeout = job.get("timeouts", {}).get("sample", 180) + 120
    for i, sample in enumerate(job["samples"]):
        path = os.path.join(out, f"job_{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**job, "samples": [sample]}, f)
        start = time.time()
        code = sh([sys.executable, runner, path, os.path.join(out, "results.jsonl")], repo, env, timeout, out, "app")
        if code != 0:  # a crash on one sample is that sample's result, not the run's
            with open(os.path.join(out, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": sample["id"], "start": start, "end": time.time(), "answer": "",
                                    "error": f"app process exited with {code} (see app.log)"}) + "\n")
    log(out, f"isolated runner: {len(job['samples'])} process(es)")


def run_function_mode(job: dict, repo: str, env: dict, out: str) -> None:
    if job.get("isolated"):
        return run_isolated(job, repo, env, out)
    runner = os.path.join(HERE, "runner_fn.py")
    timeout = job.get("timeouts", {}).get("run", 1800)
    code = sh([sys.executable, runner, os.path.join(out, "job.json"), os.path.join(out, "results.jsonl")],
              repo, env, timeout, out, "app")
    log(out, f"function runner exited with {code}")
    if code != 0:
        with open(os.path.join(out, "app.log"), encoding="utf-8", errors="replace") as f:
            tail = f.read()[-600:]
        raise RuntimeError(f"the app crashed (exit {code}): {tail.strip().splitlines()[-1] if tail.strip() else ''}")


def wait_port(port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def _extract(payload, path: str | None) -> str:
    if path:
        for part in path.split("."):
            payload = payload[int(part)] if isinstance(payload, list) else (payload or {}).get(part)
        return payload if isinstance(payload, str) else json.dumps(payload)
    if isinstance(payload, dict):
        for key in ("answer", "response", "output", "text", "content", "message", "result"):
            if isinstance(payload.get(key), str):
                return payload[key]
    return payload if isinstance(payload, str) else json.dumps(payload)


def run_http_mode(job: dict, repo: str, env: dict, out: str) -> None:
    plan = job["plan"]
    port = int(plan.get("port", 8765))
    env = {**env, "PORT": str(port)}
    with open(os.path.join(out, "server.log"), "a", encoding="utf-8") as f:
        server = subprocess.Popen(plan["start_cmd"], cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  shell=isinstance(plan["start_cmd"], str))
    try:
        if not wait_port(port, job.get("timeouts", {}).get("start", 180)):
            raise RuntimeError("app server did not open its port (see server.log)")
        for s in job["samples"]:
            body = json.loads(json.dumps(plan.get("body", {"question": "{question}"})).replace("{question}", json.dumps(s["question"])[1:-1]))
            req = urllib.request.Request(f"http://127.0.0.1:{port}{plan.get('path', '/')}", data=json.dumps(body).encode(),
                                         method=plan.get("method", "POST"), headers={"Content-Type": "application/json"})
            start, answer, error = time.time(), "", None
            try:
                with urllib.request.urlopen(req, timeout=job.get("timeouts", {}).get("sample", 180)) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                try:
                    answer = _extract(json.loads(raw), plan.get("response_path"))
                except json.JSONDecodeError:
                    answer = raw
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            with open(os.path.join(out, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": s["id"], "start": start, "end": time.time(), "answer": answer[:4000],
                                    "error": error}) + "\n")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> None:
    job_path = sys.argv[1]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    out = os.path.dirname(os.path.abspath(job_path))
    work = job.get("work", "/work")
    os.makedirs(work, exist_ok=True)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("DIAL_", "EVALGRAPH_SECRET"))}
    base_env["PATH"] = os.pathsep.join([os.path.expanduser("~/.local/bin"), base_env.get("PATH", "")])
    with open(os.path.join(out, "probes.json"), "w", encoding="utf-8") as f:
        json.dump(job.get("probes", []), f)
    status = {"ok": False, "stage": "start"}
    try:
        status["stage"] = "acquire"
        repo = acquire(job["source"], work, out, base_env)
        if job.get("patches"):
            status["stage"] = "patch"
            apply_patches(job["patches"], repo, out)
        status["stage"] = "install"
        install(job["plan"], repo, base_env, out, job.get("timeouts", {}).get("install", 900))
        env = app_env(base_env, job["gateway"], out, repo, job["plan"].get("llm_env"))
        status["stage"] = "run"
        if job["plan"]["mode"] == "python_function":
            run_function_mode(job, repo, env, out)
        elif job["plan"]["mode"] == "http":
            run_http_mode(job, repo, env, out)
        else:
            raise RuntimeError(f"unsupported run mode {job['plan']['mode']}")
        done = 0
        if os.path.exists(os.path.join(out, "results.jsonl")):
            with open(os.path.join(out, "results.jsonl"), encoding="utf-8") as f:
                done = sum(1 for _ in f)
        if done < len(job["samples"]):
            raise RuntimeError(f"only {done} of {len(job['samples'])} samples produced a result")
        status["ok"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        log(out, status["error"])
    with open(os.path.join(out, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f)


if __name__ == "__main__":
    main()
