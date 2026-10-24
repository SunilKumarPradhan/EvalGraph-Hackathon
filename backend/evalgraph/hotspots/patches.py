"""Rule-based fix patches for literal-valued hotspots (conf_07 Q-31). No LLM; generic; evidence lines only.

A patch is a list of whole-line edits taken straight from a hotspot's evidence: the line as analyzed (`before`) and the
same line with one literal changed (`after`). The sandbox applies them after cloning and refuses if a line no longer
matches, so a fix is never guessed onto changed code. Only literal changes are proposed (numbers, booleans, None,
a missing keyword argument on a one-line call); anything that needs real code (e.g. history truncation) gets none.
"""

from __future__ import annotations

import re

from ..core.schemas import Hotspot, PatchEdit
from ..understanding.buffer import RepoBuffer

NUM = r"-?\d+(?:\.\d+)?"
# rule -> list of (key pattern, predicate on the old literal, new literal)
NUMERIC: dict[str, list[tuple[str, callable, str]]] = {
    "RAG_TOPK_NO_RERANK": [(r"(?i)^(k|top_?k|similarity_top_k|\w*TOP_K\w*|n_results|limit)$", lambda v: v >= 8, "4")],
    "RAG_BIG_CHUNKS": [(r"(?i)chunk_?size", lambda v: v >= 800, "500"), (r"(?i)chunk_?overlap", lambda v: v == 0, "50")],
    "LLM_HIGH_TEMP": [(r"(?i)temperature", lambda v: v > 0.7, "0.2")],
}
NONE_TO = {  # rule -> (key pattern, replacement for None / null)
    "EXT_NO_TIMEOUT": (r"(?i)timeout", "10"),
    "LLM_UNBOUNDED_CONTEXT": (r"(?i)(max_context|budget|max_chars)", "6000"),
    "MEMORY_UNBOUNDED": (r"(?i)(window|max_turns|max_history)", "6"),
}
# flag-constant rules: the flag's risky value is the evidence; the fix is the opposite boolean
FLIP = {"LLM_UNGROUNDED", "NO_GUARDRAIL", "ROUTER_NO_SCHEMA", "PARSER_NAIVE_JSON", "TOOL_BRITTLE_LOOKUP",
        "TOOL_UNVALIDATED", "EXTRA_LLM_HOP", "AGENT_UNGROUNDED", "RAG_TOPK_NO_RERANK"}
INSERT_KWARG = {"EXT_NO_TIMEOUT": "timeout=10", "LLM_NO_MAX_TOKENS": "max_tokens=800"}
STRING_SWAP = {"BIG_MODEL_FOR_CLASSIFY": {"large": "small"}}
# rule -> (key pattern, values worth sweeping) for single numeric literals
SWEEP = {
    "RAG_TOPK_NO_RERANK": (r"(?i)(^k$|top_?k|similarity_top_k|n_results|limit)", ["2", "4", "6", "8"]),
    "LLM_HIGH_TEMP": (r"(?i)temperature", ["0", "0.2", "0.5"]),
    "RAG_BIG_CHUNKS": (r"(?i)chunk_?size", ["300", "500", "800"]),
    "MEMORY_UNBOUNDED": (r"(?i)(window|max_turns|max_history)", ["2", "4", "8"]),
    "LLM_UNBOUNDED_CONTEXT": (r"(?i)(max_context|budget|max_chars)", ["3000", "6000", "12000"]),
}


def _sweep_meta(rule_id: str, before: str, after: str, summaries: list[str]) -> tuple[list[int] | None, list[str]]:
    """Span of the one changed literal and the sweep values, when the edit changed exactly one numeric literal."""
    if len(summaries) != 1 or rule_id not in SWEEP:
        return None, []
    key = summaries[0].split(" ", 1)[0]
    if not re.search(SWEEP[rule_id][0], key):
        return None, []
    p = 0
    while p < min(len(before), len(after)) and before[p] == after[p]:
        p += 1
    s = 0
    while s < min(len(before), len(after)) - p and before[-1 - s] == after[-1 - s]:
        s += 1
    start, end = p, len(after) - s
    while start > 0 and after[start - 1] in "0123456789.-":  # widen to the whole literal (0.9 → 0.2 differ in one char)
        start -= 1
    while end < len(after) and after[end] in "0123456789.":
        end += 1
    if not re.fullmatch(NUM, after[start:end] or "x"):  # the changed text must be the literal itself
        return None, []
    return [start, end], SWEEP[rule_id][1]


def with_value(edit: PatchEdit, value: str) -> PatchEdit:
    """The same edit with the swept literal set to `value`."""
    if not edit.span or not re.fullmatch(NUM, value):
        raise ValueError("this edit has no sweepable numeric literal")
    a, b = edit.span
    key = edit.summary.split(" ", 1)[0]
    m = re.match(NUM + r"|None|null", edit.before[a:])  # the literal as written in the code
    old = m.group(0) if m else "?"
    return edit.model_copy(update={"after": edit.after[:a] + value + edit.after[b:], "summary": f"{key} {old} → {value}",
                                   "span": [a, a + len(value)]})


def _replace_value(line: str, key_pat: str, new: str, accept) -> tuple[str, str] | None:
    """Change `KEY = literal`, `key=literal`, `key: literal` or `"key": literal` on this line, if accept(literal)."""
    rx = re.compile(r"""(?P<key>["']?[A-Za-z_][\w.]*["']?)(?P<sep>\s*(?::\s*[\w\[\]|. ]+?)?\s*=\s*|\s*:\s*)(?P<val>"""
                    + NUM + r"|None|null|True|False|true|false)\b")
    for m in rx.finditer(line):
        key = m.group("key").strip("\"'").split(".")[-1]
        if not re.search(key_pat, key):
            continue
        val = m.group("val")
        if not accept(val):
            continue
        new_line = line[: m.start("val")] + new + line[m.end("val"):]
        return new_line, f"{key} {val} → {new}"
    return None


def _flip_bool(line: str) -> tuple[str, str] | None:
    m = re.match(r"^(\s*)([A-Z][A-Z0-9_]*)(\s*(?::\s*[\w|\[\] ]+)?\s*=\s*)(True|False)\b", line)
    if not m:
        return None
    new = "False" if m.group(4) == "True" else "True"
    return line[: m.start(4)] + new + line[m.end(4):], f"{m.group(2)} {m.group(4)} → {new}"


def _insert_kwarg(line: str, kwarg: str, path: str) -> tuple[str, str] | None:
    """Add a keyword argument to the outermost call on a one-line Python call, if absent."""
    if not path.endswith(".py") or kwarg.split("=")[0] + "=" in line:
        return None
    start = line.find("(")
    if start < 0:
        return None
    depth, quote = 0, None
    for i in range(start, len(line)):
        ch = line[i]
        if quote:
            if ch == quote and line[i - 1] != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inner = line[start + 1:i].strip()
                sep = ", " if inner else ""
                return line[:i] + sep + kwarg + line[i:], f"add {kwarg}"
    return None  # call spans several lines: not a literal fix


JS_EXT = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")


def _call_span(line: str, open_idx: int) -> tuple[int, list[tuple[int, int]]] | None:
    """Index of the matching ")" for the "(" at open_idx, and the top-level argument spans; None if the call spans lines."""
    depth, quote, args, start = 0, None, [], open_idx + 1
    for i in range(open_idx, len(line)):
        ch = line[i]
        if quote:
            if ch == quote and line[i - 1] != "\\":
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                if line[start:i].strip():
                    args.append((start, i))
                return i, args
        elif ch == "," and depth == 1:
            args.append((start, i))
            start = i + 1
    return None


def _js_fetch_timeout(line: str, path: str) -> tuple[str, str] | None:
    """fetch(url) -> fetch(url, { signal: AbortSignal.timeout(10000) }); an options object gets the signal added."""
    if not path.endswith(JS_EXT) or "signal" in line:
        return None
    idx = line.find("fetch(")
    if idx < 0:
        return None
    span = _call_span(line, idx + len("fetch"))
    if span is None:
        return None
    close, args = span
    if len(args) == 1:
        return line[:close] + ", { signal: AbortSignal.timeout(10000) }" + line[close:], "add fetch timeout (10 s)"
    if len(args) == 2:
        a, b = args[1]
        opt = line[a:b]
        if opt.strip().startswith("{"):
            brace = a + opt.index("{")
            return line[: brace + 1] + " signal: AbortSignal.timeout(10000)," + line[brace + 1:], "add fetch timeout (10 s)"
    return None


def _swap_string(line: str, swaps: dict[str, str]) -> tuple[str, str] | None:
    for old, new in swaps.items():
        m = re.search(rf"""(["']){old}\1""", line)
        if m:
            return line[: m.start()] + f"{m.group(1)}{new}{m.group(1)}" + line[m.end():], f'"{old}" → "{new}"'
    return None


def edit_for(rule_id: str, line: str, path: str) -> tuple[str, str] | None:
    for key_pat, pred, new in NUMERIC.get(rule_id, []):
        if hit := _replace_value(line, key_pat, new, lambda v, p=pred: bool(re.fullmatch(NUM, v)) and p(float(v))):
            return hit
    if rule_id in NONE_TO:
        key_pat, new = NONE_TO[rule_id]
        if hit := _replace_value(line, key_pat, new, lambda v: v in ("None", "null")):
            return hit
    if rule_id in FLIP and (hit := _flip_bool(line)):
        return hit
    if rule_id in INSERT_KWARG and (hit := _insert_kwarg(line, INSERT_KWARG[rule_id], path)):
        return hit
    if rule_id == "EXT_NO_TIMEOUT" and (hit := _js_fetch_timeout(line, path)):
        return hit
    if rule_id in STRING_SWAP and (hit := _swap_string(line, STRING_SWAP[rule_id])):
        return hit
    return None


def patch_for(hotspot: Hotspot, buffer: RepoBuffer) -> list[PatchEdit]:
    edits: list[PatchEdit] = []
    seen: set[tuple[str, int]] = set()
    for ev in hotspot.evidence:
        if ev.line is None or ev.file not in buffer.files or ev.file.endswith(".ipynb") or (ev.file, ev.line) in seen:
            continue
        lines = buffer.files[ev.file].splitlines()
        if not 0 < ev.line <= len(lines):
            continue
        before = after = lines[ev.line - 1]
        summaries: list[str] = []
        for _ in range(3):  # a line can carry several risky literals (chunk_size=1500, chunk_overlap=0)
            hit = edit_for(hotspot.rule_id, after, ev.file)
            if not hit or hit[0] == after:
                break
            after, summaries = hit[0], [*summaries, hit[1]]
        if after != before:
            seen.add((ev.file, ev.line))
            span, sweep = _sweep_meta(hotspot.rule_id, before, after, summaries)
            edits.append(PatchEdit(file=ev.file, line=ev.line, before=before, after=after, summary="; ".join(summaries),
                                   span=span, sweep=sweep))
    return edits


def attach_patches(hotspots: list[Hotspot], buffer: RepoBuffer) -> int:
    """Fill Hotspot.patch where a literal fix exists; returns how many hotspots got one."""
    n = 0
    for h in hotspots:
        h.patch = patch_for(h, buffer)
        n += bool(h.patch)
    return n


def unified_diff(edits: list[PatchEdit]) -> str:
    """A zero-context unified diff (apply with `git apply --unidiff-zero`)."""
    out: list[str] = []
    for file in dict.fromkeys(e.file for e in edits):
        out += [f"--- a/{file}", f"+++ b/{file}"]
        for e in sorted((e for e in edits if e.file == file), key=lambda e: e.line):
            out += [f"@@ -{e.line},1 +{e.line},1 @@", f"-{e.before}", f"+{e.after}"]
    return "\n".join(out) + "\n"
