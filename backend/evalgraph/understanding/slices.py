"""Slices: compact, line-numbered views of the code that matters, under a token budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analyzer.facts import FileFacts
from .buffer import RepoBuffer
from .components import Component
from .signals import SymbolInfo

MAX_LINES = 70
HEAD_LINES = 18
SLICE_BUDGET_CHARS = 90_000      # ~22k tokens across all map calls
BATCH_CHARS = 14_000             # ~3.5k tokens per map call
MAX_DISCOVERY = 12


@dataclass
class Slice:
    id: str
    key: str
    path: str
    text: str
    lines: set[int] = field(default_factory=set)
    component_ids: list[str] = field(default_factory=list)   # empty = discovery slice

    @property
    def size(self) -> int:
        return len(self.text)


def _numbered(lines: list[str], start: int, keep: set[int]) -> tuple[str, set[int]]:
    out, shown, prev = [], set(), None
    for i, text in enumerate(lines, start):
        if i not in keep:
            continue
        if prev is not None and i != prev + 1:
            out.append("   ...")
        out.append(f"{i:4d}| {text}")
        shown.add(i)
        prev = i
    return "\n".join(out), shown


def make_slice(sid: str, info: SymbolInfo, buffer: RepoBuffer, facts: dict[str, FileFacts]) -> Slice:
    src = buffer.files.get(info.path, "").splitlines()
    start, end = info.d.line, min(info.d.end_line, len(src))
    body = src[start - 1:end]
    if len(body) <= MAX_LINES:
        keep = set(range(start, end + 1))
    else:  # signature + head + windows around signal lines
        keep = set(range(start, start + HEAD_LINES))
        for s in info.signals:
            keep |= set(range(max(start, s.line - 3), min(end, s.line + 3) + 1))
        keep = set(sorted(keep)[:MAX_LINES + 30])
    code, shown = _numbered(body, start, keep)
    ctx = []
    f = facts.get(info.path)
    if f is not None:
        used = set(info.d.names_used)
        for c in f.consts:
            if c.name in used:
                ctx.append(f"{c.line:4d}| {c.snippet}")
                shown.add(c.line)
        for p in f.prompts:
            if p.name in used:
                ctx.append(f"{p.line:4d}| {p.name} = {p.preview!r}")
                shown.add(p.line)
    header = f"### slice {sid} · {info.path} · {info.d.qualname}()"
    text = header + "\n" + code + ("\n# referenced constants/prompts:\n" + "\n".join(ctx) if ctx else "")
    return Slice(id=sid, key=info.key, path=info.path, text=text, lines=shown)


def select_slices(comps: list[Component], symbols: dict[str, SymbolInfo], scope: set[str], ranks: dict[str, float],
                  buffer: RepoBuffer, files: list[FileFacts]) -> list[Slice]:
    facts = {f.path: f for f in files}
    slices: list[Slice] = []
    by_key: dict[str, list[str]] = {}
    for c in comps:
        by_key.setdefault(c.key, []).append(c.id)
    used = 0
    for i, key in enumerate(sorted(by_key, key=lambda k: -ranks.get(k, 0))):
        s = make_slice(f"s{i}", symbols[key], buffer, facts)
        s.component_ids = by_key[key]
        if used + s.size > SLICE_BUDGET_CHARS:
            break
        slices.append(s)
        used += s.size
    # discovery: well-connected in-scope symbols without a facet (plain tools, custom pipelines, ...)
    taken = set(by_key)
    candidates = sorted((k for k in scope if k not in taken and symbols[k].d.name != "<module>"
                         and not symbols[k].d.name.startswith("__")), key=lambda k: -ranks.get(k, 0))
    for j, key in enumerate(candidates[:MAX_DISCOVERY]):
        s = make_slice(f"d{j}", symbols[key], buffer, facts)
        if used + s.size > SLICE_BUDGET_CHARS:
            break
        slices.append(s)
        used += s.size
    return slices


def batches(slices: list[Slice]) -> list[list[Slice]]:
    out, cur, size = [], [], 0
    for s in slices:
        if cur and size + s.size > BATCH_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += s.size
    if cur:
        out.append(cur)
    return out
