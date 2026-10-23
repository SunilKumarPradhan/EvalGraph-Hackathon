"""Per-file fact extraction with the stdlib `ast` module."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from ..core.schemas import ConfigValue
from .facts import (
    AssignFact,
    CallFact,
    ConstFact,
    DefFact,
    FileFacts,
    ImportFact,
    LangGraphFact,
    PromptFact,
    RegistryFact,
)

INTERESTING_DEFAULTS = {"top_k", "k", "chunk_size", "chunk_overlap", "temperature", "max_tokens", "model", "timeout"}
PROMPT_MARKERS = re.compile(r"\byou are\b|return (only )?(raw )?json|respond in json|classify|rewrite|answer (only|the)"
                            r"|rules:|instructions", re.I)
INTERNAL_NOTES = re.compile(r"internal (debug )?note|build tag|secret|api[_ ]?key|password|token|confidential", re.I)
UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
MAX_SNIPPET = 200
MUTATORS = {"append", "extend", "insert", "add", "update", "appendleft", "extendleft", "setdefault", "put", "push"}
DOTTED = re.compile(r"[A-Za-z_][\w.]*")
RECV = "<recv>"     # CallFact.kwarg_names key for the receiver of a method call (`TEMPLATE.format(...)`)
RETURN = "<return>"   # AssignFact.target of `return x` / `yield x`


def dotted(node: ast.AST) -> str:
    """Best-effort dotted name for a callee: a.b.c, get_client().chat, x[...].y"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{dotted(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{dotted(node.value)}[]"
    return "?"


def literal(node: ast.AST | None) -> tuple[bool, ConfigValue]:
    if node is None:
        return False, None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None
    if value is None or isinstance(value, (bool, int, float, str)):
        return True, value
    return False, None


def string_value(node: ast.AST) -> str | None:
    """Text of a str constant, implicit concatenation, or f-string (placeholders kept as {expr})."""
    ok, value = literal(node)
    if ok and isinstance(value, str):
        return value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append("{" + ast.unparse(v.value) + "}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = string_value(node.left), string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _target_names(node: ast.AST) -> list[str]:
    """Variables written by an assignment target: `x`, `self.x`, `a.b.c`, the container of `x[k]`, tuple items."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in _target_names(e)]
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, ast.Subscript):
        return _target_names(node.value)
    if isinstance(node, ast.Attribute):
        name = re.sub(r"^cls\.", "self.", dotted(node))
        return [name] if DOTTED.fullmatch(name) else []
    return []


class _Extractor:
    def __init__(self, rel: str, source: str):
        self.rel = rel
        self.lines = source.splitlines()
        self.facts = FileFacts(path=rel)
        self.parents: dict[ast.AST, ast.AST] = {}
        self.module_assigns: list[AssignFact] = []

    def snippet(self, line: int) -> str:
        return self.lines[line - 1].strip()[:MAX_SNIPPET] if 0 < line <= len(self.lines) else ""

    # ---------------------------------------------------------- calls

    def _context(self, node: ast.AST) -> tuple[bool, bool]:
        in_loop = in_try = False
        child, parent = node, self.parents.get(node)
        while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if isinstance(parent, (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                in_loop = True
            if isinstance(parent, ast.Try) and child in parent.body:
                in_try = True
            child, parent = parent, self.parents.get(parent)
        return in_loop, in_try

    # ---------------------------------------------------------- data flow

    @staticmethod
    def refs(node: ast.AST | None, index: dict[int, int]) -> tuple[list[str], list[int]]:
        """Names read by an expression and the calls whose results it uses (nested calls are not entered:
        what flows through a call is decided by the call)."""
        names: set[str] = set()
        calls: set[int] = set()
        stack = [node] if node is not None else []
        while stack:
            n = stack.pop()
            if isinstance(n, ast.Call) and id(n) in index:
                calls.add(index[id(n)])
                continue
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):  # state["chunks"]: one key of a dict
                ok, key = literal(n.slice)
                if ok and isinstance(key, str):
                    names.add(f"{n.value.id}[{key}]")
                    continue
            if isinstance(n, ast.Attribute):  # `msg.tool_calls`, `self.history`: readers look up prefixes too
                name = re.sub(r"^cls\.", "self.", dotted(n))
                if DOTTED.fullmatch(name):
                    names.add(name)
                    continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            stack.extend(ast.iter_child_nodes(n))
        return sorted(names), sorted(calls)

    def assigns(self, body: list[ast.stmt], index: dict[int, int]) -> list[AssignFact]:
        out: list[AssignFact] = []

        def add(targets: list[str], value: ast.AST | None, line: int, extra: list[str] = ()) -> None:
            names, calls = self.refs(value, index)
            names = sorted(set(names) | set(extra))
            if not names and not calls:
                return
            for t in targets:
                out.append(AssignFact(target=t, line=line, sources=names, calls=calls))

        def walk(node: ast.AST, depth: int) -> None:
            if isinstance(node, ast.Assign):
                add([n for t in node.targets for n in _target_names(t)], node.value, node.lineno)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                add(_target_names(node.target), node.value, node.lineno)
            elif isinstance(node, ast.AugAssign):
                tgts = _target_names(node.target)
                add(tgts, node.value, node.lineno, extra=tgts)
            elif isinstance(node, ast.NamedExpr):
                add(_target_names(node.target), node.value, node.lineno)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                add(_target_names(node.target), node.iter, node.lineno)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                add(_target_names(node.optional_vars), node.context_expr, getattr(node.context_expr, "lineno", 0))
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and depth == 0 and node.value is not None:
                add([RETURN], node.value, node.lineno)
                if isinstance(node.value, ast.Dict):  # return {"chunks": ...}: state updates, key by key
                    for k, v in zip(node.value.keys, node.value.values):
                        ok, key = literal(k)
                        if ok and isinstance(key, str):
                            add([f"{RETURN}[{key}]"], v, node.lineno)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
                tgts = _target_names(node.func.value)
                for a in [*node.args, *(k.value for k in node.keywords)]:
                    add(tgts, a, node.lineno)
            nested = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            for child in ast.iter_child_nodes(node):
                walk(child, depth + 1 if nested else depth)

        for stmt in body:
            walk(stmt, 0)
        return out

    def call_fact(self, call: ast.Call, index: dict[int, int] | None = None) -> CallFact:
        index = index or {}
        kwargs: dict[str, ConfigValue] = {}
        kwarg_names: dict[str, list[str]] = {}
        kwarg_calls: dict[str, list[int]] = {}
        for kw in call.keywords:
            names, calls = self.refs(kw.value, index)
            key = kw.arg or "**"
            if names:
                kwarg_names[key] = names
            if calls:
                kwarg_calls[key] = calls
            if kw.arg is None:
                continue
            ok, value = literal(kw.value)
            kwargs[kw.arg] = value if ok else ast.unparse(kw.value)[:120]
        if isinstance(call.func, ast.Attribute):
            names, calls = self.refs(call.func.value, index)
            if names:
                kwarg_names[RECV] = names
            if calls:
                kwarg_calls[RECV] = calls
        arg_refs = [self.refs(a, index) for a in call.args]
        str_args, num_args = [], []
        for arg in call.args:
            ok, value = literal(arg)
            if ok and isinstance(value, str):
                str_args.append(value)
            elif isinstance(arg, ast.JoinedStr):  # f"{base_url}/chat/completions" keeps its literal parts
                text = string_value(arg)
                if text:
                    str_args.append(text)
            elif ok and isinstance(value, (int, float)) and not isinstance(value, bool):
                num_args.append(float(value))
        in_loop, in_try = self._context(call)
        parent = self.parents.get(call)
        target = None
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and getattr(parent, "value", None) is call:
            tgts = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            if tgts:
                target = dotted(tgts[0]).replace("self.", "")
        return CallFact(callee=dotted(call.func), line=call.lineno, kwargs=kwargs, str_args=str_args,
                        snippet=self.snippet(call.lineno), in_loop=in_loop, in_try=in_try, target=target,
                        num_args=num_args, arg_names=[r[0] for r in arg_refs], arg_calls=[r[1] for r in arg_refs],
                        kwarg_names=kwarg_names, kwarg_calls=kwarg_calls)

    def langgraph(self, call: ast.Call) -> None:
        name = dotted(call.func).rsplit(".", 1)[-1]
        args = call.args
        if name == "add_node" and args:
            ok, node = literal(args[0])
            fn = dotted(args[1]) if len(args) > 1 else None
            if len(args) > 1 and isinstance(args[1], ast.Call):  # ToolNode([search, calc]) -> the tools themselves
                listed = [dotted(e) for a in args[1].args if isinstance(a, (ast.List, ast.Tuple)) for e in a.elts]
                if listed:
                    fn = ",".join(listed)
            if ok and isinstance(node, str):
                self.facts.langgraph.append(LangGraphFact(kind="add_node", line=call.lineno, node=node, fn=fn,
                                                          snippet=self.snippet(call.lineno)))
        elif name == "add_edge" and len(args) >= 2:
            src, dst = self._node_ref(args[0]), self._node_ref(args[1])
            if src and dst:
                self.facts.langgraph.append(LangGraphFact(kind="add_edge", line=call.lineno, source=src, target=dst,
                                                          snippet=self.snippet(call.lineno)))
        elif name == "add_conditional_edges" and args:
            src = self._node_ref(args[0])
            mapping: dict[str, str] = {}
            path_map = args[2] if len(args) > 2 else next((k.value for k in call.keywords if k.arg == "path_map"), None)
            if isinstance(path_map, ast.Dict):
                for key, value in zip(path_map.keys, path_map.values):
                    if key is None and isinstance(value, ast.DictComp):
                        # **{r: "node" for r in REGISTRY}
                        ok, target = literal(value.value)
                        gen = value.generators[0] if value.generators else None
                        if ok and isinstance(target, str) and gen is not None and isinstance(gen.iter, ast.Name):
                            mapping[f"@registry:{gen.iter.id}"] = target
                        continue
                    ok_k, k = literal(key)
                    ok_v, v = literal(value)
                    if ok_k and ok_v and isinstance(v, str):
                        mapping[str(k)] = v
            if not mapping and len(args) > 1:
                router_fn = dotted(args[1])
                mapping = {"tools": "tools"} if router_fn.endswith("tools_condition") else {"@any": "@any"}
            if src:
                self.facts.langgraph.append(LangGraphFact(kind="add_conditional_edges", line=call.lineno, source=src,
                                                          mapping=mapping, snippet=self.snippet(call.lineno)))

    @staticmethod
    def _node_ref(node: ast.AST) -> str | None:
        ok, value = literal(node)
        if ok and isinstance(value, str):
            return value
        if isinstance(node, ast.Name) and node.id in {"START", "END"}:
            return f"__{node.id.lower()}__"
        return None

    def loop_add_nodes(self, loop: ast.For) -> None:
        """for name, fn in [("guard", guard), ...]: g.add_node(name, fn)"""
        uses_add_node = any(isinstance(n, ast.Call) and dotted(n.func).endswith("add_node") for n in ast.walk(loop))
        if not uses_add_node or not isinstance(loop.iter, (ast.List, ast.Tuple)):
            return
        for elt in loop.iter.elts:
            if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                ok, node = literal(elt.elts[0])
                if ok and isinstance(node, str):
                    self.facts.langgraph.append(LangGraphFact(kind="add_node", line=elt.lineno, node=node,
                                                              fn=dotted(elt.elts[1]), snippet=self.snippet(elt.lineno)))

    # ---------------------------------------------------------- defs

    def def_fact(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, qual: str) -> DefFact:
        decorators, dec_args = [], []
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            decorators.append(dotted(target))
            if isinstance(d, ast.Call):
                for a in d.args:
                    ok, v = literal(a)
                    if ok and isinstance(v, str):
                        dec_args.append(v)
        call_nodes: list[ast.Call] = []
        names: set[str] = set()
        subscripted: set[str] = set()
        body_nodes = node.body
        for stmt in body_nodes:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    call_nodes.append(sub)
                    self.langgraph(sub)
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    names.add(sub.id)
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                    subscripted.add(sub.value.id)
                elif isinstance(sub, ast.For):
                    self.loop_add_nodes(sub)
        index = {id(c): i for i, c in enumerate(call_nodes)}
        calls = [self.call_fact(c, index) for c in call_nodes]
        assigns = self.assigns(body_nodes, index) if not isinstance(node, ast.ClassDef) else []
        args, defaults = [], {}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            args = [a.arg for a in all_args]
            positional = node.args.posonlyargs + node.args.args
            for a, d in zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults):
                ok, v = literal(d)
                if ok and a.arg in INTERESTING_DEFAULTS:
                    defaults[a.arg] = v
            for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
                ok, v = literal(d)
                if ok and a.arg in INTERESTING_DEFAULTS:
                    defaults[a.arg] = v
        return DefFact(
            name=node.name, qualname=qual, kind="class" if isinstance(node, ast.ClassDef) else "function",
            line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
            decorators=decorators, decorator_str_args=dec_args, doc=(ast.get_docstring(node) or "").split("\n")[0][:200],
            args=args, defaults=defaults, calls=calls, names_used=sorted(names), subscripted=sorted(subscripted), snippet=self.snippet(node.lineno),
            assigns=assigns,
        )

    def visit_defs(self, body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.facts.defs.append(self.def_fact(node, prefix + node.name))
            elif isinstance(node, ast.ClassDef):
                self.facts.defs.append(self.def_fact(node, prefix + node.name))
                self.visit_defs(node.body, prefix + node.name + ".")

    # ---------------------------------------------------------- module level

    def module_level(self, tree: ast.Module) -> None:
        call_nodes: list[ast.Call] = []
        code: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.facts.imports.append(ImportFact(module=alias.name, line=node.lineno))
                elif node.module:
                    self.facts.imports.append(ImportFact(module=node.module, names=[a.name for a in node.names],
                                                         line=node.lineno))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None:
                    continue
                for t in targets:
                    if not isinstance(t, ast.Name):
                        continue
                    self._assignment(t.id, value, node.lineno)
            elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test) and "__main__" in ast.unparse(node.test):
                self.facts.has_main_guard = True
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                code.append(node)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        call_nodes.append(sub)
                        self.langgraph(sub)
        index = {id(c): i for i, c in enumerate(call_nodes)}
        self.facts.module_calls = [self.call_fact(c, index) for c in call_nodes]
        self.module_assigns = self.assigns(code, index)

    def _assignment(self, name: str, value: ast.AST, line: int) -> None:
        text = string_value(value)
        if text is not None and len(text) >= 40 and PROMPT_MARKERS.search(text):
            self.facts.prompts.append(PromptFact(name=name, line=line, preview=text[:MAX_SNIPPET],
                                                 has_internal_notes=bool(INTERNAL_NOTES.search(text))))
        if isinstance(value, ast.Dict) and value.keys and all(isinstance(v, ast.Name) for v in value.values):
            entries = {}
            for k, v in zip(value.keys, value.values):
                ok, key = literal(k)
                if ok and isinstance(key, str):
                    entries[key] = v.id  # type: ignore[union-attr]
            if entries:
                self.facts.registries.append(RegistryFact(name=name, line=line, entries=entries))
        if UPPER.match(name):
            ok, lit = literal(value)
            if ok and not (isinstance(lit, str) and len(lit) > 120):
                self.facts.consts.append(ConstFact(name=name, value=lit, line=line, snippet=self.snippet(line)))


def extract_file(path: Path, root: Path) -> FileFacts:
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FileFacts(path=rel, error=f"{type(exc).__name__}: {exc}")
    return extract_source(rel, source)


MODULE = "<module>"


def extract_source(rel: str, source: str) -> FileFacts:
    """Facts for one Python source held in memory (files never need to exist on disk)."""
    try:
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, ValueError) as exc:
        return FileFacts(path=rel, error=f"{type(exc).__name__}: {exc}")
    ex = _Extractor(rel, source)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            ex.parents[child] = parent
    ex.module_level(tree)
    ex.visit_defs(tree.body)
    if ex.facts.module_calls:
        # Scripts, notebooks and Streamlit apps do their work at module level: model it as a pseudo-function.
        names = sorted({n.id for stmt in tree.body if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        last = max((getattr(s, "end_lineno", s.lineno) or s.lineno for s in tree.body), default=1)
        ex.facts.defs.append(DefFact(name=MODULE, qualname=MODULE, kind="function", line=1, end_line=last,
                                     calls=list(ex.facts.module_calls), names_used=names,
                                     snippet=f"module-level code of {rel}", assigns=ex.module_assigns))
    # de-duplicate langgraph facts (a call can be seen from both a def and a loop pass)
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for lg in ex.facts.langgraph:
        key = (lg.kind, lg.line, lg.node, lg.source, lg.target)
        if key not in seen:
            seen.add(key)
            unique.append(lg)
    ex.facts.langgraph = unique
    return ex.facts
