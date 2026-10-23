"""PARSE for JavaScript / TypeScript: tree-sitter → FileFacts (same shape as the Python extractor).

Object-literal arguments are treated like keyword arguments: `new ChatOpenAI({ temperature: 0.9 })`
yields kwargs {"temperature": 0.9}, so the same miners work across languages.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..analyzer.extract import INTERNAL_NOTES, MAX_SNIPPET, MODULE, PROMPT_MARKERS, UPPER
from ..analyzer.facts import CallFact, ConstFact, DefFact, FileFacts, ImportFact, LangGraphFact, PromptFact, RegistryFact
from ..core.schemas import ConfigValue

LANG_BY_EXT = {".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
               ".jsx": "javascript"}
FUNC_NODES = {"function_declaration", "generator_function_declaration", "method_definition"}
FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
LOOP_NODES = {"for_statement", "for_in_statement", "while_statement", "do_statement"}
LOOP_METHODS = {"map", "forEach", "flatMap", "filter", "reduce"}


@lru_cache
def _parser(lang: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(lang)


def _text(node) -> str:
    return node.text.decode("utf-8", "replace") if node is not None else ""


def _line(node) -> int:
    return node.start_point[0] + 1


def _literal(node) -> tuple[bool, ConfigValue]:
    if node is None:
        return False, None
    t = node.type
    if t == "number":
        raw = _text(node).replace("_", "")
        try:
            return True, int(raw) if re.fullmatch(r"-?\d+", raw) else float(raw)
        except ValueError:
            return False, None
    if t in ("true", "false"):
        return True, t == "true"
    if t in ("null", "undefined"):
        return True, None
    if t == "string":
        return True, _text(node)[1:-1]
    if t == "template_string" and not any(c.type == "template_substitution" for c in node.children):
        return True, _text(node)[1:-1]
    if t == "unary_expression" and _text(node).startswith("-"):
        ok, v = _literal(node.named_children[0] if node.named_children else None)
        return (ok, -v) if ok and isinstance(v, (int, float)) else (False, None)
    return False, None


def _string_value(node) -> str | None:
    """Text of a string/template literal (substitutions kept as ${expr})."""
    if node is None:
        return None
    if node.type in ("string", "template_string"):
        return _text(node)[1:-1]
    if node.type == "binary_expression":
        parts = [_string_value(c) for c in node.named_children]
        return "".join(parts) if all(p is not None for p in parts) else None
    if node.type == "parenthesized_expression" and node.named_children:
        return _string_value(node.named_children[0])
    return None


def _dotted(node) -> str:
    if node is None:
        return "?"
    t = node.type
    if t in ("identifier", "property_identifier", "this", "super", "private_property_identifier"):
        return _text(node)
    if t == "member_expression":
        return f"{_dotted(node.child_by_field_name('object'))}.{_text(node.child_by_field_name('property'))}"
    if t == "call_expression":
        return f"{_dotted(node.child_by_field_name('function'))}()"
    if t == "new_expression":
        return f"{_dotted(node.child_by_field_name('constructor'))}()"
    if t in ("await_expression", "parenthesized_expression", "non_null_expression", "as_expression") and node.named_children:
        return _dotted(node.named_children[0])
    if t == "subscript_expression":
        return f"{_dotted(node.child_by_field_name('object'))}[]"
    return "?"


class _JS:
    def __init__(self, rel: str, source: str):
        self.rel = rel
        self.lines = source.splitlines()
        self.facts = FileFacts(path=rel)

    def snippet(self, line: int) -> str:
        return self.lines[line - 1].strip()[:MAX_SNIPPET] if 0 < line <= len(self.lines) else ""

    def _context(self, node) -> tuple[bool, bool]:
        in_loop = in_try = False
        child, parent = node, node.parent
        while parent is not None and parent.type not in FUNC_NODES | FUNC_VALUES | {"class_declaration"}:
            if parent.type in LOOP_NODES:
                in_loop = True
            if parent.type == "try_statement" and child == parent.child_by_field_name("body"):
                in_try = True
            child, parent = parent, parent.parent
        # callbacks of array iteration methods count as loops
        p = node.parent
        while p is not None:
            if p.type in FUNC_VALUES and p.parent is not None and p.parent.type == "arguments":
                call = p.parent.parent
                if call is not None and call.type == "call_expression" and _dotted(call.child_by_field_name("function")).rsplit(".", 1)[-1] in LOOP_METHODS:
                    in_loop = True
                break
            if p.type in FUNC_NODES:
                break
            p = p.parent
        return in_loop, in_try

    def call_fact(self, node) -> CallFact:
        if node.type == "new_expression":
            callee = _dotted(node.child_by_field_name("constructor"))
            args = node.child_by_field_name("arguments")
        else:
            callee = _dotted(node.child_by_field_name("function"))
            args = node.child_by_field_name("arguments")
        kwargs: dict[str, ConfigValue] = {}
        str_args: list[str] = []
        num_args: list[float] = []
        for a in (args.named_children if args is not None else []):
            if a.type == "object":
                for pair in a.named_children:
                    if pair.type == "pair":
                        key = _text(pair.child_by_field_name("key")).strip("'\"")
                        ok, v = _literal(pair.child_by_field_name("value"))
                        kwargs[key] = v if ok else _text(pair.child_by_field_name("value"))[:120]
                    elif pair.type == "shorthand_property_identifier":
                        kwargs[_text(pair)] = _text(pair)
            else:
                ok, v = _literal(a)
                if ok and isinstance(v, str):
                    str_args.append(v)
                elif ok and isinstance(v, (int, float)) and not isinstance(v, bool):
                    num_args.append(float(v))
        in_loop, in_try = self._context(node)
        target = None
        p = node.parent
        while p is not None and p.type in ("await_expression", "parenthesized_expression", "as_expression"):
            p = p.parent
        if p is not None and p.type == "variable_declarator":
            target = _text(p.child_by_field_name("name"))
        elif p is not None and p.type == "assignment_expression":
            target = _dotted(p.child_by_field_name("left")).replace("this.", "")
        return CallFact(callee=callee, line=_line(node), kwargs=kwargs, str_args=str_args,
                        snippet=self.snippet(_line(node)), in_loop=in_loop, in_try=in_try, target=target,
                        num_args=num_args)

    def langgraph(self, call: CallFact) -> None:
        name = call.callee.rsplit(".", 1)[-1]
        if name == "addNode" and call.str_args:
            self.facts.langgraph.append(LangGraphFact(kind="add_node", line=call.line, node=call.str_args[0],
                                                      fn=None, snippet=call.snippet))
        elif name == "addEdge" and len(call.str_args) >= 2:
            src, dst = (a if a not in ("__start__", "__end__") else a for a in call.str_args[:2])
            self.facts.langgraph.append(LangGraphFact(kind="add_edge", line=call.line, source=src, target=dst,
                                                      snippet=call.snippet))

    def walk_calls(self, body) -> tuple[list[CallFact], set[str], set[str]]:
        calls, names, subscripted = [], set(), set()
        stack = [body]
        while stack:
            n = stack.pop()
            if n.type in ("call_expression", "new_expression"):
                fact = self.call_fact(n)
                calls.append(fact)
                self.langgraph(fact)
                # addNode("name", fn): capture the function identifier
                if fact.callee.endswith("addNode") and self.facts.langgraph and self.facts.langgraph[-1].fn is None:
                    args = n.child_by_field_name("arguments")
                    if args is not None and len(args.named_children) > 1:
                        self.facts.langgraph[-1].fn = _dotted(args.named_children[1])
            elif n.type in ("identifier", "shorthand_property_identifier"):
                names.add(_text(n))
            elif n.type == "subscript_expression":
                obj = n.child_by_field_name("object")
                if obj is not None and obj.type == "identifier":
                    subscripted.add(_text(obj))
            if n.type in FUNC_NODES and n is not body:
                continue  # nested named functions are their own defs
            stack.extend(reversed(n.children))
        calls.sort(key=lambda c: c.line)
        return calls, names, subscripted

    def add_def(self, name: str, qual: str, node, body, kind: str = "function", decorators=()) -> None:
        calls, names, subscripted = self.walk_calls(body) if body is not None else ([], set(), set())
        params = node.child_by_field_name("parameters")
        args = [_text(p).split(":")[0].split("=")[0].strip("?{} ") for p in (params.named_children if params else [])]
        self.facts.defs.append(DefFact(
            name=name, qualname=qual, kind=kind, line=_line(node), end_line=node.end_point[0] + 1,
            decorators=list(decorators), args=[a for a in args if a], calls=calls, names_used=sorted(names),
            subscripted=sorted(subscripted), snippet=self.snippet(_line(node))))

    def visit(self, root) -> None:
        module_nodes = []
        for top in root.named_children:
            node = top.child_by_field_name("declaration") if top.type == "export_statement" else top
            if node is None:
                node = top.named_children[0] if top.named_children else top
            t = node.type
            if t == "import_statement":
                src = node.child_by_field_name("source")
                names = [_text(i) for i in node.named_children if i.type == "import_clause" for i in _idents(i)]
                self.facts.imports.append(ImportFact(module=_text(src).strip("'\""), names=names, line=_line(node)))
            elif t in ("function_declaration", "generator_function_declaration"):
                nm = _text(node.child_by_field_name("name"))
                self.add_def(nm, nm, node, node.child_by_field_name("body"))
            elif t in ("class_declaration", "abstract_class_declaration"):
                cname = _text(node.child_by_field_name("name"))
                self.add_def(cname, cname, node, None, kind="class")
                body = node.child_by_field_name("body")
                for m in (body.named_children if body else []):
                    if m.type == "method_definition":
                        mn = _text(m.child_by_field_name("name"))
                        self.add_def(mn, f"{cname}.{mn}", m, m.child_by_field_name("body"))
            elif t in ("lexical_declaration", "variable_declaration"):
                handled = False
                for decl in node.named_children:
                    if decl.type != "variable_declarator":
                        continue
                    nm = _text(decl.child_by_field_name("name"))
                    value = decl.child_by_field_name("value")
                    if value is not None and value.type in FUNC_VALUES:
                        self.add_def(nm, nm, decl, value.child_by_field_name("body"))
                        handled = True
                        continue
                    self.assignment(nm, value, _line(decl))
                if not handled:
                    module_nodes.append(node)
            else:
                module_nodes.append(node)
        # module-level code: requires, top-level calls, scripts
        all_calls, names, subs = [], set(), set()
        for n in module_nodes:
            c, nm, sb = self.walk_calls(n)
            all_calls += c
            names |= nm
            subs |= sb
            for call in c:
                if call.callee in ("require",) and call.str_args:
                    self.facts.imports.append(ImportFact(module=call.str_args[0], line=call.line))
        self.facts.module_calls = all_calls
        if all_calls:
            self.facts.defs.append(DefFact(name=MODULE, qualname=MODULE, kind="function", line=1,
                                           end_line=max(len(self.lines), 1), calls=all_calls, names_used=sorted(names),
                                           subscripted=sorted(subs), snippet=f"module-level code of {self.rel}"))

    def assignment(self, name: str, value, line: int) -> None:
        if value is None:
            return
        text = _string_value(value)
        if text is not None and len(text) >= 40 and PROMPT_MARKERS.search(text):
            self.facts.prompts.append(PromptFact(name=name, line=line, preview=text[:MAX_SNIPPET],
                                                 has_internal_notes=bool(INTERNAL_NOTES.search(text))))
        if value.type == "object":
            entries = {}
            for pair in value.named_children:
                if pair.type == "shorthand_property_identifier":
                    entries[_text(pair)] = _text(pair)
                elif pair.type == "pair" and pair.child_by_field_name("value") is not None \
                        and pair.child_by_field_name("value").type == "identifier":
                    entries[_text(pair.child_by_field_name("key")).strip("'\"")] = _text(pair.child_by_field_name("value"))
            if entries and len(entries) == len(value.named_children):
                self.facts.registries.append(RegistryFact(name=name, line=line, entries=entries))
        if UPPER.match(name):
            ok, lit = _literal(value)
            if ok and not (isinstance(lit, str) and len(lit) > 120):
                self.facts.consts.append(ConstFact(name=name, value=lit, line=line, snippet=self.snippet(line)))


def _idents(node):
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            out.append(n)
        stack.extend(n.named_children)
    return out


def extract_js(rel: str, source: str, ext: str) -> FileFacts:
    try:
        tree = _parser(LANG_BY_EXT[ext]).parse(source.encode("utf-8", "replace"))
    except Exception as exc:  # grammar download / parse failures must not kill the analysis
        return FileFacts(path=rel, error=f"{type(exc).__name__}: {exc}")
    js = _JS(rel, source)
    js.visit(tree.root_node)
    if tree.root_node.has_error and not js.facts.defs:
        js.facts.error = "parse errors"
    return js.facts
