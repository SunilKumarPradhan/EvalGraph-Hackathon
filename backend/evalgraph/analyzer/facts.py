"""Facts extracted from source code (internal to the analyzer; not part of the HTTP contract)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.schemas import ConfigValue, Evidence, NodeType


class ImportFact(BaseModel):
    module: str
    names: list[str] = Field(default_factory=list)
    line: int


class CallFact(BaseModel):
    callee: str                      # dotted callee, e.g. "requests.get", "get_client().chat", "StateGraph"
    line: int
    kwargs: dict[str, ConfigValue] = Field(default_factory=dict)   # literals, or source text of the expression
    str_args: list[str] = Field(default_factory=list)              # positional string-literal args
    snippet: str = ""
    in_loop: bool = False            # inside for/while/comprehension (same function)
    in_try: bool = False             # inside a try body (same function)
    target: str | None = None        # variable the result is assigned to (`llm = ChatOpenAI()` -> "llm")
    num_args: list[float] = Field(default_factory=list)  # positional numeric literal args
    # data flow (Python only): names / nested calls (indexes into the owning DefFact.calls) inside each argument
    arg_names: list[list[str]] = Field(default_factory=list)
    kwarg_names: dict[str, list[str]] = Field(default_factory=dict)
    arg_calls: list[list[int]] = Field(default_factory=list)
    kwarg_calls: dict[str, list[int]] = Field(default_factory=dict)

    @property
    def receiver(self) -> str:
        return self.callee.split(".", 1)[0].replace("()", "")

    @property
    def last(self) -> str:
        return self.callee.rsplit(".", 1)[-1].replace("()", "")


class AssignFact(BaseModel):
    """`target` may receive data from these names and call results. Also covers `for t in x`, `x += y`,
    `x.append(y)` / `x[k] = y` (target x) and `return y` (target "<return>")."""
    target: str
    line: int
    sources: list[str] = Field(default_factory=list)
    calls: list[int] = Field(default_factory=list)      # indexes into DefFact.calls


class DefFact(BaseModel):
    name: str
    qualname: str                    # "Class.method" or "func"
    kind: Literal["function", "class"]
    line: int
    end_line: int
    decorators: list[str] = Field(default_factory=list)
    decorator_str_args: list[str] = Field(default_factory=list)
    doc: str = ""
    args: list[str] = Field(default_factory=list)
    defaults: dict[str, ConfigValue] = Field(default_factory=dict)
    calls: list[CallFact] = Field(default_factory=list)
    names_used: list[str] = Field(default_factory=list)   # module-level names referenced (for registries)
    subscripted: list[str] = Field(default_factory=list)  # names indexed as NAME[...] (registry dispatch)
    snippet: str = ""
    assigns: list[AssignFact] = Field(default_factory=list)  # data flow inside the body (Python only)


class ConstFact(BaseModel):
    name: str
    value: ConfigValue
    line: int
    snippet: str = ""


class PromptFact(BaseModel):
    name: str
    line: int
    preview: str
    has_internal_notes: bool = False


class RegistryFact(BaseModel):
    """Module-level dict literal mapping string keys to function names, e.g. TOOLS = {"x": fn}."""
    name: str
    line: int
    entries: dict[str, str]


class LangGraphFact(BaseModel):
    kind: Literal["add_node", "add_edge", "add_conditional_edges", "entry"]
    line: int
    source: str | None = None        # node name (add_edge / conditional source)
    target: str | None = None        # node name (add_edge)
    node: str | None = None          # add_node name
    fn: str | None = None            # add_node function name
    mapping: dict[str, str] = Field(default_factory=dict)  # conditional label -> node
    snippet: str = ""


class FileFacts(BaseModel):
    path: str                        # repo-relative, forward slashes
    imports: list[ImportFact] = Field(default_factory=list)
    defs: list[DefFact] = Field(default_factory=list)
    consts: list[ConstFact] = Field(default_factory=list)
    prompts: list[PromptFact] = Field(default_factory=list)
    registries: list[RegistryFact] = Field(default_factory=list)
    module_calls: list[CallFact] = Field(default_factory=list)
    langgraph: list[LangGraphFact] = Field(default_factory=list)
    has_main_guard: bool = False
    error: str | None = None


class Candidate(BaseModel):
    temp_id: str                     # "type:symbol"
    type: NodeType
    label: str
    file: str
    symbol: str
    line: int
    provider: str | None = None
    config: dict[str, ConfigValue] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    score: float = 0.5               # 1.0 = explicit (decorator-typed), lower = heuristic
    reason: str = ""


class CallEdge(BaseModel):
    source: str                      # "file::qualname"
    target: str
    line: int


class RepoFacts(BaseModel):
    name: str
    root: str
    commit: str | None = None
    repo_url: str | None = None
    local_path: str | None = None
    files_scanned: int = 0
    truncated: bool = False
    frameworks: list[str] = Field(default_factory=list)
    has_eval_suite: bool = False
    evalgraph_yaml: dict[str, Any] | None = None
    files: list[FileFacts] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    call_edges: list[CallEdge] = Field(default_factory=list)

    def file(self, path: str) -> FileFacts | None:
        return next((f for f in self.files if f.path == path), None)

    def all_consts(self) -> list[tuple[str, ConstFact]]:
        return [(f.path, c) for f in self.files for c in f.consts]
