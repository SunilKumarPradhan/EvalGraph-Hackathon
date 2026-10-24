from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def search_docs(query: str) -> str:
    """Search the internal docs."""
    return f"results for {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic."""
    return str(eval(expression))  # noqa: S307  (deliberately unsafe sample)


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).bind_tools([search_docs, calculator])


def call_model(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def build():
    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode([search_docs, calculator]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())


app = build()
