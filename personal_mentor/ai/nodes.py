from langgraph.graph import MessagesState

from .langchain import llm_with_tools
from .prompts import sys_msg


def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
