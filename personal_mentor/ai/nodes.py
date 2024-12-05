from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

from .tools import llm_with_tools

sys_prompt = """
You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.
"""

sys_msg = SystemMessage(content=sys_prompt)


def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
