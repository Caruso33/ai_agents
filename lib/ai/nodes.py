from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState

from .state import State
from ..tools import llm_with_tools

sys_prompt = """
You are a helpful assistant who can use several tools at your disposal.

Performing arithmetics on a set of inputs, getting the weather forecast and 
running a websearch for things you don't know.
"""
# If you need to search, try duckduckgo search first, if you get an error or rate limited, use brave search.

sys_msg = SystemMessage(content=sys_prompt)


def reasoner(state: MessagesState):
    """
    This node is the reasoning engine for the chatbot application.
    It takes a state containing a list of messages and returns a new state
    with the messages processed by the reasoning engine.

    Parameters
    ----------
    state : MessagesState
        The state of the application containing a list of messages.

    Returns
    -------
    State
        A new state with the messages processed by the reasoning engine.
    """
    # print('state:')
    # for k, v in state.items():
    #     print(f'{k}: {v}')

    # query = state["query"]
    # print(f'query {query}\n')

    messages = state["messages"]
    # if len(messages) == 1:
    #     messages = [sys_msg, *messages]

    # print(f"last message {messages[-1]}\n")

    # if hasattr(messages[-1], "tool_calls"):
    #     return {
    #         "messages": [llm_with_tools.invoke(messages)],
    #         "tools_called": messages[-1].tool_calls,
    #     }

    return {"messages": [llm_with_tools.invoke(messages)]}


if __name__ == "__main__":
    state = {"messages": [HumanMessage(content="What is 2 times Brad Pitt's age?")]}

    result = reasoner(state)
    for m in result["messages"]:
        m.pretty_print()
