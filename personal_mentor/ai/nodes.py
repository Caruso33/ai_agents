from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

from .llms import LLM
from .state import State
from .tools import llm_with_tools

sys_prompt = """
You are a helpful assistant who can use several tools at your disposal.

Performing arithmetics on a set of inputs, getting the weather forecast and running a websearch for things you don't know.
"""
# If you need to search, try duckduckgo search first, if you get an error or rate limited, use brave search.

sys_msg = SystemMessage(content=sys_prompt)


def chatbot(state: State):
    return {"messages": [LLM.invoke(state["messages"])]}


def reasoner(state: MessagesState):
    # print('state:')
    # for k, v in state.items():
    #     print(f'{k}: {v}')

    # query = state["query"]
    # print(f'query {query}\n')

    messages = state["messages"]
    # if len(messages) == 1:
    #     messages = [sys_msg, *messages]

    # print(f"messages {messages}\n")

    return {"messages": [llm_with_tools.invoke(messages)]}


if __name__ == "__main__":
    state = {"messages": [HumanMessage(content="What is 2 times Brad Pitt's age?")]}

    result = reasoner(state)
    for m in result["messages"]:
        m.pretty_print()
