from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState

from lib.ai.llms import LLM
from lib.tools import llm_with_tools

from .models import RequestAssistance
from .state import State

sys_prompt = """
You are a helpful assistant who can use several tools at your disposal.

Performing arithmetics on a set of inputs, getting the weather forecast and 
running a websearch for things you don't know.
"""
# If you need to search, try duckduckgo search first, if you get an error or rate limited, use brave search.

sys_msg = SystemMessage(content=sys_prompt)


def chatbot(state: State):
    """
    This node is the entry point for the chatbot application.
    It takes a state containing a list of messages and returns a new state
    with the messages processed by the chatbot.

    Parameters
    ----------
    state : State
        The state of the application containing a list of messages.

    Returns
    -------
    State
        A new state with the messages processed by the chatbot.
    """
    return {"messages": [LLM.invoke(state["messages"])]}


def chatbot_with_ask_human(state: State):
    """
    This node is the entry point for the chatbot application.
    It takes a state containing a list of messages and returns a new state
    with the messages processed by the chatbot and a boolean indicating
    whether the chatbot requests human assistance.

    Parameters
    ----------
    state : State
        The state of the application containing a list of messages.

    Returns
    -------
    State
        A new state with the messages processed by the chatbot and a boolean
        indicating whether the chatbot requests human assistance.
    """
    response = llm_with_tools.invoke(state["messages"])

    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True

    return {"messages": [response], "ask_human": ask_human}


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

    # print(f"messages {messages}\n")

    return {"messages": [llm_with_tools.invoke(messages)]}


def create_tool_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_tool_response(
                "No response from human.",
                state["messages"][-1],
            )
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


if __name__ == "__main__":
    state = {"messages": [HumanMessage(content="What is 2 times Brad Pitt's age?")]}

    result = reasoner(state)
    for m in result["messages"]:
        m.pretty_print()
