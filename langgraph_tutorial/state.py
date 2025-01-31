# import operator
from typing import Annotated, Sequence, TypedDict

# from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class State(TypedDict):  # MessageState is simple State as well to use
    """
    The State of the graph.

    This class represents the state of the graph which is a dictionary that
    contains the following keys:

    - query: The current query from the user.
    - messages: A list of AnyMessage objects which are the messages exchanged
      between the user and the AI in the chat.

    This class uses the TypedDict type hint from the typing module to specify
    the types of the keys in the dictionary.
    """

    # query: str
    ask_human: bool
    # weather: str
    # weather_answer: str

    # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    # messages: Annotated[Sequence[AnyMessage], operator.add]
    messages: Annotated[list, add_messages]
