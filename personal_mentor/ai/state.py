# import operator
from typing import Annotated, Sequence, TypedDict

# from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State of the graph."""

    query: str

    # weather: str
    # weather_answer: str

    # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    # messages: Annotated[Sequence[AnyMessage], operator.add]
    messages: Annotated[list, add_messages]
