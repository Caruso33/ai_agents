from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator


class State(BaseModel):
    """
    The State of the graph.

    This class represents the state of the graph which is a dictionary that
    contains the following keys:

    - messages: A list of AnyMessage objects which are the messages exchanged
      between the user and the AI in the chat.

    This class uses the TypedDict type hint from the typing module to specify
    the types of the keys in the dictionary.
    """

    # tools_called: Annotated[list, add_messages]

    messages: Annotated[list, add_messages]

    @field_validator("messages")
    def coerce_to_list(cls, v):
        if not isinstance(v, list):
            return [v]
        return v
