import json
import os
from typing import Optional

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph

from ..state import State


def save_graph(graph):
    graph_png = graph.get_graph(xray=True).draw_mermaid_png()
    path = os.path.join(os.getcwd(), "out", "graph.png")

    with open(path, "wb") as f:
        f.write(graph_png)


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def resume_graph(
    graph: CompiledStateGraph, config: dict, message: Optional[AnyMessage] = None
):
    """
    Resume the graph from the last node.

    This function takes a graph and a config and message as input. If the graph is
    currently paused at a node, and the message is an AIMessage or a ToolMessage,
    it will resume the graph from that node and return the next events until the
    graph is paused again.

    Args:
        graph: The graph to resume.
        config: The config to use when resuming the graph.
        message: The message to pass to the graph when resuming.

    Returns:
        An iterator over the events of the graph.
    """
    snapshot = graph.get_state(config)

    if snapshot.next[0] == "tools":
        events = graph.stream(
            message, config, stream_mode="values"
        )  # None means continue the graph
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


def stream_graph_updates(
    graph: CompiledStateGraph, user_input: str, config: dict = None
) -> None:
    """
    Stream events from the graph.

    This function takes a graph, user input, and a config as input. It will
    stream the events of the graph as it runs, starting with the given user
    input. If the graph is paused, it will resume at the last node.

    Args:
        graph: The graph to stream.
        user_input: The user input to pass to the graph.
        config: The config to use when running the graph. If None, use the default
            config.
    """

    if not config:
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    else:
        for event in graph.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()

    should_play_with_state = True

    if should_play_with_state:
        intervention = True

        if not intervention:
            resume_graph(graph, config, None)

        else:
            snapshot = graph.get_state(config)
            print("next snapshot: ", snapshot.next)

            existing_message = snapshot.values["messages"][-1]
            print("tool_calls: ", existing_message.tool_calls)

            # overriding tool call
            print("Original")
            print("Message ID", existing_message.id)
            print(existing_message.tool_calls[0])
            new_tool_call = existing_message.tool_calls[0].copy()
            new_tool_call["args"]["query"] = "What the heck is 0 times 0?"
            new_message = AIMessage(
                content=existing_message.content,
                tool_calls=[new_tool_call],
                # Important! The ID is how LangGraph knows to REPLACE the message
                # in the state rather than APPEND this messages
                id=existing_message.id,
            )
            graph.update_state(config, {"messages": [new_message]})

            # adding new messages to state
            tool_answer = "0 is the answer!"
            new_messages = [
                # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
                ToolMessage(
                    content=tool_answer,
                    tool_call_id=existing_message.tool_calls[0]["id"],
                ),
                # And then directly "put words in the LLM's mouth" by populating its response.
                AIMessage(content=tool_answer),
            ]
            new_messages[-2].pretty_print()
            new_messages[-1].pretty_print()
            graph.update_state(
                # Which state to update
                config,
                # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
                # to the existing state. We will review how to update existing messages in the next section!
                {"messages": new_messages},
            )

            # overriding the last ai message
            existing_message = graph.get_state(config).values["messages"][-1]
            new_ai_message = existing_message.copy()
            new_ai_message.content = "It's 0 dumbass! What a stupid question."

            new_message = AIMessage(
                content=new_ai_message.content,
                # Important! The ID is how LangGraph knows to REPLACE the message
                # in the state rather than APPEND this messages
                id=existing_message.id,
            )
            new_message.pretty_print()
            graph.update_state(
                # Which state to update
                config,
                # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
                # to the existing state. We will review how to update existing messages in the next section!
                {"messages": [new_message]},
            )

            # add a new ai message
            graph.update_state(
                config,
                {
                    "messages": [
                        AIMessage(content="I am stupid and don't know the answer.")
                    ]
                },
                # Which node for this function to act as. It will automatically continue
                # processing as if this node just ran.
                as_node="reasoner",
            )
            snapshot = graph.get_state(config)
            snapshot.values["messages"][-1].pretty_print()
            print("next snapshot: ", snapshot.next)

        snapshot = graph.get_state(config)
        existing_messages = snapshot.values["messages"]

        print("Existing Messages:")
        for i, msg in enumerate(existing_messages, 1):
            print(f"Message {i}:")
            print(f"  Type: {type(msg).__name__}")
            print(f"  Content: {msg.content}")
            if hasattr(msg, "tool_calls"):
                print(f"  Tool Calls: {msg.tool_calls}")
            print()


def run_graph(graph: CompiledStateGraph, config: dict = None):
    """
    Run the graph with the given config.

    This function is used to run a graph with a given config. It will start the
    graph from the beginning and continue running until the graph stops.

    Parameters
    ----------
    graph : CompiledStateGraph
        The graph to run
    config : dict, optional
        The config to use when running the graph, by default None
    """
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph, user_input, config)

        except (KeyboardInterrupt, EOFError):
            print("Goodbye!")
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            break
