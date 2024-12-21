import json
import os

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph

from ..state import State


def save_graph(graph):
    graph_png = graph.get_graph(xray=True).draw_mermaid_png()
    path = os.path.join(os.getcwd(), "out", "graph.png")

    with open(path, "wb") as f:
        f.write(graph_png)


def stream_graph_updates(
    graph: CompiledStateGraph, user_input: str, config: dict = None
):

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

    # snapshot = graph.get_state(config)
    # print("next snapshot: ", snapshot.next)

    # existing_message = snapshot.values["messages"][-1]
    # print("tool_calls: ", existing_message.tool_calls)

    # if snapshot.next[0] == "tools":
    #     events = graph.stream(None, config, stream_mode="values") # None means continue the graph
    #     for event in events:
    #         if "messages" in event:
    #             event["messages"][-1].pretty_print()
    #         else:
    #             print("NO MESSAGES IN EVENT: ", event)


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


def run_graph(graph: CompiledStateGraph, config: dict = None):
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
