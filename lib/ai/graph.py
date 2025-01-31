from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..tools import tools
from .nodes import reasoner
from .state import State


def build_graph():
    """
    Constructs and compiles a state graph for managing AI tool interactions.

    The graph is initialized with a reasoner node and a tools node. The reasoner
    node is the starting point and processes the input data. If the reasoner
    determines a tool is required, it routes to the tools node; otherwise, it
    concludes the process.

    The graph is compiled with memory checkpointing to save state between
    interactions.

    Returns:
        A compiled StateGraph instance with memory management.
    """
    builder = StateGraph(State)

    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "reasoner")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


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

            first_msg = HumanMessage(content=user_input)
            first_msg.pretty_print()

            for event in graph.stream({"messages": [first_msg]}, config):
                for value in event.values():
                    value["messages"][-1].pretty_print()

            # snapshot = graph.get_state(config)
            # tool_calls = snapshot.values["tools_called"]
            # print(f"Tool calls: {tool_calls}")

        except (KeyboardInterrupt, EOFError):
            print("Goodbye!")
            break

        except Exception as e:
        
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    graph = build_graph()
    # save_graph(graph)

    # query = "What is 2 times Brad Pitt's age?"
    # # query = "what is the current weather in Bangkok in Thailand times
    # 3 minus the age of Brad Pitt?"

    # messages = graph.invoke({"messages": [HumanMessage(content=query)]})

    # for m in messages["messages"]:
    #     m.pretty_print()

    config = {"configurable": {"thread_id": "1"}}

    run_graph(graph, config)
