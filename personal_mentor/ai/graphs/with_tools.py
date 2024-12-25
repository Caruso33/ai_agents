from langgraph.graph import END, START, StateGraph

from ..nodes import reasoner
from ..state import State
from ..tools import tools
from .utils import BasicToolNode, route_tools, run_graph


def build_graph():
    """
    Builds the graph for the personal mentor with tools.

    This graph consists of a reasoner node and a tools node. The reasoner node
    is the entry point of the graph and it routes the input messages to either
    the tools node or the end of the graph depending on whether the latest
    message from the reasoner is a tool call or not.

    The tools node is responsible for running the tools requested in the last
    AIMessage.

    The graph is built using the StateGraph class and the add_node and
    add_conditional_edges methods.

    Returns:
        A compiled StateGraph instance.
    """

    builder = StateGraph(State)  # MessagesState

    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", BasicToolNode(tools))
    # builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        # If the latest message (result) from node reasoner is
        #   a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is
        #   a not a tool call -> tools_condition routes to END
        route_tools,
        # tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "reasoner")

    graph = builder.compile()

    return graph


if __name__ == "__main__":
    graph = build_graph()
    # save_graph(graph)

    # query = "What is 2 times Brad Pitt's age?"
    # # query = "what is the current weather in Bangkok in Thailand times 3 minus the age of Brad Pitt?"

    # messages = graph.invoke({"messages": [HumanMessage(content=query)]})

    # for m in messages["messages"]:
    #     m.pretty_print()

    run_graph(graph)
