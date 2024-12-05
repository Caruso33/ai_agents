import os

from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .nodes import reasoner
from .tools import tools


def build_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "reasoner")

    graph = builder.compile()

    return graph


def save_graph(graph):
    graph_png = graph.get_graph(xray=True).draw_mermaid_png()
    path = os.path.join(os.getcwd(), "out", "graph.png")

    with open(path, "wb") as f:
        f.write(graph_png)


if __name__ == "__main__":
    graph = build_graph()

    save_graph(graph)

    # messages = [HumanMessage(content="What is 2 times Brad Pitt's age?")]
    # messages = graph.invoke({"messages": messages})

    # for m in messages["messages"]:
    #     m.pretty_print()
