from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..nodes import reasoner
from ..state import State
from ..tools import tools
from .utils import BasicToolNode, route_tools, run_graph, stream_graph_updates


def build_graph():

    builder = StateGraph(State)  # MessagesState

    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", BasicToolNode(tools))
    # builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        route_tools,
        # tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "reasoner")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


if __name__ == "__main__":
    graph = build_graph()
    # save_graph(graph)

    # query = "What is 2 times Brad Pitt's age?"
    # # query = "what is the current weather in Bangkok in Thailand times 3 minus the age of Brad Pitt?"

    # messages = graph.invoke({"messages": [HumanMessage(content=query)]})

    # for m in messages["messages"]:
    #     m.pretty_print()

    config = {"configurable": {"thread_id": "1"}}

    run_graph(graph, config)
