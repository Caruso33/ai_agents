from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..nodes import chatbot_with_ask_human, human_node
from ..state import State
from ..tools import tools
from .utils import run_graph, select_next_node


def build_graph():

    builder = StateGraph(State)  # MessagesState

    builder.add_node("chatbot", chatbot_with_ask_human)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human", human_node)

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END},
    )
    builder.add_edge("tools", "chatbot")
    builder.add_edge("human", "chatbot")

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )

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
