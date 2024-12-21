from langgraph.graph import END, START, StateGraph

from ..nodes import chatbot
from ..state import State
from .utils import run_graph


def build_graph():

    builder = StateGraph(State)

    builder.add_node("chatbot", chatbot)

    builder.add_edge(START, "chatbot")

    builder.add_edge("chatbot", END)

    graph = builder.compile()

    return graph


if __name__ == "__main__":
    graph = build_graph()

    run_graph(graph)
