from lib.ai.graph import build_graph, run_graph


def run_mentor():
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}

    run_graph(graph, config)


if __name__ == "__main__":
    run_mentor()
