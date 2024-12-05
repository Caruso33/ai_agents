from langchain_community.tools import DuckDuckGoSearchRun


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


ddg_search = DuckDuckGoSearchRun()

tools = [add, multiply, divide, ddg_search]


if __name__ == "__main__":
    print(ddg_search.invoke("How old is Brad Pitt?"))
