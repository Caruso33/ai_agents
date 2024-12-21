import os

from dotenv import load_dotenv
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun
from langchain.tools import tool
from ..llms import LLM
from .maths import add, divide, multiply, subtract
from .weather import weather_forecast

load_dotenv()

ddg_search = DuckDuckGoSearchRun()

brave_search = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY"), search_kwargs={"count": 3}
)

tools = [
    tool(add),
    tool(subtract),
    tool(multiply),
    tool(divide),
    # ddg_search,
    # brave_search,
    tool(weather_forecast),
]
llm_with_tools = LLM.bind_tools(tools)


if __name__ == "__main__":

    query = "How old is Brad Pitt?"
    # query = "What is 2 times 5?"
    # query = "What is the current weather in Bangkok in Thailand?"
    # query = "what is the current weather in Bangkok in Thailand times 3 minus the age of Brad Pitt?"

    # result = ddg_search.invoke(query)
    result = brave_search.run(query)
    # result = llm_with_tools.invoke([query])

    print(f"result {result}\n")
