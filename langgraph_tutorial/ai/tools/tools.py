"""
A collection of tools for the AI.

This module exports a list of tools which can be used in the graph. 
The tools are:

- `add`: a tool that adds two numbers
- `divide`: a tool that divides two numbers
- `multiply`: a tool that multiplies two numbers
- `subtract`: a tool that subtracts two numbers
- `weather_forecast`: a tool that retrieves the current weather forecast for a
  given location
- `BraveSearch`: a tool that searches the web using Brave
- `DuckDuckGoSearchRun`: a tool that searches the web using DuckDuckGo

The tools are loaded from their respective modules and exposed in this module
for convenience.
"""

import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun

from langgraph_tutorial.ai.models import RequestAssistance

from ..llms import LLM
from .maths import add, divide, multiply, subtract
from .weather import weather_forecast

load_dotenv()

ddg_search = DuckDuckGoSearchRun()

brave_search = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY"), search_kwargs={"count": 3}
)

math_tools = [
    tool(add),
    tool(subtract),
    tool(multiply),
    tool(divide),
]
search_tools = [
    # ddg_search,
    brave_search,
]
other_tools = [
    tool(weather_forecast),
    RequestAssistance,
]

tools = [
    *math_tools,
    *other_tools,
    *search_tools,
]
llm_with_tools = LLM.bind_tools(tools)


if __name__ == "__main__":

    query = "How old is Brad Pitt?"
    # query = "What is 2 times 5?"
    # query = "What is the current weather in Bangkok in Thailand?"
    query = "what is the current weather in Bangkok in Thailand times 3 minus the age of Brad Pitt?"

    # result = ddg_search.invoke(query)
    # result = brave_search.invoke(query)
    result = llm_with_tools.invoke([query])

    print(f"result {result}\n")
