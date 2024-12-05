from .llms import LLM
from .tools import tools

llm_with_tools = LLM.bind_tools(tools)
