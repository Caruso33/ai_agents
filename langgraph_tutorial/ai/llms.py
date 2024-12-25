"""
Langchain Google Vertex AI module.

This module provides a langchain interface to Google Cloud Vertex AI's 
chat models.
It uses the `langchain_google_vertexai` library under the hood.

This module is used to define the LLM that is used by the graph.

"""

import os

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = rf"{os.getcwd()}/vertexai.json"


def get_vertex_models(model_name: str, **kwargs):
    """
    Retrieves a ChatVertexAI model instance configured with the specified
    model name and additional parameters.

    Args:
        model_name: A string representing the name of the model.
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        An instance of ChatVertexAI configured with the specified model name and parameters.
    """
    return ChatVertexAI(
        model_name=model_name,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        max_tokens=8192,
        # **kwargs,
    )


LLM = get_vertex_models("gemini-2.0-flash-exp")
LLM_RANDOM = get_vertex_models("gemini-1.5-pro-002", temperature=1.0)

CONTEXT_WINDOW = {
    ChatVertexAI: 1_000_000,
}.get(type(LLM), 0)


TOKENS_RESERVED_FOR_PROMPT = 2_000
MAX_TOKEN_THRESHOLD = 200_000
MAX_TOKENS_ALLOWED = (
    min([CONTEXT_WINDOW, MAX_TOKEN_THRESHOLD]) - TOKENS_RESERVED_FOR_PROMPT
)
