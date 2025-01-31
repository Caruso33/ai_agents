"""
Langchain Google Vertex AI module.

This module provides a langchain interface to Google Cloud Vertex AI's 
chat models.
It uses the `langchain_google_vertexai` library under the hood.

This module is used to define the LLM that is used by the graph.

"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_vertex_models(model_name: str, **kwargs):
    """
    Retrieves a ChatGoogleGenerativeAI model instance configured with the specified
    model name and additional parameters.

    Args:
        model_name: A string representing the name of the model.
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        An instance of ChatGoogleGenerativeAI configured with the specified model name and parameters.
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("AI_STUDIO_API_KEY"),
        max_output_tokens=8192,
        **kwargs,
    )


LLM = get_vertex_models("gemini-2.0-flash-exp")
LLM_RANDOM = get_vertex_models("gemini-1.5-pro-002", temperature=1.0)

CONTEXT_WINDOW = {
    ChatGoogleGenerativeAI: 1_000_000,
}.get(type(LLM), 0)


TOKENS_RESERVED_FOR_PROMPT = 2_000
MAX_TOKEN_THRESHOLD = 200_000
MAX_TOKENS_ALLOWED = (
    min([CONTEXT_WINDOW, MAX_TOKEN_THRESHOLD]) - TOKENS_RESERVED_FOR_PROMPT
)
