from pydantic import BaseModel


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable
    to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can
    provide the right guidance.
    """

    request: str
