from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chatbot_service import ChatbotService

# Define the router for chatbot endpoints
router = APIRouter()
chatbot_service = ChatbotService()
# Pydantic model for the request body
class ChatRequest(BaseModel):
    message: str

# Pydantic model for the response body
class ChatResponse(BaseModel):
    reply: str

@router.post("/message", response_model=ChatResponse)
async def handle_chat_message(request: ChatRequest):
    """
    Endpoint to handle an incoming chat message.
    This is a placeholder and would typically call a service function.
    """
    # In a real application, you would call a service function like:
    # reply = chatbot_service.get_response(request.user_id, request.message)
    try:
        reply = chatbot_service.get_concise_answer(request.message)
    except:
        reply = "Sorry, I'm having trouble processing your request right now."

    return ChatResponse(reply=reply)
