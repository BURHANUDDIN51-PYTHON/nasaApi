from fastapi import FastAPI
from app.api.endpoints import chatbot, summarize, dashboard

# Create the FastAPI app instance
app = FastAPI(
    title="AI Services API",
    description="An API for chatbot, summarization, and dashboard services.",
    version="1.0.0",
)

# Include the routers from the endpoints
# Each router handles a specific feature (chatbot, summarize, etc.)
app.include_router(chatbot.router, prefix="/api/v1/chatbot", tags=["Chatbot"])
app.include_router(summarize.router, prefix="/api/v1/summarize", tags=["Summarize"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])

# A simple root endpoint to check if the API is running
@app.get("/", tags=["Root"])
async def read_root():
    """
    A simple health check endpoint.
    """
    return {"message": "Welcome to the AI Services API!"}
