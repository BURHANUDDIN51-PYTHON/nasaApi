from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.services.summarize_service import SummarizeService

# Define the router for summarization endpoints
router = APIRouter()
summarize_service = SummarizeService()

# Pydantic model for the request body
class SummarizeRequest(BaseModel):
    text: str


# Pydantic model for the response body
class SummarizeResponse(BaseModel):
    summary: str
    visualization_data: Dict[str, Any]

@router.post("/", response_model=SummarizeResponse)
async def create_summary(request: SummarizeRequest):
    # Call the summarization service with the incoming text
    result = summarize_service.generate_summary(request.text)

    # Extract expected fields with safe defaults
    summary_text = result.get("summary", "")
    visualization_data = result.get("visualization_data", {})

    # Return a properly typed Pydantic response
    return SummarizeResponse(
        summary=summary_text,
        visualization_data=visualization_data
    )