from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

# Define the router for dashboard endpoints
router = APIRouter()

# Pydantic model for the response body
class DashboardData(BaseModel):
    total_requests: int
    active_users: int
    model_performance: Dict[str, float]


@router.get("/data", response_model=DashboardData)
async def get_dashboard_data():
    """
    Endpoint to retrieve data for the dashboard.
    This is a placeholder and would typically call a service function.
    """
    # In a real application, you would fetch this data from a database or a monitoring service.
    # data = dashboard_service.get_analytics()
    
    # For this example, we return simple mocked data.
    mock_data = {
        "total_requests": 1024,
        "active_users": 128,
        "model_performance": {
            "chatbot_accuracy": 0.94,
            "summarizer_rouge_score": 0.88,
        },
    }
    return DashboardData(**mock_data)
