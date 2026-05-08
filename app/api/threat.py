from fastapi import APIRouter
from app.models.message import Message
from app.models.threat import Threat
from app.services.llm_service import LLMService
from app.services.multi_model_service import MultiModelLLMService

router = APIRouter()
llm_service = LLMService()
multi_service = MultiModelLLMService()

@router.post("/analyze", response_model=Threat)
async def analyze(message: Message):
    return await llm_service.analyze_message(message.text)