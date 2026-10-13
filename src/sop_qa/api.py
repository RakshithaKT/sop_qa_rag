from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class AnswerResponse(BaseModel):
    answer: str

@router.post("/api/v1/query", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
# Temporary response; replace this with the RAG pipeline later.
    return AnswerResponse(
answer=f"Your question was received: {request.question}"
)
