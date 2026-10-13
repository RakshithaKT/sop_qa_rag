from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sop_qa.ingestion.extractors import extract_pdf

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


@router.post("/api/v1/documents/extract")
async def extract_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        return extract_pdf(content, file.filename)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not read the PDF.",
        ) from exc

    finally:
        await file.close()
