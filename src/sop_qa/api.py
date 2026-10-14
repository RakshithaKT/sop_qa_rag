from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sop_qa.ingestion.extractors import extract_pdf
from sop_qa.ingestion.chunking import chunk_document
from sop_qa.VectorDb.vector_store import VectorStore
from sop_qa.ingestion.chunking import chunk_document

router = APIRouter()

vector_store = VectorStore(path = "data/qdrant")
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



@router.post("/api/v1/documents/chunk")
async def chunk_uploaded_document(
    file: UploadFile = File(...),
):
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

        extracted = extract_pdf(content, file.filename)
        chunks = chunk_document(extracted)

        return {
            "filename": file.filename,
            "page_count": extracted["page_count"],
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not process the PDF.",
        ) from exc

    finally:
        await file.close()



class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


@router.post("/api/v1/documents/index")
async def index_document(file: UploadFile = File(...)):
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

        extracted = extract_pdf(content, file.filename)
        chunks = chunk_document(extracted)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text was found in the PDF.",
            )

        stored_count = vector_store.add_chunks(chunks)

        return {
            "filename": file.filename,
            "chunk_count": len(chunks),
            "indexed_count": stored_count,
            "message": "Document indexed successfully.",
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not index the PDF.",
        ) from exc

    finally:
        await file.close()


@router.post("/api/v1/search")
def search_documents(request: SearchRequest):
    try:
        results = vector_store.search(
            question=request.question,
            limit=request.limit,
        )

        return {
            "question": request.question,
            "result_count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Search failed.",
        ) from exc
