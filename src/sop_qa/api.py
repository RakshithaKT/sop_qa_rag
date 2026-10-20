from functools import lru_cache
import logging
import hashlib
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from sop_qa.VectorDb.vector_store import VectorStore
from sop_qa.generation.response import (
    AnswerGenerator,
    RetrievedEvidence,
)
from sop_qa.ingestion.chunking import chunk_document
from sop_qa.ingestion.extractors import extract_pdf
from sop_qa.ingestion.document_registry import (
    DOCUMENTS_DIR,
    claim_document,
    initialize_registry,
    update_document_status,
)

router = APIRouter()

vector_store = VectorStore(path="data/qdrant")
initialize_registry()

@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    """Create the generator only when an answer is requested."""
    return AnswerGenerator()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


def _to_evidence(result: dict) -> RetrievedEvidence:
    """Convert a Qdrant search result into the generation input."""
    source = (
        result.get("source")
        or result.get("filename")
        or result.get("source_file")
        or "Unknown source"
    )

    page = result.get("page", result.get("page_number"))
    chunk_id = result.get("chunk_id")

    return RetrievedEvidence(
        text=str(result.get("text") or ""),
        source=str(source),
        page=page if isinstance(page, int) else None,
        chunk_id=str(chunk_id) if chunk_id is not None else None,
    )


@router.post("/api/v1/query", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Retrieve SOP evidence and generate a grounded answer."""
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
    )
    
    
    try:
        results = vector_store.search(
            question=request.question,
            limit=5,
        )

        evidence = [
            _to_evidence(result)
            for result in results
            if result.get("text")
        ]

        generator = get_answer_generator()
        generated = await generator.generate(
            question=request.question,
            evidence=evidence,
        )

        sources = []
        for item in generated.sources:
            # Match the returned source to the retrieved result.
            matching_result = next(
                (
                    result
                    for result in results
                    if result.get("chunk_id") == item.chunk_id
                    and item.chunk_id is not None
                ),
                None,
            )

            # Fall back to source/page matching when chunk IDs are absent.
            if matching_result is None:
                matching_result = next(
                    (
                        result
                        for result in results
                        if (
                            result.get("source")
                            or result.get("filename")
                            or result.get("source_file")
                            or "Unknown source"
                        ) == item.source
                        and result.get(
                            "page", result.get("page_number")
                        ) == item.page
                    ),
                    {},
                )

            sources.append(
                SourceCitation(
                    source=item.source,
                    page=item.page,
                    chunk_id=item.chunk_id,
                    score=float(matching_result.get("score", 0.0)),
                )
            )

        return AnswerResponse(
            answer=generated.answer,
            sources=sources,
        )


    except Exception as exc:
        logger.error("Answer generation failed: %s", exc, exc_info=True)
        # Do not expose API keys, provider responses, or internal details.
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed. Check the LLM configuration.",
        ) from exc


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
            detail="Could not read the PDF.",
        ) from exc
    finally:
        await file.close()


@router.post("/api/v1/documents/chunk")
async def chunk_uploaded_document(file: UploadFile = File(...)):
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
            detail="Could not process the PDF.",
        ) from exc
    finally:
        await file.close()


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

    sha256 = None
    claimed = False

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Identify the PDF by content, not by filename.
        sha256 = hashlib.sha256(content).hexdigest()
        stored_path = DOCUMENTS_DIR / f"{sha256}.pdf"

        state, record = claim_document(
            sha256=sha256,
            filename=file.filename,
            stored_path=str(stored_path),
        )

        if state == "already_uploaded":
            return {
                "status": "already_uploaded",
                "message": "This PDF has already been uploaded and indexed.",
                "document_id": record["document_id"],
                "filename": record["original_filename"],
            }

        if state == "already_processing":
            return {
                "status": "processing",
                "message": "This PDF is already being processed.",
                "document_id": record["document_id"],
            }

        claimed = True

        # Persist the original PDF before processing it.
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        stored_path.write_bytes(content)

        extracted = extract_pdf(content, file.filename)
        chunks = chunk_document(extracted)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text was found in the PDF.",
            )

        stored_count = vector_store.add_chunks(chunks)

        update_document_status(sha256, "completed")

        return {
            "status": "uploaded",
            "message": "PDF saved and indexed successfully.",
            "document_id": record["document_id"],
            "filename": file.filename,
            "chunk_count": len(chunks),
            "indexed_count": stored_count,
        }

    except HTTPException as exc:
        if claimed and sha256:
            update_document_status(
                sha256,
                "failed",
                str(exc.detail),
            )
        raise

    except ValueError as exc:
        if claimed and sha256:
            update_document_status(
                sha256,
                "failed",
                str(exc),
            )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Document indexing failed")

        if claimed and sha256:
            update_document_status(
                sha256,
                "failed",
                "Document indexing failed. Check the server logs.",
            )

        raise HTTPException(
            status_code=422,
            detail="Could not index the PDF. Check the server logs.",
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
