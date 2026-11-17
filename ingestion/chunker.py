import json
import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "db" / "chunked_docs"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64

log = logging.getLogger(__name__)

# Lazily initialized chunker to avoid reloading tokenizer on every request
_tokenizer = None
_chunker = None


def get_chunker():
    global _tokenizer, _chunker
    if _chunker is None:
        _tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
            max_tokens=MAX_TOKENS,
        )
        _chunker = HybridChunker(tokenizer=_tokenizer, merge_peers=True)
    return _chunker


def chunk_document(doc_path: Path) -> tuple[Path, list[dict]]:
    """
    Chunks a parsed Docling JSON document and saves chunks to db/chunked_docs/<doc_stem>/chunks.json.
    Returns (chunks_file_path, records)
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = CHUNKS_DIR / doc_path.stem
    chunks_file = output_dir / "chunks.json"

    # If chunks already exist, we can load and return them
    if chunks_file.is_file():
        try:
            with chunks_file.open("r", encoding="utf-8") as f:
                records = json.load(f)
            log.info("Chunks already exist for %s: %d chunks loaded.", doc_path.name, len(records))
            return chunks_file, records
        except Exception:
            pass

    doc = DoclingDocument.load_from_json(doc_path)
    chunker = get_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "chunk_index": i,
            "text": chunker.contextualize(chunk=chunk),
            "source_document": doc_path.name,
            "document_id": doc_path.stem,
        }
        for i, chunk in enumerate(chunks)
    ]

    with chunks_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    log.info("Saved %d chunks to %s", len(records), chunks_file)
    return chunks_file, records


def main():
    doc_path = (
        BASE_DIR
        / "db"
        / "parsed_docs"
        / "a9b78223f7f2e3fb06defcebe9585ed6ea867aa86b823f2ac0c59d911a1b859c.json"
    )
    if doc_path.exists():
        chunk_document(doc_path)
    else:
        print(f"Sample doc not found at {doc_path}")


if __name__ == "__main__":
    main()
