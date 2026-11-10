
import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64

DOC_PATH = Path(
    "D:/RAGA/sop_qa_rag/db/parsed_docs/"
    "a9b78223f7f2e3fb06defcebe9585ed6ea867aa86b823f2ac0c59d911a1b859c.json"
)
CHUNKS_DIR = Path("D:/RAGA/sop_qa_rag/db/chunked_docs")


doc = DoclingDocument.load_from_json(DOC_PATH)

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
chunks = list(chunker.chunk(dl_doc=doc))

output_dir = CHUNKS_DIR / DOC_PATH.stem
output_dir.mkdir(parents=True, exist_ok=True)

records = [
    {
        "chunk_index": i,
        "text": chunker.contextualize(chunk=chunk),
        "source_document": DOC_PATH.name,
    }
    for i, chunk in enumerate(chunks)
]

with (output_dir / "chunks.json").open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} chunks to {output_dir / 'chunks.json'}")
