
import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient

CHUNKS_DIR = Path("D:/RAGA/sop_qa_rag/db/chunked_docs")

COLLECTION = "sop_chunks"


def main():
    files = list(CHUNKS_DIR.glob("*/chunks.json"))
    all_chunks = []
    empty_chunks = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            text = record.get("text", "").strip()

            if not text:
                empty_chunks += 1
                continue

            chunk_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            all_chunks.append((path, record, chunk_hash))

    unique_hashes = {item[2] for item in all_chunks}

    print("=== CHUNK AUDIT ===")
    print(f"Chunk files: {len(files)}")
    print(f"Non-empty chunks: {len(all_chunks)}")
    print(f"Unique chunk texts: {len(unique_hashes)}")
    print(f"Duplicate chunk occurrences: {len(all_chunks) - len(unique_hashes)}")
    print(f"Empty chunks: {empty_chunks}")

    if not files:
        print("No chunk files found. Check CHUNKS_DIR.")
        return

    client = QdrantClient(url="http://localhost:6333")

    try:
        if not client.collection_exists(COLLECTION):
            print(f"\nCollection '{COLLECTION}' does not exist.")
            return

        info = client.get_collection(COLLECTION)
        stored_count = client.count(
            collection_name=COLLECTION,
            exact=True,
        ).count

        print("\n=== QDRANT AUDIT ===")
        print(f"Stored points: {stored_count}")
        print(f"Vector dimension: {info.config.params.vectors.size}")
        print(f"Distance metric: {info.config.params.vectors.distance}")

        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=5,
            with_payload=True,
            with_vectors=True,
        )

        print("\n=== SAMPLE POINTS ===")
        for point in points:
            payload = point.payload or {}
            vector = point.vector

            print(f"\nPoint ID: {point.id}")
            print(f"Vector dimension: {len(vector) if vector else 0}")
            print(f"Text present: {bool(payload.get('text'))}")
            print(f"Source: {payload.get('source_document')}")
            print(f"Chunk index: {payload.get('chunk_index')}")
            print(f"Hash present: {bool(payload.get('chunk_hash'))}")

        print("\n=== BASIC VALIDATION ===")
        print(
            "Unique chunk count matches stored count:",
            len(unique_hashes) == stored_count,
        )
        print(
            "Expected MiniLM vector dimension (384):",
            info.config.params.vectors.size == 384,
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()
