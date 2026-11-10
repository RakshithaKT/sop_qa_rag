
import hashlib
import json 
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("D:/RAGA/sop_qa_rag/db/chunked_docs")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "sop_chunks"
BATCH_SIZE = 64


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

    # Collect hashes of chunks already stored in Qdrant.
    existing_hashes = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["chunk_hash"],
            with_vectors=False,
        )

        for point in points:
            if point.payload and point.payload.get("chunk_hash"):
                existing_hashes.add(point.payload["chunk_hash"])

        if offset is None:
            break

    # Load chunks and skip duplicates.
    new_chunks = {}

    for file_path in CHUNKS_DIR.glob("*/chunks.json"):
        with file_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        for record in records:
            text = record.get("text", "").strip()
            if not text:
                continue

            chunk_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if chunk_hash in existing_hashes or chunk_hash in new_chunks:
                continue

            new_chunks[chunk_hash] = {
                "text": text,
                "source_document": record.get(
                    "source_document", file_path.parent.name
                ),
                "document_id": record.get(
                    "document_id", file_path.parent.name
                ),
                "chunk_index": record.get("chunk_index"),
            }

    if not new_chunks:
        print("No new chunks to embed.")
        client.close()
        return

    items = list(new_chunks.items())

    # Embed and store in batches.
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        texts = [record["text"] for _, record in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            models.PointStruct(
                id= str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_hash)),
                vector=embedding.tolist(),
                payload={
                    "chunk_hash": chunk_hash,
                    **record,
                },
            )
            for (chunk_hash, record), embedding in zip(batch, embeddings)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Stored {min(start + len(batch), len(items))}/{len(items)} chunks")

    print(f"Finished. Added {len(items)} new chunks.")
    client.close()


if __name__ == "__main__":
    main()
