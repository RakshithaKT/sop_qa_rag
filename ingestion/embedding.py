# import hashlib
# import json
# import logging
# import uuid
# from pathlib import Path

# from qdrant_client import QdrantClient, models
# from sentence_transformers import SentenceTransformer

# BASE_DIR = Path(__file__).resolve().parent.parent
# CHUNKS_DIR = BASE_DIR / "db" / "chunked_docs"

# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# COLLECTION_NAME = "sop_chunks"
# BATCH_SIZE = 64

# log = logging.getLogger(__name__)

# _model = None
# _client = None


# def get_embedding_model():
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(MODEL_NAME)
#     return _model


# def get_qdrant_client():
#     global _client
#     if _client is None:
#         _client = QdrantClient(url="http://localhost:6333")
#     return _client


# def ensure_collection():
#     client = get_qdrant_client()
#     model = get_embedding_model()
#     if not client.collection_exists(COLLECTION_NAME):
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=models.VectorParams(
#                 size=model.get_sentence_embedding_dimension(),
#                 distance=models.Distance.COSINE,
#             ),
#         )


# def embed_and_store_chunks(records: list[dict]) -> int:
#     """
#     Takes a list of chunk dicts (with 'text', 'source_document', 'document_id', 'chunk_index'),
#     checks for duplicates against Qdrant, embeds new ones, and upserts them into Qdrant.
#     Returns number of newly stored chunks.
#     """
#     if not records:
#         return 0

#     ensure_collection()
#     client = get_qdrant_client()
#     model = get_embedding_model()

#     # Query Qdrant for existing chunk hashes
#     existing_hashes = set()
#     offset = None
#     while True:
#         points, offset = client.scroll(
#             collection_name=COLLECTION_NAME,
#             limit=1000,
#             offset=offset,
#             with_payload=["chunk_hash"],
#             with_vectors=False,
#         )
#         for point in points:
#             if point.payload and point.payload.get("chunk_hash"):
#                 existing_hashes.add(point.payload["chunk_hash"])
#         if offset is None:
#             break

#     new_chunks = {}
#     for record in records:
#         text = record.get("text", "").strip()
#         if not text:
#             continue
#         chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
#         if chunk_hash in existing_hashes or chunk_hash in new_chunks:
#             continue
#         new_chunks[chunk_hash] = {
#             "text": text,
#             "source_document": record.get("source_document", "unknown"),
#             "document_id": record.get("document_id", "unknown"),
#             "chunk_index": record.get("chunk_index"),
#         }

#     if not new_chunks:
#         log.info("No new chunks to embed (all already exist in Qdrant).")
#         return 0

#     items = list(new_chunks.items())
#     for start in range(0, len(items), BATCH_SIZE):
#         batch = items[start : start + BATCH_SIZE]
#         texts = [rec["text"] for _, rec in batch]

#         embeddings = model.encode(
#             texts,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         )

#         points = [
#             models.PointStruct(
#                 id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_hash)),
#                 vector=embedding.tolist(),
#                 payload={
#                     "chunk_hash": chunk_hash,
#                     **rec,
#                 },
#             )
#             for (chunk_hash, rec), embedding in zip(batch, embeddings)
#         ]

#         client.upsert(
#             collection_name=COLLECTION_NAME,
#             points=points,
#             wait=True,
#         )
#         log.info("Stored %d/%d chunks in Qdrant", min(start + len(batch), len(items)), len(items))

#     return len(items)


# def main():
#     ensure_collection()
#     all_records = []
#     for file_path in CHUNKS_DIR.glob("*/chunks.json"):
#         with file_path.open("r", encoding="utf-8") as file:
#             all_records.extend(json.load(file))

#     added = embed_and_store_chunks(all_records)
#     print(f"Finished embedding scan. Added {added} new chunks.")


# if __name__ == "__main__":
#     main()





import hashlib
import json
import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "db" / "chunked_docs"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "sop_chunks"
BATCH_SIZE = 64

log = logging.getLogger(__name__)

_model = None
_client = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_qdrant_client():
    global _client
    if _client is None:
        _client = QdrantClient(url="http://localhost:6333")
    return _client


def ensure_collection():
    client = get_qdrant_client()
    model = get_embedding_model()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )
        log.info("Created collection: %s", COLLECTION_NAME)

    # Create BM25 full-text index on the existing text payload.
    # This does NOT delete or recreate existing vectors.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            scoring=True,
        ),
        wait=True,
    )

    log.info("BM25 text index ensured on 'text' field.")


def embed_and_store_chunks(records: list[dict]) -> int:
    """
    Takes a list of chunk dicts containing:
    text, source_document, document_id, chunk_index.

    Checks for duplicate chunks, embeds new chunks,
    and upserts them into Qdrant.

    Returns the number of newly stored chunks.
    """
    if not records:
        return 0

    ensure_collection()

    client = get_qdrant_client()
    model = get_embedding_model()

    # Retrieve existing chunk hashes from Qdrant.
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

    # Identify new chunks.
    new_chunks = {}

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
            "source_document": record.get("source_document", "unknown"),
            "document_id": record.get("document_id", "unknown"),
            "chunk_index": record.get("chunk_index"),
        }

    if not new_chunks:
        log.info("No new chunks to embed.")
        return 0

    items = list(new_chunks.items())

    # Batch embedding and upserting.
    for start in range(0, len(items), BATCH_SIZE):

        batch = items[start:start + BATCH_SIZE]

        texts = [
            rec["text"] for _, rec in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            models.PointStruct(
                id=str(
                    uuid.uuid5(uuid.NAMESPACE_URL, chunk_hash)
                ),
                vector=embedding.tolist(),
                payload={
                    "chunk_hash": chunk_hash,
                    **rec,
                },
            )
            for (chunk_hash, rec), embedding
            in zip(batch, embeddings)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        log.info(
            "Stored %d/%d chunks in Qdrant",
            min(start + len(batch), len(items)),
            len(items),
        )

    return len(items)


def main():
    ensure_collection()

    all_records = []

    for file_path in CHUNKS_DIR.glob("*/chunks.json"):
        with file_path.open("r", encoding="utf-8") as file:
            all_records.extend(json.load(file))

    added = embed_and_store_chunks(all_records)

    print(
        f"Finished embedding scan. Added {added} new chunks."
    )


if __name__ == "__main__":
    main()
