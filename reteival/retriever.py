from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "sop_chunks"


def query_embedding(query: str):
    query_embeddings = embedding_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return query_embeddings.tolist()

def context_retriever(
    query: str,
    top_k: int = 6
) -> list[dict]:

    dense_vector = query_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,

        prefetch=[
            # 1. Dense semantic search
            models.Prefetch(
                query=dense_vector,
                limit=20,
                params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                ),
            ),

            # 2. BM25 full-text search
            models.Prefetch(
                query=models.TextQuery(text=query),
                using="text",
                limit=20,
            ),
        ],

        # 3. Reciprocal Rank Fusion
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),

        limit=top_k,
        with_payload=True,
    ) 

    print("Number of points returned:", len(results.points))

    for i, point in enumerate(results.points):
        print(f"\nResult {i + 1}")
        print("Point ID:", point.id)
        print("Score:", point.score)
        print("Payload keys:", point.payload.keys() if point.payload else None)
        print("Source:", point.payload.get("source_document") if point.payload else None)

    chunks = []

    for point in results.points:
        if point.payload and "text" in point.payload:
            chunks.append({
                "text": point.payload["text"],
                "source": point.payload.get(
                    "source_document", "unknown"
                ),
                "score": point.score,
            })

    return chunks
