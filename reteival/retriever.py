from sentence_transformers import SentenceTransformer 
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333") 
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def query_embedding(query): 

    query_embeddings = embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )  
    return query_embeddings

def context_retriever(query: str, top_k: int = 3)-> list[dict]:
#Cosine similarity search 
    results = client.query_points(
        collection_name="sop_chunks",
        query= query_embedding(query),
        # query_filter=models.Filter(
        #     # must=[
        #     #     models.FieldCondition(
        #     #         key="city",
        #     #         match=models.MatchValue(
        #     #             value="London",
        #     #         ),
        #     #     )
        #     # ]
        # ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=top_k, 
        ) 
    chunks = []
    for point in results.points:
        if point.payload and "text" in point.payload: 
            chunks.append({ 
                "text": point.payload["text"], 
                "source": point.payload.get("source_document", "unknown"),
                "score": point.score
            })


    return chunks