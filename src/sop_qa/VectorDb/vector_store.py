
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from sop_qa.embedding.embeddings import embed_texts


class VectorStore:
    def __init__(
        self,
        path: str = "data/qdrant",
        collection_name: str = "sop_chunks",
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed and store chunks with their source metadata."""
        if not chunks:
            return 0

        vectors = embed_texts([chunk["text"] for chunk in chunks])
        points = []

        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid5(NAMESPACE_URL, chunk["chunk_id"])
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search(
        self,
        question: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve the most similar chunks for a question."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if limit < 1:
            raise ValueError("limit must be positive.")

        query_vector = embed_texts([question])[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "score": result.score,
                **(result.payload or {}),
            }
            for result in results
        ]

    def close(self) -> None:
        self.client.close()
