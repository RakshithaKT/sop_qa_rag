
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert text strings into normalized embedding vectors."""
    if not texts:
        return []

    model = get_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    """Generate an embedding for a search query."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    return embed_texts([question])[0]
