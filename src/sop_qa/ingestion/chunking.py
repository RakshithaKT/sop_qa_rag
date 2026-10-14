
def chunk_document(
    document: dict,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """Split extracted document pages into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    filename = document["filename"]
    chunk_number = 0

    for page in document["pages"]:
        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                # Prefer a paragraph, sentence, or word boundary.
                minimum_break = start + chunk_size // 2

                for separator in ("\n\n", "\n", ". ", " "):
                    position = text.rfind(
                        separator, minimum_break, end
                    )
                    if position != -1:
                        end = position + len(separator)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_number += 1
                chunks.append(
                    {
                        "chunk_id": f"{filename}-chunk-{chunk_number}",
                        "filename": filename,
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            # Move forward while retaining some preceding context.
            next_start = max(start + 1, end - overlap)

            # Avoid repeatedly starting inside whitespace.
            while (
                next_start < end
                and text[next_start].isspace()
            ):
                next_start += 1

            start = next_start

    return chunks
