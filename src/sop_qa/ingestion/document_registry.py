import sqlite3
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "raw_docs.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_registry() -> None:
    """Create the document registry table if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_registry (
                document_id TEXT NOT NULL UNIQUE,
                sha256 TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('processing', 'completed', 'failed')),
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def claim_document(
    sha256: str,
    filename: str,
    stored_path: str,
) -> tuple[str, dict]:
    """
    Atomically claim a document for indexing.

    Returns:
        ("claimed", record) for a new or retryable document.
        ("already_uploaded", record) if indexing already completed.
        ("already_processing", record) if another request is processing it.
    """
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            "SELECT * FROM document_registry WHERE sha256 = ?",
            (sha256,),
        ).fetchone()

        if row and row["status"] == "completed":
            return "already_uploaded", dict(row)

        if row and row["status"] == "processing":
            return "already_processing", dict(row)

        if row and row["status"] == "failed":
            connection.execute(
                """
                UPDATE document_registry
                SET original_filename = ?,
                    stored_path = ?,
                    status = 'processing',
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE sha256 = ?
                """,
                (filename, stored_path, sha256),
            )
            record = connection.execute(
                "SELECT * FROM document_registry WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
            return "claimed", dict(record)

        document_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO document_registry (
                document_id, sha256, original_filename,
                stored_path, status
            )
            VALUES (?, ?, ?, ?, 'processing')
            """,
            (document_id, sha256, filename, stored_path),
        )

        record = connection.execute(
            "SELECT * FROM document_registry WHERE sha256 = ?",
            (sha256,),
        ).fetchone()

        return "claimed", dict(record)


def update_document_status(
    sha256: str,
    status: str,
    error_message: str | None = None,
) -> None:
    if status not in {"processing", "completed", "failed"}:
        raise ValueError(f"Invalid document status: {status}")

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE document_registry
            SET status = ?,
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE sha256 = ?
            """,
            (status, error_message, sha256),
        )