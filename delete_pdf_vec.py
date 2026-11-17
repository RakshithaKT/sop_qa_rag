import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
RAW_DOCS_DIR = DB_DIR / "raw_docs"
TO_DELETE_DIR = DB_DIR / "to_delete_docs"
PARSED_DOCS_DIR = DB_DIR / "parsed_docs"
PARSED_MANIFEST_PATH = PARSED_DOCS_DIR / "parsed_manifest.json"
CHUNKED_DOCS_DIR = DB_DIR / "chunked_docs"

COLLECTION_NAME = "sop_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("delete_pdf_vec")


def calculate_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def delete_parsed_data(doc_hash: str) -> dict:
    """Delete parsed JSON and its record in parsed_manifest.json."""
    details = {"deleted_json": False, "manifest_updated": False}

    parsed_file = PARSED_DOCS_DIR / f"{doc_hash}.json"
    if parsed_file.is_file():
        parsed_file.unlink()
        details["deleted_json"] = True
        log.info("  [1/4] Deleted parsed JSON: %s", parsed_file.name)
    else:
        log.info("  [1/4] Parsed JSON not found: %s.json (skipping)", doc_hash)

    if PARSED_MANIFEST_PATH.is_file():
        try:
            with PARSED_MANIFEST_PATH.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            if doc_hash in manifest:
                del manifest[doc_hash]
                with PARSED_MANIFEST_PATH.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                details["manifest_updated"] = True
                log.info("  [1/4] Removed hash %s from parsed_manifest.json", doc_hash)
        except Exception as e:
            log.warning("  [1/4] Error updating parsed_manifest.json: %s", e)

    return details


def delete_chunked_data(doc_hash: str) -> tuple[dict, list[str]]:
    """
    Extract point IDs / chunk hashes from chunks.json for Qdrant purging,
    then delete the chunks directory.
    """
    details = {"deleted_chunk_dir": False, "chunk_count": 0}
    point_ids = []

    chunk_dir = CHUNKED_DOCS_DIR / doc_hash
    chunks_file = chunk_dir / "chunks.json"

    if chunks_file.is_file():
        try:
            with chunks_file.open("r", encoding="utf-8") as f:
                chunks = json.load(f)
            details["chunk_count"] = len(chunks)
            for chk in chunks:
                chk_text = chk.get("text", "").strip()
                if chk_text:
                    chk_hash = hashlib.sha256(chk_text.encode("utf-8")).hexdigest()
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chk_hash))
                    point_ids.append(point_id)
        except Exception as e:
            log.warning("  [2/4] Could not read chunks.json for ID extraction: %s", e)

    if chunk_dir.is_dir():
        shutil.rmtree(chunk_dir, ignore_errors=True)
        details["deleted_chunk_dir"] = True
        log.info("  [2/4] Deleted chunk directory: %s", chunk_dir.name)
    else:
        log.info("  [2/4] Chunk directory not found: %s (skipping)", doc_hash)

    return details, point_ids


def delete_qdrant_embeddings(doc_hash: str, filename: str, point_ids: list[str]) -> dict:
    """Delete all corresponding points from Qdrant vector DB."""
    details = {"qdrant_connected": False, "deleted": False, "message": ""}
    try:
        from qdrant_client import QdrantClient, models

        client = QdrantClient(url=QDRANT_URL, timeout=5)
        details["qdrant_connected"] = True

        filter_conditions = [
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=doc_hash),
            ),
            models.FieldCondition(
                key="source_document",
                match=models.MatchValue(value=f"{doc_hash}.json"),
            ),
            models.FieldCondition(
                key="source_document",
                match=models.MatchValue(value=filename),
            ),
        ]

        # Delete by payload metadata filter
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(should=filter_conditions)
            ),
            wait=True,
        )

        # Fallback/Fail-safe: Delete by explicit UUID5 point IDs if known
        if point_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )

        details["deleted"] = True
        details["message"] = f"Deleted embeddings from collection '{COLLECTION_NAME}'"
        log.info("  [3/4] Successfully deleted vector embeddings from Qdrant")

    except ImportError:
        details["message"] = "qdrant_client package not installed"
        log.warning("  [3/4] qdrant_client package not found. Skipped vector deletion.")
    except Exception as e:
        details["message"] = f"Could not connect or delete from Qdrant: {e}"
        log.warning("  [3/4] Qdrant deletion skipped or failed: %s", e)

    return details


def delete_raw_pdf(doc_hash: str, filename: str, source_path: Path | None = None) -> dict:
    """Delete PDF from raw_docs and/or to_delete_docs."""
    details = {"deleted_from_raw": False, "deleted_from_queue": False}

    # Delete from raw_docs by filename
    raw_file = RAW_DOCS_DIR / filename
    if raw_file.is_file():
        raw_file.unlink()
        details["deleted_from_raw"] = True
        log.info("  [4/4] Deleted raw PDF: %s", raw_file.name)

    # Search raw_docs for any file that matches this hash even if named differently
    if not details["deleted_from_raw"]:
        for candidate in RAW_DOCS_DIR.glob("*.pdf"):
            try:
                if calculate_hash(candidate) == doc_hash:
                    candidate.unlink()
                    details["deleted_from_raw"] = True
                    log.info("  [4/4] Deleted raw PDF by hash match: %s", candidate.name)
                    break
            except Exception:
                continue

    # Delete from to_delete_docs if located there
    if source_path and source_path.is_file() and source_path.parent == TO_DELETE_DIR:
        source_path.unlink()
        details["deleted_from_queue"] = True
        log.info("  [4/4] Cleaned up PDF from to_delete_docs queue: %s", source_path.name)
    else:
        queued_file = TO_DELETE_DIR / filename
        if queued_file.is_file():
            queued_file.unlink()
            details["deleted_from_queue"] = True
            log.info("  [4/4] Cleaned up PDF from to_delete_docs queue: %s", queued_file.name)

    return details


def purge_document(pdf_path: Path) -> dict:
    """Perform complete step-by-step deletion for a single PDF."""
    if not pdf_path.exists():
        return {"error": f"File does not exist: {pdf_path}"}

    filename = pdf_path.name
    doc_hash = calculate_hash(pdf_path)

    log.info("==================================================")
    log.info("Purging Document: %s", filename)
    log.info("SHA-256 Hash:     %s", doc_hash)
    log.info("==================================================")

    # Step 1: Parsed Doc & Manifest
    parsed_res = delete_parsed_data(doc_hash)

    # Step 2: Chunked Docs (and capture point IDs)
    chunk_res, point_ids = delete_chunked_data(doc_hash)

    # Step 3: Embeddings from Qdrant Vector DB
    qdrant_res = delete_qdrant_embeddings(doc_hash, filename, point_ids)

    # Step 4: Raw PDF and Queue files
    raw_res = delete_raw_pdf(doc_hash, filename, source_path=pdf_path)

    log.info("Finished purge for: %s\n", filename)

    return {
        "filename": filename,
        "hash": doc_hash,
        "parsed_data": parsed_res,
        "chunked_data": chunk_res,
        "qdrant": qdrant_res,
        "raw_pdf": raw_res,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Completely delete documents across raw, parsed, chunked, and vector DB."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional: specific PDF filenames or paths to delete. If omitted, scans db/to_delete_docs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON results.",
    )
    args = parser.parse_args()

    targets: list[Path] = []

    if args.files:
        for f in args.files:
            p = Path(f)
            if not p.is_absolute():
                # Check in to_delete_docs first, then raw_docs, then current dir
                if (TO_DELETE_DIR / f).is_file():
                    p = TO_DELETE_DIR / f
                elif (RAW_DOCS_DIR / f).is_file():
                    p = RAW_DOCS_DIR / f
                else:
                    p = Path.cwd() / f
            if p.is_file():
                targets.append(p)
            else:
                log.warning("Target file not found: %s", f)
    else:
        # Default: process all PDFs in db/to_delete_docs
        targets = list(TO_DELETE_DIR.glob("*.pdf"))

    if not targets:
        msg = f"No PDF files found to delete. Place PDFs into '{TO_DELETE_DIR}' or specify filenames as arguments."
        log.info(msg)
        if args.json:
            print(json.dumps({"status": "no_action", "message": msg, "results": []}))
        return

    log.info("Found %d document(s) targeted for complete deletion.", len(targets))

    results = []
    for target in targets:
        res = purge_document(target)
        results.append(res)

    if args.json:
        print(json.dumps({"status": "success", "results": results}, indent=2))
    else:
        print("\n=== PURGE SUMMARY ===")
        for r in results:
            print(f"- {r.get('filename')} (Hash: {r.get('hash', 'unknown')[:12]}...): Complete Purge Finished")


if __name__ == "__main__":
    main()
