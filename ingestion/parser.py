import hashlib
import json
import logging
import time
from pathlib import Path

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "db" / "raw_docs"
OUTPUT_DIR = BASE_DIR / "db" / "parsed_docs"
MANIFEST_PATH = OUTPUT_DIR / "parsed_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def calculate_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_pdf(pdf_path: Path) -> tuple[Path, str, bool]:
    """
    Parse a single PDF using Docling if not already parsed.
    Returns (output_json_path, file_hash, is_newly_parsed)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            with MANIFEST_PATH.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except Exception:
            manifest = {}

    file_hash = calculate_hash(pdf_path)
    output_path = OUTPUT_DIR / f"{file_hash}.json"

    if file_hash in manifest and output_path.is_file():
        log.info("Already parsed: %s (%s)", pdf_path.name, file_hash)
        return output_path, file_hash, False

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ThreadedDoclingParseDocumentBackend,
            )
        }
    )

    result = converter.convert(pdf_path, raises_on_error=True)
    if result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        result.document.save_as_json(output_path)
        manifest[file_hash] = output_path.name
        temp_path = OUTPUT_DIR / "parsed_manifest.tmp"
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
        temp_path.replace(MANIFEST_PATH)
        log.info("Successfully parsed: %s -> %s", pdf_path.name, output_path.name)
        return output_path, file_hash, True
    else:
        raise RuntimeError(f"Conversion failed for {pdf_path.name}")


def main():
    start_time = time.perf_counter()

    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load previously parsed hashes.
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError("Invalid manifest format.")
    else:
        manifest = {}

    pdf_files = []
    skipped_non_pdf = 0
    skipped_parsed = 0

    # Find PDFs that have not been parsed successfully.
    for path in sorted(INPUT_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() != ".pdf":
            log.info("Skipping non-PDF: %s", path.name)
            skipped_non_pdf += 1
            continue

        file_hash = calculate_hash(path)
        output_path = OUTPUT_DIR / f"{file_hash}.md"

        if file_hash in manifest and output_path.is_file():
            log.info("Already parsed: %s", path.name)
            skipped_parsed += 1
            continue

        pdf_files.append((path, file_hash))

    if not pdf_files:
        log.info("No new PDFs to parse.")
        log.info(
            "Skipped: %d already parsed, %d non-PDF files.",
            skipped_parsed,
            skipped_non_pdf,
        )
        return

    # Configure Docling.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ThreadedDoclingParseDocumentBackend,
            )
        }
    )

    successful = 0
    partial = 0
    failed = 0

    # Convert and export one PDF at a time.
    for path, file_hash in pdf_files:
        try:
            result = converter.convert(path, raises_on_error=False)

            if result.status == ConversionStatus.SUCCESS:
                output_path = OUTPUT_DIR / f"{file_hash}.json"

                result.document.save_as_json(output_path)

                # Record only after successful Markdown export.
                manifest[file_hash] = output_path.name

                # Persist progress after each successful PDF.
                temp_path = OUTPUT_DIR / "parsed_manifest.tmp"
                with temp_path.open("w", encoding="utf-8") as file:
                    json.dump(manifest, file, indent=2)
                    file.write("\n")
                temp_path.replace(MANIFEST_PATH)

                log.info("Parsed: %s -> %s", path.name, output_path.name)
                successful += 1

            elif result.status == ConversionStatus.PARTIAL_SUCCESS:
                partial += 1
                log.warning("Partially converted: %s", path.name)

                for error in result.errors:
                    log.warning("%s", error.error_message)

            else:
                failed += 1
                log.error("Conversion failed: %s", path.name)

                for error in result.errors:
                    log.error("%s", error.error_message)

        except Exception:
            failed += 1
            log.exception("Error processing: %s", path.name)

    log.info(
        "Finished in %.2f seconds. Successful: %d, partial: %d, "
        "failed: %d, already parsed: %d, non-PDF: %d.",
        time.perf_counter() - start_time,
        successful,
        partial,
        failed,
        skipped_parsed,
        skipped_non_pdf,
    )

    if failed or partial:
        raise RuntimeError(
            f"Parsing finished with {failed} failures "
            f"and {partial} partial conversions."
        )


if __name__ == "__main__":
    main()


