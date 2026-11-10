# from db import document_registry 

from docling.document_converter import DocumentConverter

source = "D:/RAGA/sop_qa_rag/db/raw_docs/researchP1_table.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]" 

import json
import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


import hashlib
import json
import logging
import time
from pathlib import Path

from docling.backend.docling_parse_backend import (
    ThreadedDoclingParseDocumentBackend,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


# Configure paths
INPUT_DIR = Path(r"D:/RAGA/sop_qa_rag/db/raw_docs")
OUTPUT_DIR = Path(r"D:/RAGA/sop_qa_rag/db/parsed_docs")
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
