import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ingestion.chunker import chunk_document
from ingestion.embedding import embed_and_store_chunks
from ingestion.parser import calculate_hash, parse_pdf
from reteival.llm import answer_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DOCS_DIR = BASE_DIR / "db" / "raw_docs"
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SOP QA RAG API")


@app.get("/", response_class=HTMLResponse)
def index():
    """Simple UI with file browsing & upload, plus query interaction."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SOP QA RAG Assistant</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f7f9fa; margin: 0; padding: 30px; color: #333; }
            .container { max-width: 760px; margin: 0 auto; background: #fff; padding: 25px 30px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
            h1 { font-size: 1.6rem; color: #1a202c; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
            h2 { font-size: 1.2rem; color: #2d3748; margin-top: 25px; margin-bottom: 12px; }
            .section { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 20px; }
            input[type="file"] { display: block; margin-bottom: 12px; font-size: 14px; }
            input[type="text"] { width: 100%; box-sizing: border-box; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; margin-bottom: 12px; }
            button { background-color: #2563eb; color: white; border: none; border-radius: 6px; padding: 10px 18px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.2s; }
            button:hover { background-color: #1d4ed8; }
            button:disabled { background-color: #94a3b8; cursor: not-allowed; }
            .status { margin-top: 12px; font-size: 14px; white-space: pre-wrap; font-family: monospace; background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 6px; display: none; }
            .sources { margin-top: 10px; font-size: 13px; color: #64748b; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; background: #e0e7ff; color: #3730a3; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
                <h1 style="margin: 0; border-bottom: none; padding-bottom: 0;">📑 SOP QA RAG Pipeline</h1>
                <a href="/delete_pdf_vec" style="color: #dc2626; text-decoration: none; font-weight: 500; font-size: 14px; padding: 6px 12px; border: 1px solid #fca5a5; border-radius: 6px; background: #fef2f2; transition: background 0.2s;">🗑️ Delete PDFs & Vectors</a>
            </div>
            
            <div class="section">
                <h2>1. Upload SOP PDF Document</h2>
                <form id="uploadForm">
                    <input type="file" id="pdfFile" accept=".pdf" required />
                    <button type="submit" id="uploadBtn">Browse Device & Upload</button>
                </form>
                <div id="uploadStatus" class="status"></div>
            </div>

            <div class="section">
                <h2>2. Ask a Question</h2>
                <form id="queryForm">
                    <input type="text" id="userQuery" placeholder="Ask question based on uploaded SOPs..." required />
                    <button type="submit" id="queryBtn">Submit Query</button>
                </form>
                <div id="queryStatus" class="status"></div>
            </div>
        </div>

        <script>
            const uploadForm = document.getElementById('uploadForm');
            const uploadStatus = document.getElementById('uploadStatus');
            const uploadBtn = document.getElementById('uploadBtn');

            uploadForm.onsubmit = async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('pdfFile');
                if (!fileInput.files.length) return;

                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('new_pdf', file);

                uploadBtn.disabled = true;
                uploadStatus.style.display = 'block';
                uploadStatus.innerText = 'Processing: Uploading, verifying hash, parsing with Docling, chunking & storing into Qdrant vector DB... Please wait.';

                try {
                    const res = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    uploadStatus.innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    uploadStatus.innerText = 'Error: ' + err.message;
                } finally {
                    uploadBtn.disabled = false;
                }
            };

            const queryForm = document.getElementById('queryForm');
            const queryStatus = document.getElementById('queryStatus');
            const queryBtn = document.getElementById('queryBtn');

            queryForm.onsubmit = async (e) => {
                e.preventDefault();
                const query = document.getElementById('userQuery').value;
                if (!query) return;

                queryBtn.disabled = true;
                queryStatus.style.display = 'block';
                queryStatus.innerText = 'Thinking and querying Qdrant vector database...';

                try {
                    const res = await fetch('/llm_endpoint', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await res.json();
                    queryStatus.innerText = (data.answer || JSON.stringify(data, null, 2)) + 
                        (data.sources && data.sources.length ? '\\n\\nSources: ' + data.sources.join(', ') : '') +
                        (data.retrieval_performed !== undefined ? '\\nRetrieval performed: ' + data.retrieval_performed : '');
                } catch (err) {
                    queryStatus.innerText = 'Error: ' + err.message;
                } finally {
                    queryBtn.disabled = false;
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload")
@app.post("/")
async def upload_doc(new_pdf: UploadFile = File(...)):
    """
    1. Read uploaded PDF bytes and compute SHA-256 hash.
    2. Save to db/raw_docs/ if not existing.
    3. Check if already parsed / indexed via SHA-256 hash.
    4. Automatically run parser -> chunker -> embedding to vector DB (Qdrant).
    """
    content = await new_pdf.read()
    file_hash = hashlib.sha256(content).hexdigest()
    filename = new_pdf.filename or f"{file_hash}.pdf"
    
    # Save raw file
    raw_pdf_path = RAW_DOCS_DIR / filename
    if not raw_pdf_path.exists():
        with raw_pdf_path.open("wb") as f:
            f.write(content)
        log.info("Saved new PDF to %s", raw_pdf_path)
    else:
        log.info("PDF with name %s already exists in raw_docs", filename)

    # 1. Parse PDF (Docling)
    parsed_json_path, _, is_newly_parsed = parse_pdf(raw_pdf_path)

    # 2. Chunking (HybridChunker)
    chunks_file_path, chunk_records = chunk_document(parsed_json_path)

    # 3. Embedding and storing into Qdrant
    new_points_added = embed_and_store_chunks(chunk_records)

    message = (
        "Document already indexed. No new chunks added."
        if not is_newly_parsed and new_points_added == 0
        else f"Document processed successfully. {new_points_added} new chunks stored in vector DB."
    )

    return {
        "status": "success",
        "message": message,
        "filename": filename,
        "sha256": file_hash,
        "already_existed": not is_newly_parsed,
        "total_chunks": len(chunk_records),
        "new_embeddings_stored": new_points_added,
        "parsed_json": parsed_json_path.name,
    }


from typing import Any
from fastapi import Body


@app.post("/llm_endpoint")
def user_query(query_data: Any = Body(...)):
    """
    Accept user query without strict schema validation (to be enhanced with Pydantic later).
    Supports either {"query": "..."} or raw string query.
    """
    if isinstance(query_data, dict):
        query_text = query_data.get("query", "")
    elif isinstance(query_data, str):
        query_text = query_data
    else:
        query_text = str(query_data)

    if not query_text.strip():
        return {"error": "Query cannot be empty"}

    result = answer_query(query_text)
    return result


@app.get("/delete_pdf_vec", response_class=HTMLResponse)
def delete_pdf_vec_page():
    """
    UI displaying all existing raw PDFs with selection checkboxes.
    Allows user to select files and trigger the deletion script.
    """
    pdf_files = sorted(RAW_DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    file_rows = ""
    for f in pdf_files:
        try:
            f_hash = calculate_hash(f)
            short_hash = f_hash[:16] + "..."
            size_kb = f"{f.stat().st_size / 1024:.1f} KB"
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            short_hash = "N/A"
            size_kb = "N/A"
            mtime = "N/A"
            f_hash = ""

        file_rows += f"""
        <tr>
            <td style="text-align: center;"><input type="checkbox" name="selected_pdf" value="{f.name}" class="file-chk"></td>
            <td style="font-weight: 500;">{f.name}</td>
            <td style="font-family: monospace; font-size: 12px; color: #475569;" title="{f_hash}">{short_hash}</td>
            <td style="color: #64748b; font-size: 13px;">{size_kb}</td>
            <td style="color: #64748b; font-size: 13px;">{mtime}</td>
        </tr>
        """

    if not file_rows:
        file_rows = """
        <tr>
            <td colspan="5" style="text-align: center; color: #64748b; padding: 20px;">
                No raw PDF documents found in <code>db/raw_docs/</code>.
            </td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delete PDFs & Vectors - SOP QA RAG</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f7f9fa; margin: 0; padding: 30px; color: #333; }}
            .container {{ max-width: 860px; margin: 0 auto; background: #fff; padding: 25px 30px; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 20px; }}
            h1 {{ font-size: 1.5rem; color: #1a202c; margin: 0; }}
            .nav-link {{ color: #2563eb; text-decoration: none; font-size: 14px; font-weight: 500; }}
            .nav-link:hover {{ text-decoration: underline; }}
            .table-container {{ margin-top: 15px; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ background: #f8fafc; padding: 12px 14px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }}
            td {{ padding: 10px 14px; border-bottom: 1px solid #f1f5f9; }}
            tr:hover {{ background-color: #f8fafc; }}
            .actions-bar {{ display: flex; justify-content: space-between; align-items: center; margin-top: 18px; }}
            .btn {{ padding: 9px 16px; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; border: none; transition: background 0.2s; }}
            .btn-danger {{ background-color: #dc2626; color: white; }}
            .btn-danger:hover {{ background-color: #b91c1c; }}
            .btn-danger:disabled {{ background-color: #f87171; cursor: not-allowed; }}
            .btn-secondary {{ background-color: #e2e8f0; color: #334155; }}
            .btn-secondary:hover {{ background-color: #cbd5e1; }}
            .status-box {{ margin-top: 20px; background: #0f172a; color: #f8fafc; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px; white-space: pre-wrap; display: none; max-height: 350px; overflow-y: auto; }}
            .badge-warn {{ background: #fef3c7; color: #92400e; padding: 10px 14px; border-radius: 6px; font-size: 13px; margin-bottom: 15px; border: 1px solid #fde68a; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🗑️ Delete PDFs & Vector Embeddings</h1>
                <a href="/" class="nav-link">← Back to Q&A Pipeline</a>
            </div>

            <div class="badge-warn">
                ⚠️ <strong>Note:</strong> Executing deletion will purge the document across all layers:
                the <strong>raw PDF</strong>, the <strong>parsed Docling JSON</strong> (and manifest), the <strong>chunk JSONs</strong>, and the <strong>vector embeddings in Qdrant</strong>.
            </div>

            <div class="actions-bar" style="margin-bottom: 12px; margin-top: 0;">
                <div>
                    <button type="button" class="btn btn-secondary" onclick="toggleSelect(true)">Select All</button>
                    <button type="button" class="btn btn-secondary" onclick="toggleSelect(false)">Deselect All</button>
                </div>
                <button type="button" id="deleteBtn" class="btn btn-danger" onclick="executeDeletion()">Execute Deletion</button>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40px; text-align: center;"><input type="checkbox" id="selectAllHeader" onclick="toggleSelect(this.checked)"></th>
                            <th>Filename</th>
                            <th>SHA-256 Hash</th>
                            <th>Size</th>
                            <th>Last Modified</th>
                        </tr>
                    </thead>
                    <tbody id="fileTableBody">
                        {file_rows}
                    </tbody>
                </table>
            </div>

            <div id="outputConsole" class="status-box"></div>
        </div>

        <script>
            function toggleSelect(state) {{
                document.querySelectorAll('.file-chk').forEach(chk => chk.checked = state);
                const headerChk = document.getElementById('selectAllHeader');
                if (headerChk) headerChk.checked = state;
            }}

            async function executeDeletion() {{
                const selected = Array.from(document.querySelectorAll('.file-chk:checked')).map(c => c.value);
                if (selected.length === 0) {{
                    alert("Please select at least one document to delete.");
                    return;
                }}

                const confirmMsg = `Are you sure you want to permanently delete ${{selected.length}} file(s) and all their embeddings from Qdrant?\\n\\n` + selected.join('\\n');
                if (!confirm(confirmMsg)) return;

                const btn = document.getElementById('deleteBtn');
                const consoleBox = document.getElementById('outputConsole');
                btn.disabled = true;
                btn.innerText = 'Deleting...';
                consoleBox.style.display = 'block';
                consoleBox.innerText = 'Invoking delete_pdf_vec.py script to purge documents...\\n';

                try {{
                    const response = await fetch('/delete_pdf_vec', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ filenames: selected }})
                    }});
                    const data = await response.json();
                    
                    if (data.status === 'success') {{
                        consoleBox.innerText += '\\n' + data.output;
                        consoleBox.innerText += '\\n--- Deletion finished successfully. Refreshing page in 3 seconds... ---';
                        setTimeout(() => {{ window.location.reload(); }}, 3000);
                    }} else {{
                        consoleBox.innerText += '\\nError: ' + (data.error || data.output || 'Unknown error');
                    }}
                }} catch (err) {{
                    consoleBox.innerText += '\\nRequest failed: ' + err.message;
                }} finally {{
                    btn.disabled = false;
                    btn.innerText = 'Execute Deletion';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/delete_pdf_vec")
def execute_delete_pdf_vec(payload: dict = Body(...)):
    """
    API endpoint that accepts a list of filenames to delete,
    runs the delete_pdf_vec.py script, and returns the step-by-step logs.
    """
    filenames = payload.get("filenames", [])
    if isinstance(filenames, str):
        filenames = [filenames]

    if not filenames:
        raise HTTPException(status_code=400, detail="No filenames provided for deletion.")

    # Sanitize filenames to prevent path traversal
    safe_filenames = [Path(f).name for f in filenames if Path(f).name]

    script_path = BASE_DIR / "delete_pdf_vec.py"
    if not script_path.is_file():
        raise HTTPException(status_code=500, detail="Deletion script delete_pdf_vec.py not found.")

    cmd = [sys.executable, str(script_path), *safe_filenames]
    log.info("Running deletion command: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(BASE_DIR),
        )

        return {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "output": proc.stdout,
            "error": proc.stderr,
            "deleted_files": safe_filenames,
        }
    except Exception as e:
        log.exception("Failed to execute deletion script")
        raise HTTPException(status_code=500, detail=str(e))
