import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from rag import RAGEngine

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
ALLOWED_EXTENSIONS = {".pdf"}

app = FastAPI(
    title="Research Paper Assistant",
    description="A document-grounded RAG assistant for research papers.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

rag = RAGEngine(
    persist_directory=str(BASE_DIR / "faiss_db"),
    collection_name="research_papers",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "documents": rag.list_documents(),
            "stats": rag.stats(),
        },
    )

@app.get("/api/status")
async def status():
    return {
        "documents": rag.list_documents(),
        "document_count": len(rag.list_documents()),
        "chunk_count": rag.chunk_count(),
    }


@app.post("/api/upload")
async def upload_papers(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Please select at least one PDF.")

    uploaded = []
    errors = []

    for file in files:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            errors.append(f"{filename or 'Unnamed file'}: only PDF files are supported.")
            continue

        try:
            content = await file.read()

            if not content:
                errors.append(f"{filename}: the file is empty.")
                continue

            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                errors.append(
                    f"{filename}: exceeds the {MAX_FILE_SIZE_MB} MB file-size limit."
                )
                continue

            safe_name = f"{uuid.uuid4().hex[:10]}_{filename}"
            destination = UPLOAD_DIR / safe_name
            destination.write_bytes(content)

            result = rag.index_pdf(destination, display_name=filename)

            if result["chunks"] == 0:
                destination.unlink(missing_ok=True)
                errors.append(f"{filename}: no extractable text was found.")
                continue

            uploaded.append(
                {
                    "name": filename,
                    "pages": result["pages"],
                    "chunks": result["chunks"],
                }
            )

        except Exception as exc:
            errors.append(f"{filename}: {str(exc)}")

    return {
        "success": bool(uploaded),
        "uploaded": uploaded,
        "errors": errors,
        "documents": rag.list_documents(),
        "chunk_count": rag.chunk_count(),
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question about the uploaded papers.",
        )

    if not rag.has_documents():
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one research paper before asking a question.",
        )

    try:
        return rag.answer(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate an answer right now: {str(exc)}",
        )


@app.delete("/api/documents")
async def clear_documents():
    try:
        rag.clear()

        for path in UPLOAD_DIR.iterdir():
            if path.is_file():
                path.unlink()

        return {"success": True, "message": "All documents and indexed data were cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
