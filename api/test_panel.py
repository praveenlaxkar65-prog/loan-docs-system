"""
Test Panel API — isolated `/test` route
Does NOT touch existing Groq/Telegram/SQLite system.
"""

import logging
import traceback
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from services.ocr_service import process_uploaded_file, page_result_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test", tags=["test-panel"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "admin" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".pdf"}
MAX_FILE_SIZE_MB = 25


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def test_page(request: Request):
    """Render the test panel UI."""
    return templates.TemplateResponse("test.html", {"request": request})


@router.post("/upload")
async def test_upload(
    files: list[UploadFile] = File(...),
    multi_doc: bool = Form(default=True),
):
    """
    Accepts multiple image/PDF files, runs OCR pipeline, returns JSON.
    Isolated from main /api routes — no DB writes, no Telegram, no Groq.
    """
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded"})

    results = []
    errors = []

    for upload in files:
        filename = upload.filename or "unknown"
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            errors.append({
                "filename": filename,
                "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            })
            continue

        try:
            content = await upload.read()

            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                errors.append({
                    "filename": filename,
                    "error": f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_SIZE_MB}MB"
                })
                continue

            if len(content) == 0:
                errors.append({"filename": filename, "error": "Empty file"})
                continue

            page_results = await process_uploaded_file(
                file_bytes=content,
                filename=filename,
                multi_doc=multi_doc,
            )

            results.append({
                "filename": filename,
                "pages": [page_result_to_dict(pr) for pr in page_results],
            })

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")
            errors.append({
                "filename": filename,
                "error": f"Processing failed: {str(e)}"
            })

    return JSONResponse(content={
        "success": len(results) > 0,
        "results": results,
        "errors": errors,
        "total_files": len(files),
        "processed": len(results),
        "failed": len(errors),
    })


@router.get("/health")
async def test_health():
    """Quick check that PaddleOCR + OpenCV deps are available."""
    status = {"opencv": False, "paddleocr": False, "pdf_support": False}

    try:
        import cv2
        status["opencv"] = True
        status["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    try:
        from services.ocr_service import OCREngine
        OCREngine.get_instance()
        status["paddleocr"] = True
    except Exception as e:
        status["paddleocr_error"] = str(e)

    try:
        import fitz  # noqa
        status["pdf_support"] = True
        status["pdf_backend"] = "pymupdf"
    except ImportError:
        try:
            import pdf2image  # noqa
            status["pdf_support"] = True
            status["pdf_backend"] = "pdf2image"
        except ImportError:
            status["pdf_backend"] = None

    return JSONResponse(content=status)