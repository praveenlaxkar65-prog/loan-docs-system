"""
api/upload.py — Document upload, processing, application management endpoints
"""
import json
import random
import string
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import io

from database import get_db, Application, Document, DocumentMaster, LoanType, ApiUser
from services.image_service import process_image
from services.ai_service import recognize_and_extract
from services.telegram_service import upload_file as tg_upload
from services.checklist_service import get_application_status
from services.pdf_service import generate_application_pdf
from config import APP_ID_PREFIX

router = APIRouter(prefix="/api/v1", tags=["Documents"])


# ──────────────────────────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    user = db.query(ApiUser).filter(
        ApiUser.api_key == x_api_key,
        ApiUser.is_active == True
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return user


# ──────────────────────────────────────────────────────────────────
# APPLICATION ENDPOINTS
# ──────────────────────────────────────────────────────────────────

@router.post("/applications/create")
async def create_application(
    loan_type_code: str = Form(...),
    applicant_name: str = Form(""),
    applicant_phone: str = Form(""),
    applicant_email: str = Form(""),
    db: Session = Depends(get_db),
    _: ApiUser = Depends(verify_api_key),
):
    """Create a new loan application and get a unique Application ID."""
    # Verify loan type exists
    lt = db.query(LoanType).filter(LoanType.code == loan_type_code, LoanType.is_active == True).first()
    if not lt:
        raise HTTPException(status_code=404, detail=f"Loan type '{loan_type_code}' not found")

    app_id = _generate_app_id()
    app = Application(
        id=app_id,
        loan_type_code=loan_type_code,
        applicant_name=applicant_name,
        applicant_phone=applicant_phone,
        applicant_email=applicant_email,
        status="incomplete",
    )
    db.add(app)
    db.commit()

    return {
        "success": True,
        "application_id": app_id,
        "loan_type": loan_type_code,
        "loan_type_name": lt.name,
        "message": f"Application created. Upload documents using application_id: {app_id}",
    }


@router.get("/applications/{app_id}")
async def get_application(
    app_id: str,
    db: Session = Depends(get_db),
    _: ApiUser = Depends(verify_api_key),
):
    """Get full application details with document status."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    docs = db.query(Document).filter(Document.application_id == app_id).order_by(Document.serial_order).all()
    checklist = get_application_status(db, app_id, app.loan_type_code)

    return {
        "application_id": app.id,
        "loan_type": app.loan_type_code,
        "applicant_name": app.applicant_name,
        "applicant_phone": app.applicant_phone,
        "applicant_email": app.applicant_email,
        "status": app.status,
        "created_at": app.created_at.isoformat(),
        "checklist": checklist,
        "documents": [
            {
                "id": d.id,
                "doc_key": d.doc_key,
                "doc_display_name": d.doc_display_name,
                "extracted_fields": d.get_extracted_fields(),
                "confidence_score": d.confidence_score,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ],
    }


@router.get("/applications/{app_id}/missing")
async def get_missing_documents(
    app_id: str,
    db: Session = Depends(get_db),
    _: ApiUser = Depends(verify_api_key),
):
    """Get list of missing documents for an application."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    return get_application_status(db, app_id, app.loan_type_code)


# ──────────────────────────────────────────────────────────────────
# UPLOAD ENDPOINT
# ──────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_documents(
    application_id: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    _: ApiUser = Depends(verify_api_key),
):
    """
    Upload one or more documents for an application.
    Each file goes through: image processing → AI recognition → field extraction → Telegram storage.
    """
    app = db.query(Application).filter(Application.id == application_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    # Get all document masters for AI context
    all_masters = db.query(DocumentMaster).filter(DocumentMaster.is_active == True).all()
    known_doc_keys = [m.doc_key for m in all_masters]
    fields_map = {m.doc_key: m.get_extraction_fields() for m in all_masters}

    results = []
    for file in files:
        try:
            result = await _process_single_file(file, application_id, app.loan_type_code,
                                                 known_doc_keys, fields_map, db)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })

    # Update application status
    checklist = get_application_status(db, application_id, app.loan_type_code)
    if checklist.get("is_complete"):
        app.status = "complete"
        db.commit()

    return {
        "success": True,
        "application_id": application_id,
        "processed": len(results),
        "results": results,
        "checklist": checklist,
    }


# ──────────────────────────────────────────────────────────────────
# PDF EXPORT
# ──────────────────────────────────────────────────────────────────

@router.get("/applications/{app_id}/pdf")
async def export_pdf(
    app_id: str,
    db: Session = Depends(get_db),
    _: ApiUser = Depends(verify_api_key),
):
    """Generate and download final PDF for an application."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    pdf_bytes = await generate_application_pdf(db, app_id)

    app.status = "exported"
    db.commit()

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={app_id}_documents.pdf"},
    )


# ──────────────────────────────────────────────────────────────────
# LOAN TYPES (public info)
# ──────────────────────────────────────────────────────────────────

@router.get("/loan-types")
async def list_loan_types(db: Session = Depends(get_db), _: ApiUser = Depends(verify_api_key)):
    """List all active loan types with their required documents."""
    loan_types = db.query(LoanType).filter(LoanType.is_active == True).all()
    return {
        "loan_types": [
            {
                "code": lt.code,
                "name": lt.name,
                "description": lt.description,
                "required_documents": lt.get_required_docs(),
            }
            for lt in loan_types
        ]
    }


# ──────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────

async def _process_single_file(
    file: UploadFile,
    application_id: str,
    loan_type_code: str,
    known_doc_keys: list,
    fields_map: dict,
    db: Session,
) -> dict:
    raw_bytes = await file.read()

    # 1. Image enhance + crop
    processed_bytes = process_image(raw_bytes, file.filename or "doc.jpg")

    # 2. AI: recognize + extract
    ai_result = await recognize_and_extract(processed_bytes, known_doc_keys, fields_map)

    doc_key         = ai_result.get("doc_key", "unknown")
    confidence      = float(ai_result.get("confidence", 0.0))
    display_name    = ai_result.get("doc_display_name", doc_key.replace("_", " ").title())
    extracted_fields = ai_result.get("extracted_fields", {})

    # 3. Upload to Telegram
    safe_name = f"{application_id}_{doc_key}.jpg"
    caption   = f"AppID: {application_id} | Doc: {display_name}"
    tg_file_id = await tg_upload(processed_bytes, safe_name, caption)

    # 4. Get serial order
    master = db.query(DocumentMaster).filter(DocumentMaster.doc_key == doc_key).first()
    serial = master.serial_order if master else 999

    # 5. Remove old record if same doc_key already uploaded (replace)
    existing = db.query(Document).filter(
        Document.application_id == application_id,
        Document.doc_key == doc_key
    ).first()
    if existing:
        db.delete(existing)
        db.flush()

    # 6. Save to DB
    doc_record = Document(
        application_id=application_id,
        doc_key=doc_key,
        doc_display_name=display_name,
        telegram_file_id=tg_file_id,
        original_filename=file.filename or "",
        file_type="pdf" if (file.filename or "").endswith(".pdf") else "image",
        extracted_fields=json.dumps(extracted_fields),
        confidence_score=confidence,
        serial_order=serial,
    )
    db.add(doc_record)
    db.commit()

    return {
        "filename": file.filename,
        "success": True,
        "doc_key": doc_key,
        "doc_display_name": display_name,
        "confidence": confidence,
        "extracted_fields": extracted_fields,
        "telegram_file_id": tg_file_id,
    }


def _generate_app_id() -> str:
    year = datetime.now().year
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{APP_ID_PREFIX}-{year}-{suffix}"
