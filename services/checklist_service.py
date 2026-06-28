"""
services/checklist_service.py — Check uploaded vs required documents
"""
from sqlalchemy.orm import Session
from database import Document, LoanType, DocumentMaster


def get_application_status(db: Session, application_id: str, loan_type_code: str) -> dict:
    """
    Compare uploaded documents against loan type requirements.
    Returns complete status with missing docs list.
    """
    # Get loan type requirements
    loan_type = db.query(LoanType).filter(LoanType.code == loan_type_code).first()
    if not loan_type:
        return {"error": f"Loan type '{loan_type_code}' not found"}

    required_docs = loan_type.get_required_docs()

    # Get uploaded documents for this application
    uploaded_docs = db.query(Document).filter(
        Document.application_id == application_id
    ).all()
    uploaded_keys = {doc.doc_key for doc in uploaded_docs}

    # Build status for each required document
    doc_status = []
    missing_docs = []
    uploaded_count = 0

    for doc_key in required_docs:
        master = db.query(DocumentMaster).filter(
            DocumentMaster.doc_key == doc_key
        ).first()
        display_name = master.display_name if master else doc_key.replace("_", " ").title()

        is_uploaded = doc_key in uploaded_keys
        if is_uploaded:
            uploaded_count += 1
            # Get the actual document record
            doc_record = next((d for d in uploaded_docs if d.doc_key == doc_key), None)
            doc_status.append({
                "doc_key": doc_key,
                "display_name": display_name,
                "status": "uploaded",
                "uploaded_at": doc_record.uploaded_at.isoformat() if doc_record else None,
                "confidence": doc_record.confidence_score if doc_record else 0,
            })
        else:
            missing_docs.append({
                "doc_key": doc_key,
                "display_name": display_name,
                "status": "missing",
            })
            doc_status.append({
                "doc_key": doc_key,
                "display_name": display_name,
                "status": "missing",
            })

    total_required = len(required_docs)
    is_complete    = len(missing_docs) == 0

    return {
        "application_id": application_id,
        "loan_type": loan_type_code,
        "loan_type_name": loan_type.name,
        "is_complete": is_complete,
        "total_required": total_required,
        "total_uploaded": uploaded_count,
        "total_missing": len(missing_docs),
        "completion_percent": round((uploaded_count / total_required * 100) if total_required else 0, 1),
        "missing_documents": missing_docs,
        "document_status": doc_status,
    }


def get_extra_documents(db: Session, application_id: str, loan_type_code: str) -> list:
    """
    Return documents uploaded that are NOT in the required list
    (extra / bonus documents).
    """
    loan_type = db.query(LoanType).filter(LoanType.code == loan_type_code).first()
    if not loan_type:
        return []

    required_docs = set(loan_type.get_required_docs())
    uploaded_docs = db.query(Document).filter(
        Document.application_id == application_id
    ).all()

    return [
        {"doc_key": d.doc_key, "display_name": d.doc_display_name}
        for d in uploaded_docs
        if d.doc_key not in required_docs
    ]
