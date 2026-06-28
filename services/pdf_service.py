"""
services/pdf_service.py — Generate final loan document PDF
"""
import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from sqlalchemy.orm import Session
from database import Application, Document, LoanType, DocumentMaster
from services.telegram_service import download_file
from PIL import Image as PILImage

W, H = A4

# ──────────────────────────────────────────────────────────────────
# MAIN EXPORT FUNCTION
# ──────────────────────────────────────────────────────────────────

async def generate_application_pdf(db: Session, application_id: str) -> bytes:
    """
    Build final PDF with:
    - Cover page (application summary)
    - Extracted data table
    - Each document image in serial order
    """
    app = db.query(Application).filter(Application.id == application_id).first()
    if not app:
        raise Exception(f"Application {application_id} not found")

    loan_type = db.query(LoanType).filter(LoanType.code == app.loan_type_code).first()
    required_docs = loan_type.get_required_docs() if loan_type else []

    # Get all uploaded documents, sorted by serial_order
    docs = (
        db.query(Document)
        .filter(Document.application_id == application_id)
        .order_by(Document.serial_order)
        .all()
    )

    # Sort by required doc order
    def sort_key(doc):
        try:
            return required_docs.index(doc.doc_key)
        except ValueError:
            return 999

    docs_sorted = sorted(docs, key=sort_key)

    # Download all images from Telegram
    doc_images = {}
    for doc in docs_sorted:
        if doc.telegram_file_id:
            try:
                img_bytes = await download_file(doc.telegram_file_id)
                doc_images[doc.id] = img_bytes
            except Exception:
                doc_images[doc.id] = None

    # Build PDF
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    story = []
    styles = _get_styles()

    # Cover Page
    story += _build_cover(app, loan_type, docs_sorted, styles)
    story.append(PageBreak())

    # Extracted Data Summary
    story += _build_data_summary(docs_sorted, styles)
    story.append(PageBreak())

    # Individual Documents
    for i, doc in enumerate(docs_sorted, 1):
        story += _build_doc_page(doc, i, doc_images.get(doc.id), styles)
        if i < len(docs_sorted):
            story.append(PageBreak())

    pdf.build(story)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────
# PAGE BUILDERS
# ──────────────────────────────────────────────────────────────────

def _build_cover(app, loan_type, docs, styles) -> list:
    items = []

    # Header bar
    items.append(Spacer(1, 0.5*cm))
    items.append(Paragraph("🏦 LOAN DOCUMENT PACKAGE", styles["heading_center"]))
    items.append(Spacer(1, 0.3*cm))
    items.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a56db")))
    items.append(Spacer(1, 0.8*cm))

    # Application Info Table
    info_data = [
        ["Application ID",   app.id],
        ["Loan Type",        loan_type.name if loan_type else app.loan_type_code],
        ["Applicant Name",   app.applicant_name or "—"],
        ["Phone",            app.applicant_phone or "—"],
        ["Email",            app.applicant_email or "—"],
        ["Status",           app.status.upper()],
        ["Generated On",     datetime.now().strftime("%d %b %Y, %I:%M %p")],
        ["Total Documents",  str(len(docs))],
    ]

    tbl = Table(info_data, colWidths=[5.5*cm, 12*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#1a56db")),
        ("TEXTCOLOR",   (0,0), (0,-1), colors.white),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (1,0), (1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#d1d5db")),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    items.append(tbl)
    items.append(Spacer(1, 1*cm))

    # Document checklist
    items.append(Paragraph("Document Checklist", styles["subheading"]))
    items.append(Spacer(1, 0.3*cm))

    uploaded_keys = {d.doc_key for d in docs}
    if loan_type:
        checklist_data = [["#", "Document", "Status"]]
        for idx, dk in enumerate(loan_type.get_required_docs(), 1):
            status = "✅ Uploaded" if dk in uploaded_keys else "❌ Missing"
            name = dk.replace("_", " ").title()
            checklist_data.append([str(idx), name, status])

        ctbl = Table(checklist_data, colWidths=[1*cm, 12*cm, 4*cm])
        ctbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a56db")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
            ("PADDING",     (0,0), (-1,-1), 6),
            ("ALIGN",       (0,0), (0,-1), "CENTER"),
            ("ALIGN",       (2,0), (2,-1), "CENTER"),
        ]))
        items.append(ctbl)

    return items


def _build_data_summary(docs, styles) -> list:
    items = []
    items.append(Paragraph("Extracted Data Summary", styles["heading"]))
    items.append(Spacer(1, 0.3*cm))
    items.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a56db")))
    items.append(Spacer(1, 0.5*cm))

    for doc in docs:
        fields = doc.get_extracted_fields()
        if not fields:
            continue

        items.append(Paragraph(f"📄 {doc.doc_display_name}", styles["subheading"]))
        items.append(Spacer(1, 0.2*cm))

        field_data = [["Field", "Extracted Value"]]
        for key, val in fields.items():
            label = key.replace("_", " ").title()
            value = str(val) if val is not None else "—"
            field_data.append([label, value])

        ftbl = Table(field_data, colWidths=[6*cm, 11.5*cm])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#374151")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
            ("PADDING",     (0,0), (-1,-1), 6),
            ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ]))
        items.append(ftbl)
        items.append(Spacer(1, 0.6*cm))

    return items


def _build_doc_page(doc, index: int, img_bytes: bytes | None, styles) -> list:
    items = []
    items.append(Paragraph(f"Document {index}: {doc.doc_display_name}", styles["heading"]))
    items.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a56db")))
    items.append(Spacer(1, 0.4*cm))

    # Meta info
    meta = [
        ["Document Type", doc.doc_display_name],
        ["Uploaded At",   doc.uploaded_at.strftime("%d %b %Y, %I:%M %p")],
        ["AI Confidence", f"{int(doc.confidence_score * 100)}%"],
    ]
    mtbl = Table(meta, colWidths=[4*cm, 13.5*cm])
    mtbl.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
        ("PADDING",   (0,0), (-1,-1), 5),
        ("BACKGROUND",(0,0), (0,-1), colors.HexColor("#f3f4f6")),
    ]))
    items.append(mtbl)
    items.append(Spacer(1, 0.5*cm))

    # Document image
    if img_bytes:
        try:
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            iw, ih = pil_img.size
            max_w = 17 * cm
            max_h = 20 * cm
            ratio  = min(max_w / iw, max_h / ih)
            rw, rh = iw * ratio, ih * ratio
            img_buf = io.BytesIO(img_bytes)
            rl_img = RLImage(img_buf, width=rw, height=rh)
            items.append(rl_img)
        except Exception:
            items.append(Paragraph("⚠️ Image could not be loaded", styles["normal"]))
    else:
        items.append(Paragraph("⚠️ Image not available", styles["normal"]))

    return items


# ──────────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────────

def _get_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "normal": base["Normal"],
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"],
            fontSize=13, textColor=colors.HexColor("#1a56db"),
            spaceAfter=4
        ),
        "heading_center": ParagraphStyle(
            "heading_center", parent=base["Heading1"],
            fontSize=18, textColor=colors.HexColor("#1a56db"),
            alignment=TA_CENTER, spaceAfter=6
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=base["Heading3"],
            fontSize=11, textColor=colors.HexColor("#374151"),
            spaceAfter=3
        ),
    }
