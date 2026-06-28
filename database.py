"""
database.py — SQLite database setup with SQLAlchemy
"""
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./loandocs.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class LoanType(Base):
    __tablename__ = "loan_types"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String, unique=True, nullable=False)          # "Home Loan"
    code          = Column(String, unique=True, nullable=False)          # "home_loan"
    description   = Column(Text, default="")
    required_docs = Column(Text, default="[]")                           # JSON list of doc_keys
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    def get_required_docs(self):
        return json.loads(self.required_docs or "[]")

    def set_required_docs(self, docs: list):
        self.required_docs = json.dumps(docs)


class DocumentMaster(Base):
    __tablename__ = "document_master"

    id                = Column(Integer, primary_key=True, index=True)
    doc_key           = Column(String, unique=True, nullable=False)      # "aadhaar_front"
    display_name      = Column(String, nullable=False)                   # "Aadhaar Card (Front)"
    description       = Column(Text, default="")
    extraction_fields = Column(Text, default="[]")                       # JSON list of field names
    is_active         = Column(Boolean, default=True)
    serial_order      = Column(Integer, default=0)                       # Order in final PDF
    created_at        = Column(DateTime, default=datetime.utcnow)

    def get_extraction_fields(self):
        return json.loads(self.extraction_fields or "[]")


class Application(Base):
    __tablename__ = "applications"

    id              = Column(String, primary_key=True)                   # LOAN-2024-XXXX
    loan_type_code  = Column(String, nullable=False)
    applicant_name  = Column(String, default="")
    applicant_phone = Column(String, default="")
    applicant_email = Column(String, default="")
    status          = Column(String, default="incomplete")               # incomplete/complete/exported
    notes           = Column(Text, default="")
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id                = Column(Integer, primary_key=True, index=True)
    application_id    = Column(String, nullable=False, index=True)
    doc_key           = Column(String, nullable=False)                   # "aadhaar_front"
    doc_display_name  = Column(String, default="")
    telegram_file_id  = Column(String, default="")                      # Telegram storage ref
    original_filename = Column(String, default="")
    file_type         = Column(String, default="")                      # "image" or "pdf"
    extracted_fields  = Column(Text, default="{}")                      # JSON extracted data
    confidence_score  = Column(Float, default=0.0)
    serial_order      = Column(Integer, default=0)
    uploaded_at       = Column(DateTime, default=datetime.utcnow)

    def get_extracted_fields(self):
        return json.loads(self.extracted_fields or "{}")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    id          = Column(Integer, primary_key=True, index=True)
    key         = Column(String, unique=True, nullable=False)
    value       = Column(Text, default="")
    is_secret   = Column(Boolean, default=False)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_setting(db, key: str, default: str = "") -> str:
    row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    return row.value if row and row.value is not None else default


def set_setting(db, key: str, value: str, is_secret: bool = False):
    row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if row:
        row.value = value
        row.is_secret = is_secret
    else:
        row = SystemSetting(key=key, value=value, is_secret=is_secret)
        db.add(row)
    db.commit()
    return row


class AIProvider(Base):
    __tablename__ = "ai_providers"

    id             = Column(Integer, primary_key=True, index=True)
    provider_name  = Column(String, nullable=False)
    provider_type  = Column(String, default="vision")
    request_format = Column(String, default="openai")
    api_key        = Column(Text, default="")
    base_url       = Column(Text, default="")
    model_name     = Column(String, default="")
    priority       = Column(Integer, default=1)
    is_active      = Column(Boolean, default=True)
    notes          = Column(Text, default="")
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def masked_key(self):
        key = self.api_key or ""
        if not key:
            return ""
        return key[:8] + "..." + key[-4:] if len(key) > 12 else "saved"


class ApiUser(Base):
    __tablename__ = "api_users"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    api_key    = Column(String, unique=True, nullable=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables and seed default data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        _seed_defaults(db)
    finally:
        db.close()


def _seed_defaults(db):
    """Seed loan types and document master if empty."""



    # ── Legacy System Settings table is kept for backward compatibility only.

    # ── AI Providers ──────────────────────────────────────────────
    if db.query(AIProvider).count() == 0:
        db.add(AIProvider(
            provider_name="OpenRouter Vision",
            provider_type="vision",
            request_format="openai",
            api_key="",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="google/gemini-2.5-flash",
            priority=1,
            is_active=False,
            notes="Example only. Add your key and activate from Admin → AI Providers."
        ))

    # ── Document Master ──────────────────────────────────────────────
    if db.query(DocumentMaster).count() == 0:
        default_docs = [
            DocumentMaster(doc_key="aadhaar_front",   display_name="Aadhaar Card (Front)",
                           extraction_fields=json.dumps(["name","dob","gender","address","aadhaar_number"]), serial_order=1),
            DocumentMaster(doc_key="aadhaar_back",    display_name="Aadhaar Card (Back)",
                           extraction_fields=json.dumps(["address","pin_code"]), serial_order=2),
            DocumentMaster(doc_key="pan_card",        display_name="PAN Card",
                           extraction_fields=json.dumps(["name","father_name","dob","pan_number"]), serial_order=3),
            DocumentMaster(doc_key="passport",        display_name="Passport",
                           extraction_fields=json.dumps(["name","dob","passport_number","expiry_date","address"]), serial_order=4),
            DocumentMaster(doc_key="voter_id",        display_name="Voter ID Card",
                           extraction_fields=json.dumps(["name","dob","voter_id_number","address"]), serial_order=5),
            DocumentMaster(doc_key="driving_license", display_name="Driving License",
                           extraction_fields=json.dumps(["name","dob","dl_number","expiry_date","address"]), serial_order=6),
            DocumentMaster(doc_key="salary_slip_1",   display_name="Salary Slip (Month 1)",
                           extraction_fields=json.dumps(["employee_name","company_name","month","gross_salary","net_salary","deductions"]), serial_order=7),
            DocumentMaster(doc_key="salary_slip_2",   display_name="Salary Slip (Month 2)",
                           extraction_fields=json.dumps(["employee_name","company_name","month","gross_salary","net_salary","deductions"]), serial_order=8),
            DocumentMaster(doc_key="salary_slip_3",   display_name="Salary Slip (Month 3)",
                           extraction_fields=json.dumps(["employee_name","company_name","month","gross_salary","net_salary","deductions"]), serial_order=9),
            DocumentMaster(doc_key="bank_statement",  display_name="Bank Statement (6 months)",
                           extraction_fields=json.dumps(["account_holder","account_number","bank_name","ifsc","period","average_balance"]), serial_order=10),
            DocumentMaster(doc_key="itr_1",           display_name="Income Tax Return (Year 1)",
                           extraction_fields=json.dumps(["name","pan","assessment_year","total_income","tax_paid"]), serial_order=11),
            DocumentMaster(doc_key="itr_2",           display_name="Income Tax Return (Year 2)",
                           extraction_fields=json.dumps(["name","pan","assessment_year","total_income","tax_paid"]), serial_order=12),
            DocumentMaster(doc_key="property_papers", display_name="Property Documents",
                           extraction_fields=json.dumps(["property_address","owner_name","area","survey_number"]), serial_order=13),
            DocumentMaster(doc_key="business_proof",  display_name="Business Registration Proof",
                           extraction_fields=json.dumps(["business_name","registration_number","owner_name","address","type"]), serial_order=14),
            DocumentMaster(doc_key="gst_certificate", display_name="GST Certificate",
                           extraction_fields=json.dumps(["business_name","gstin","address","registration_date"]), serial_order=15),
            DocumentMaster(doc_key="form_16",         display_name="Form 16",
                           extraction_fields=json.dumps(["employee_name","employer_name","pan","financial_year","total_income","tds_deducted"]), serial_order=16),
            DocumentMaster(doc_key="photo",           display_name="Passport Size Photo",
                           extraction_fields=json.dumps(["face_detected"]), serial_order=17),
        ]
        db.add_all(default_docs)

    # ── Loan Types ───────────────────────────────────────────────────
    if db.query(LoanType).count() == 0:
        loan_types = [
            LoanType(
                name="Home Loan", code="home_loan",
                description="Documents required for home/property loan",
                required_docs=json.dumps([
                    "aadhaar_front","aadhaar_back","pan_card","photo",
                    "salary_slip_1","salary_slip_2","salary_slip_3",
                    "bank_statement","form_16","itr_1","property_papers"
                ])
            ),
            LoanType(
                name="Personal Loan", code="personal_loan",
                description="Documents required for personal loan",
                required_docs=json.dumps([
                    "aadhaar_front","aadhaar_back","pan_card","photo",
                    "salary_slip_1","salary_slip_2","salary_slip_3",
                    "bank_statement","form_16"
                ])
            ),
            LoanType(
                name="Business Loan", code="business_loan",
                description="Documents required for business loan",
                required_docs=json.dumps([
                    "aadhaar_front","aadhaar_back","pan_card","photo",
                    "bank_statement","itr_1","itr_2",
                    "business_proof","gst_certificate"
                ])
            ),
        ]
        db.add_all(loan_types)

    db.commit()
