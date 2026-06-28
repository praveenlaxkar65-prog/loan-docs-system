"""
api/admin.py — Admin REST API routes (protected by JWT)
"""
import json
import secrets
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db, LoanType, DocumentMaster, Application, Document, ApiUser, AIProvider
from config import SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD

router = APIRouter(prefix="/admin/api", tags=["Admin"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/api/auth/login")
ALGORITHM = "HS256"


# ──────────────────────────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────────────────────────

def get_current_admin(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Not authorized")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/auth/login")
async def admin_login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != ADMIN_USERNAME or form.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": form.username, "role": "admin"}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


# ──────────────────────────────────────────────────────────────────
# DASHBOARD STATS
# ──────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def dashboard_stats(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    from datetime import date
    today = date.today()

    total_apps     = db.query(Application).count()
    complete_apps  = db.query(Application).filter(Application.status == "complete").count()
    incomplete     = db.query(Application).filter(Application.status == "incomplete").count()
    exported       = db.query(Application).filter(Application.status == "exported").count()
    total_docs     = db.query(Document).count()

    return {
        "total_applications": total_apps,
        "complete": complete_apps,
        "incomplete": incomplete,
        "exported": exported,
        "total_documents": total_docs,
        "loan_types": db.query(LoanType).filter(LoanType.is_active == True).count(),
    }


# ──────────────────────────────────────────────────────────────────
# LOAN TYPES
# ──────────────────────────────────────────────────────────────────

class LoanTypeCreate(BaseModel):
    name: str
    code: str
    description: str = ""
    required_docs: list[str] = []

class LoanTypeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    required_docs: Optional[list[str]] = None
    is_active: Optional[bool] = None


@router.get("/loan-types")
async def list_loan_types(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    lts = db.query(LoanType).all()
    return [
        {
            "id": lt.id, "name": lt.name, "code": lt.code,
            "description": lt.description,
            "required_docs": lt.get_required_docs(),
            "is_active": lt.is_active,
        }
        for lt in lts
    ]


@router.post("/loan-types")
async def create_loan_type(
    data: LoanTypeCreate,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin)
):
    if db.query(LoanType).filter(LoanType.code == data.code).first():
        raise HTTPException(status_code=400, detail="Loan type code already exists")

    lt = LoanType(
        name=data.name, code=data.code,
        description=data.description,
        required_docs=json.dumps(data.required_docs),
    )
    db.add(lt)
    db.commit()
    return {"success": True, "id": lt.id, "message": f"Loan type '{data.name}' created"}


@router.put("/loan-types/{lt_id}")
async def update_loan_type(
    lt_id: int, data: LoanTypeUpdate,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin)
):
    lt = db.query(LoanType).filter(LoanType.id == lt_id).first()
    if not lt:
        raise HTTPException(status_code=404, detail="Loan type not found")

    if data.name        is not None: lt.name        = data.name
    if data.description is not None: lt.description = data.description
    if data.required_docs is not None: lt.set_required_docs(data.required_docs)
    if data.is_active   is not None: lt.is_active   = data.is_active

    db.commit()
    return {"success": True, "message": "Loan type updated"}


@router.delete("/loan-types/{lt_id}")
async def delete_loan_type(lt_id: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    lt = db.query(LoanType).filter(LoanType.id == lt_id).first()
    if not lt:
        raise HTTPException(status_code=404, detail="Not found")
    lt.is_active = False
    db.commit()
    return {"success": True, "message": "Loan type deactivated"}


# ──────────────────────────────────────────────────────────────────
# DOCUMENT MASTER
# ──────────────────────────────────────────────────────────────────

class DocMasterCreate(BaseModel):
    doc_key: str
    display_name: str
    description: str = ""
    extraction_fields: list[str] = []
    serial_order: int = 99

class DocMasterUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    extraction_fields: Optional[list[str]] = None
    serial_order: Optional[int] = None
    is_active: Optional[bool] = None


@router.get("/document-master")
async def list_doc_master(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    docs = db.query(DocumentMaster).order_by(DocumentMaster.serial_order).all()
    return [
        {
            "id": d.id, "doc_key": d.doc_key, "display_name": d.display_name,
            "description": d.description,
            "extraction_fields": d.get_extraction_fields(),
            "serial_order": d.serial_order, "is_active": d.is_active,
        }
        for d in docs
    ]


@router.post("/document-master")
async def create_doc_master(
    data: DocMasterCreate,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin)
):
    if db.query(DocumentMaster).filter(DocumentMaster.doc_key == data.doc_key).first():
        raise HTTPException(status_code=400, detail="Document key already exists")

    dm = DocumentMaster(
        doc_key=data.doc_key, display_name=data.display_name,
        description=data.description,
        extraction_fields=json.dumps(data.extraction_fields),
        serial_order=data.serial_order,
    )
    db.add(dm)
    db.commit()
    return {"success": True, "id": dm.id}


@router.put("/document-master/{dm_id}")
async def update_doc_master(
    dm_id: int, data: DocMasterUpdate,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin)
):
    dm = db.query(DocumentMaster).filter(DocumentMaster.id == dm_id).first()
    if not dm:
        raise HTTPException(status_code=404, detail="Not found")

    if data.display_name      is not None: dm.display_name      = data.display_name
    if data.description       is not None: dm.description       = data.description
    if data.extraction_fields is not None: dm.extraction_fields = json.dumps(data.extraction_fields)
    if data.serial_order      is not None: dm.serial_order      = data.serial_order
    if data.is_active         is not None: dm.is_active         = data.is_active

    db.commit()
    return {"success": True, "message": "Updated"}



# ──────────────────────────────────────────────────────────────────
# UNIVERSAL AI PROVIDERS
# ──────────────────────────────────────────────────────────────────

class AIProviderCreate(BaseModel):
    provider_name: str
    provider_type: str = "vision"
    request_format: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model_name: str
    priority: int = 1
    is_active: bool = True
    notes: str = ""

class AIProviderUpdate(BaseModel):
    provider_name: Optional[str] = None
    provider_type: Optional[str] = None
    request_format: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None

def _provider_dict(p: AIProvider):
    return {"id": p.id, "provider_name": p.provider_name, "provider_type": p.provider_type, "request_format": p.request_format, "base_url": p.base_url, "model_name": p.model_name, "priority": p.priority, "is_active": p.is_active, "notes": p.notes, "has_api_key": bool(p.api_key), "api_key_masked": p.masked_key(), "created_at": p.created_at.isoformat() if p.created_at else None}

@router.get("/ai-providers")
async def list_ai_providers(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    return [_provider_dict(p) for p in db.query(AIProvider).order_by(AIProvider.priority.asc(), AIProvider.id.asc()).all()]

@router.post("/ai-providers")
async def create_ai_provider(data: AIProviderCreate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    fmt = data.request_format.strip().lower()
    if fmt not in {"anthropic", "openai", "gemini", "custom"}:
        raise HTTPException(status_code=400, detail="request_format must be anthropic/openai/gemini/custom")
    p = AIProvider(provider_name=data.provider_name.strip(), provider_type=data.provider_type.strip().lower() or "vision", request_format=fmt, api_key=data.api_key.strip(), base_url=data.base_url.strip(), model_name=data.model_name.strip(), priority=data.priority, is_active=data.is_active, notes=data.notes.strip())
    db.add(p); db.commit()
    return {"success": True, "id": p.id, "message": "AI provider created"}

@router.put("/ai-providers/{provider_id}")
async def update_ai_provider(provider_id: int, data: AIProviderUpdate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    p = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not p: raise HTTPException(status_code=404, detail="AI provider not found")
    if data.provider_name is not None: p.provider_name = data.provider_name.strip()
    if data.provider_type is not None: p.provider_type = data.provider_type.strip().lower() or "vision"
    if data.request_format is not None:
        fmt = data.request_format.strip().lower()
        if fmt not in {"anthropic", "openai", "gemini", "custom"}: raise HTTPException(status_code=400, detail="request_format must be anthropic/openai/gemini/custom")
        p.request_format = fmt
    if data.api_key is not None:
        if data.api_key == "__CLEAR__": p.api_key = ""
        elif data.api_key.strip(): p.api_key = data.api_key.strip()
    if data.base_url is not None: p.base_url = data.base_url.strip()
    if data.model_name is not None: p.model_name = data.model_name.strip()
    if data.priority is not None: p.priority = data.priority
    if data.is_active is not None: p.is_active = data.is_active
    if data.notes is not None: p.notes = data.notes.strip()
    db.commit()
    return {"success": True, "message": "AI provider updated"}

@router.delete("/ai-providers/{provider_id}")
async def delete_ai_provider(provider_id: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    p = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not p: raise HTTPException(status_code=404, detail="AI provider not found")
    db.delete(p); db.commit()
    return {"success": True, "message": "AI provider deleted"}

@router.post("/ai-providers/{provider_id}/activate")
async def activate_ai_provider(provider_id: int, exclusive: bool = True, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    p = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not p: raise HTTPException(status_code=404, detail="AI provider not found")
    if exclusive:
        db.query(AIProvider).filter(AIProvider.provider_type.in_([p.provider_type, "both", "vision"])).update({"is_active": False})
    p.is_active = True; db.commit()
    return {"success": True, "message": f"{p.provider_name} activated"}

# ──────────────────────────────────────────────────────────────────
# APPLICATIONS (Admin view)
# ──────────────────────────────────────────────────────────────────

@router.get("/applications")
async def list_applications(
    status: Optional[str] = None,
    loan_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin),
):
    query = db.query(Application)
    if status:    query = query.filter(Application.status == status)
    if loan_type: query = query.filter(Application.loan_type_code == loan_type)

    total = query.count()
    apps  = query.order_by(Application.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "applications": [
            {
                "id": a.id, "loan_type": a.loan_type_code,
                "applicant_name": a.applicant_name,
                "status": a.status,
                "created_at": a.created_at.isoformat(),
            }
            for a in apps
        ],
    }


@router.delete("/applications/{app_id}")
async def delete_application(
    app_id: str, db: Session = Depends(get_db), _=Depends(get_current_admin)
):
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Not found")
    db.query(Document).filter(Document.application_id == app_id).delete()
    db.delete(app)
    db.commit()
    return {"success": True, "message": f"Application {app_id} deleted"}


# ──────────────────────────────────────────────────────────────────
# API KEY MANAGEMENT
# ──────────────────────────────────────────────────────────────────

class ApiUserCreate(BaseModel):
    name: str


@router.get("/api-keys")
async def list_api_keys(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    users = db.query(ApiUser).all()
    return [
        {
            "id": u.id, "name": u.name,
            "api_key": u.api_key[:8] + "..." + u.api_key[-4:],  # masked
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.post("/api-keys")
async def create_api_key(
    data: ApiUserCreate,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin)
):
    new_key = "ld_" + secrets.token_urlsafe(32)
    user = ApiUser(name=data.name, api_key=new_key)
    db.add(user)
    db.commit()
    return {
        "success": True,
        "name": data.name,
        "api_key": new_key,   # shown only once
        "message": "Save this API key — it won't be shown again",
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    user = db.query(ApiUser).filter(ApiUser.id == key_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Not found")
    user.is_active = False
    db.commit()
    return {"success": True, "message": "API key revoked"}
