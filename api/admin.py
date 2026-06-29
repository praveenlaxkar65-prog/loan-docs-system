"""
api/admin.py — Admin REST API (JWT protected)
All sensitive config (Telegram, AI providers, API keys) managed here.
Zero hardcoded secrets — everything in DB.
"""
import json
import secrets
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from database import (
    get_db, LoanType, DocumentMaster, Application, Document,
    ApiUser, AIProvider, SystemSetting, get_setting, set_setting,
)
from config import SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD

router = APIRouter(prefix="/admin/api", tags=["Admin"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/api/auth/login")
ALGORITHM = "HS256"


# ══════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════

def get_current_admin(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Not authorized")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/auth/login")
async def admin_login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != ADMIN_USERNAME or form.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": form.username, "role": "admin"}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════

@router.get("/dashboard")
async def dashboard_stats(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    return {
        "total_applications": db.query(Application).count(),
        "complete":           db.query(Application).filter(Application.status == "complete").count(),
        "incomplete":         db.query(Application).filter(Application.status == "incomplete").count(),
        "exported":           db.query(Application).filter(Application.status == "exported").count(),
        "total_documents":    db.query(Document).count(),
        "loan_types":         db.query(LoanType).filter(LoanType.is_active == True).count(),
        "ai_providers":       db.query(AIProvider).filter(AIProvider.is_active == True).count(),
    }


# ══════════════════════════════════════════════════════════════════
# SETTINGS (Telegram + misc)
# ══════════════════════════════════════════════════════════════════

class SettingsSave(BaseModel):
    telegram_bot_token: Optional[str] = None
    telegram_channel_id: Optional[str] = None

@router.get("/settings")
async def get_settings(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    token = get_setting(db, "telegram_bot_token", "")
    return {
        "telegram_bot_token":  (token[:8] + "..." + token[-4:]) if len(token) > 12 else ("set" if token else ""),
        "telegram_bot_token_set": bool(token),
        "telegram_channel_id": get_setting(db, "telegram_channel_id", ""),
    }

@router.post("/settings")
async def save_settings(data: SettingsSave, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    if data.telegram_bot_token is not None and data.telegram_bot_token.strip():
        set_setting(db, "telegram_bot_token", data.telegram_bot_token.strip(), is_secret=True)
    if data.telegram_channel_id is not None:
        set_setting(db, "telegram_channel_id", data.telegram_channel_id.strip())
    return {"success": True, "message": "Settings saved"}

@router.post("/settings/telegram/test")
async def test_telegram(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    from services.telegram_service import test_connection
    result = await test_connection()
    return result


# ══════════════════════════════════════════════════════════════════
# AI PROVIDERS
# ══════════════════════════════════════════════════════════════════

class AIProviderCreate(BaseModel):
    provider_name:  str
    provider_type:  str = "vision"        # vision / text / both
    request_format: str = "openai"        # anthropic / openai / gemini / custom
    api_key:        str = ""
    base_url:       str = ""
    model_name:     str = ""
    priority:       int = 1
    is_active:      bool = False
    timeout_seconds: float = 90.0
    retry_count:    int = 0
    notes:          str = ""

class AIProviderUpdate(BaseModel):
    provider_name:  Optional[str]   = None
    provider_type:  Optional[str]   = None
    request_format: Optional[str]   = None
    api_key:        Optional[str]   = None   # send "__CLEAR__" to erase
    base_url:       Optional[str]   = None
    model_name:     Optional[str]   = None
    priority:       Optional[int]   = None
    is_active:      Optional[bool]  = None
    timeout_seconds: Optional[float] = None
    retry_count:    Optional[int]   = None
    notes:          Optional[str]   = None

_VALID_FORMATS = {"anthropic", "openai", "gemini", "custom"}

def _provider_dict(p: AIProvider) -> dict:
    return {
        "id":              p.id,
        "provider_name":   p.provider_name,
        "provider_type":   p.provider_type,
        "request_format":  p.request_format,
        "base_url":        p.base_url,
        "model_name":      p.model_name,
        "priority":        p.priority,
        "is_active":       p.is_active,
        "timeout_seconds": p.timeout_seconds,
        "retry_count":     p.retry_count,
        "notes":           p.notes,
        "has_api_key":     bool(p.api_key),
        "api_key_masked":  p.masked_key(),
        "created_at":      p.created_at.isoformat() if p.created_at else None,
    }


@router.get("/ai-providers")
async def list_ai_providers(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    providers = db.query(AIProvider).order_by(AIProvider.priority.asc(), AIProvider.id.asc()).all()
    return [_provider_dict(p) for p in providers]


@router.post("/ai-providers")
async def create_ai_provider(
    data: AIProviderCreate, db: Session = Depends(get_db), _=Depends(get_current_admin)
):
    fmt = data.request_format.strip().lower()
    if fmt not in _VALID_FORMATS:
        raise HTTPException(400, f"request_format must be one of: {', '.join(_VALID_FORMATS)}")
    p = AIProvider(
        provider_name=data.provider_name.strip(),
        provider_type=data.provider_type.strip().lower() or "vision",
        request_format=fmt,
        api_key=data.api_key.strip(),
        base_url=data.base_url.strip(),
        model_name=data.model_name.strip(),
        priority=data.priority,
        is_active=data.is_active,
        timeout_seconds=data.timeout_seconds,
        retry_count=data.retry_count,
        notes=data.notes.strip(),
    )
    db.add(p)
    db.commit()
    return {"success": True, "id": p.id, "message": f"Provider '{p.provider_name}' created"}


@router.put("/ai-providers/{pid}")
async def update_ai_provider(
    pid: int, data: AIProviderUpdate, db: Session = Depends(get_db), _=Depends(get_current_admin)
):
    p = db.query(AIProvider).filter(AIProvider.id == pid).first()
    if not p:
        raise HTTPException(404, "AI provider not found")

    if data.provider_name  is not None: p.provider_name  = data.provider_name.strip()
    if data.provider_type  is not None: p.provider_type  = data.provider_type.strip().lower() or "vision"
    if data.request_format is not None:
        fmt = data.request_format.strip().lower()
        if fmt not in _VALID_FORMATS:
            raise HTTPException(400, f"request_format must be one of: {', '.join(_VALID_FORMATS)}")
        p.request_format = fmt
    if data.api_key is not None:
        if data.api_key.strip() == "__CLEAR__":
            p.api_key = ""
        elif data.api_key.strip():
            p.api_key = data.api_key.strip()
    if data.base_url        is not None: p.base_url        = data.base_url.strip()
    if data.model_name      is not None: p.model_name      = data.model_name.strip()
    if data.priority        is not None: p.priority        = data.priority
    if data.is_active       is not None: p.is_active       = data.is_active
    if data.timeout_seconds is not None: p.timeout_seconds = data.timeout_seconds
    if data.retry_count     is not None: p.retry_count     = data.retry_count
    if data.notes           is not None: p.notes           = data.notes.strip()

    db.commit()
    return {"success": True, "message": "Provider updated", "provider": _provider_dict(p)}


@router.delete("/ai-providers/{pid}")
async def delete_ai_provider(pid: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    p = db.query(AIProvider).filter(AIProvider.id == pid).first()
    if not p:
        raise HTTPException(404, "AI provider not found")
    db.delete(p)
    db.commit()
    return {"success": True, "message": "Provider deleted"}


@router.post("/ai-providers/{pid}/test")
async def test_ai_provider(pid: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    """Test a provider with a dummy image. Returns connection result."""
    from services.ai_provider_service import test_provider
    result = await test_provider(pid)
    return result


@router.post("/ai-providers/{pid}/activate")
async def activate_ai_provider(
    pid: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin),
):
    """Toggle active status of a provider."""
    p = db.query(AIProvider).filter(AIProvider.id == pid).first()
    if not p:
        raise HTTPException(404, "AI provider not found")
    p.is_active = not p.is_active
    db.commit()
    return {"success": True, "is_active": p.is_active, "message": f"Provider {'activated' if p.is_active else 'deactivated'}"}


# ══════════════════════════════════════════════════════════════════
# LOAN TYPES
# ══════════════════════════════════════════════════════════════════

class LoanTypeCreate(BaseModel):
    name: str
    code: str
    description: str = ""
    required_docs: list[str] = []

class LoanTypeUpdate(BaseModel):
    name:          Optional[str]       = None
    description:   Optional[str]       = None
    required_docs: Optional[list[str]] = None
    is_active:     Optional[bool]      = None


@router.get("/loan-types")
async def list_loan_types(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    return [
        {
            "id": lt.id, "name": lt.name, "code": lt.code,
            "description": lt.description,
            "required_docs": lt.get_required_docs(),
            "is_active": lt.is_active,
        }
        for lt in db.query(LoanType).all()
    ]


@router.post("/loan-types")
async def create_loan_type(data: LoanTypeCreate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    if db.query(LoanType).filter(LoanType.code == data.code).first():
        raise HTTPException(400, "Loan type code already exists")
    lt = LoanType(
        name=data.name, code=data.code, description=data.description,
        required_docs=json.dumps(data.required_docs),
    )
    db.add(lt); db.commit()
    return {"success": True, "id": lt.id}


@router.put("/loan-types/{lt_id}")
async def update_loan_type(lt_id: int, data: LoanTypeUpdate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    lt = db.query(LoanType).filter(LoanType.id == lt_id).first()
    if not lt:
        raise HTTPException(404, "Loan type not found")
    if data.name          is not None: lt.name        = data.name
    if data.description   is not None: lt.description = data.description
    if data.required_docs is not None: lt.set_required_docs(data.required_docs)
    if data.is_active     is not None: lt.is_active   = data.is_active
    db.commit()
    return {"success": True, "message": "Updated"}


@router.delete("/loan-types/{lt_id}")
async def delete_loan_type(lt_id: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    lt = db.query(LoanType).filter(LoanType.id == lt_id).first()
    if not lt:
        raise HTTPException(404, "Not found")
    lt.is_active = False; db.commit()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════
# DOCUMENT MASTER
# ══════════════════════════════════════════════════════════════════

class DocMasterCreate(BaseModel):
    doc_key:           str
    display_name:      str
    description:       str = ""
    extraction_fields: list[str] = []
    serial_order:      int = 99

class DocMasterUpdate(BaseModel):
    display_name:      Optional[str]       = None
    description:       Optional[str]       = None
    extraction_fields: Optional[list[str]] = None
    serial_order:      Optional[int]       = None
    is_active:         Optional[bool]      = None


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
async def create_doc_master(data: DocMasterCreate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    if db.query(DocumentMaster).filter(DocumentMaster.doc_key == data.doc_key).first():
        raise HTTPException(400, "Document key already exists")
    dm = DocumentMaster(
        doc_key=data.doc_key, display_name=data.display_name,
        description=data.description,
        extraction_fields=json.dumps(data.extraction_fields),
        serial_order=data.serial_order,
    )
    db.add(dm); db.commit()
    return {"success": True, "id": dm.id}


@router.put("/document-master/{dm_id}")
async def update_doc_master(dm_id: int, data: DocMasterUpdate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    dm = db.query(DocumentMaster).filter(DocumentMaster.id == dm_id).first()
    if not dm:
        raise HTTPException(404, "Not found")
    if data.display_name      is not None: dm.display_name      = data.display_name
    if data.description       is not None: dm.description       = data.description
    if data.extraction_fields is not None: dm.extraction_fields = json.dumps(data.extraction_fields)
    if data.serial_order      is not None: dm.serial_order      = data.serial_order
    if data.is_active         is not None: dm.is_active         = data.is_active
    db.commit()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════
# APPLICATIONS
# ══════════════════════════════════════════════════════════════════

@router.get("/applications")
async def list_applications(
    status: Optional[str] = None,
    loan_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _=Depends(get_current_admin),
):
    q = db.query(Application)
    if status:    q = q.filter(Application.status == status)
    if loan_type: q = q.filter(Application.loan_type_code == loan_type)
    total = q.count()
    apps  = q.order_by(Application.created_at.desc()).offset(offset).limit(limit).all()
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
async def delete_application(app_id: str, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Not found")
    db.query(Document).filter(Document.application_id == app_id).delete()
    db.delete(app); db.commit()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════════

class ApiUserCreate(BaseModel):
    name: str


@router.get("/api-keys")
async def list_api_keys(db: Session = Depends(get_db), _=Depends(get_current_admin)):
    users = db.query(ApiUser).all()
    return [
        {
            "id": u.id, "name": u.name,
            "api_key_masked": u.api_key[:8] + "..." + u.api_key[-4:] if len(u.api_key) > 12 else u.api_key,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.post("/api-keys")
async def create_api_key(data: ApiUserCreate, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    new_key = "ld_" + secrets.token_urlsafe(32)
    user = ApiUser(name=data.name, api_key=new_key)
    db.add(user); db.commit()
    return {
        "success": True,
        "id": user.id,
        "name": data.name,
        "api_key": new_key,          # full key — shown only this once
        "message": "Copy this API key now. It will not be shown again.",
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: int, db: Session = Depends(get_db), _=Depends(get_current_admin)):
    u = db.query(ApiUser).filter(ApiUser.id == key_id).first()
    if not u:
        raise HTTPException(404, "Not found")
    u.is_active = False; db.commit()
    return {"success": True, "message": "API key revoked"}
