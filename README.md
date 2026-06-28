# 🏦 LoanDocs System

AI-powered loan document management system with automatic recognition, field extraction, and PDF generation.

---

## ✨ Features

- 📷 **Multi-format upload** — JPEG, PNG, PDF, WebP, camera photos
- 🤖 **AI Document Recognition** — Claude Vision identifies document type automatically
- 📝 **Field Extraction** — Name, DOB, Aadhaar, PAN, salary, bank details, etc.
- ✂️ **Auto Image Enhancement** — Crop, deskew, contrast boost via OpenCV
- 📁 **Telegram Storage** — Unlimited free cloud storage via Telegram Bot API
- ✅ **Missing Document Checker** — Loan-type wise checklist
- 🆔 **Unique Application IDs** — Auto-generated (LOAN-2024-XXXXXX)
- 📄 **PDF Export** — Professional bundled PDF with cover page + all documents
- 🔐 **API Key Auth** — Secure API access with key management
- 🛠️ **Full Admin Panel** — Manage loan types, documents, applications, keys

---

## 🚀 Setup (GitHub Codespaces / Local)

### 1. Clone / Open in Codespaces

```bash
# Open terminal
cd loan-docs-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your keys
nano .env
```

**Required keys:**

| Variable | How to get |
|---|---|
| `CLAUDE_API_KEY` | https://console.anthropic.com → API Keys |
| `TELEGRAM_BOT_TOKEN` | Message @BotFather on Telegram → `/newbot` |
| `TELEGRAM_CHANNEL_ID` | Create a private channel, add your bot as admin, get ID |

**How to get Telegram Channel ID:**
1. Create a private Telegram channel
2. Add your bot as administrator
3. Send a message in the channel
4. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Look for `"chat":{"id":-100XXXXXXXXX}` — that's your channel ID

### 4. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access

| URL | Description |
|---|---|
| `http://localhost:8000/admin` | Admin Panel (admin/admin123) |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |

---

## 📡 API Usage

### Step 1 — Get an API Key
Login to Admin Panel → API Keys → Generate Key

### Step 2 — Create Application

```bash
curl -X POST http://localhost:8000/api/v1/applications/create \
  -H "X-API-Key: your_api_key" \
  -F "loan_type_code=home_loan" \
  -F "applicant_name=Ramesh Kumar" \
  -F "applicant_phone=9876543210"
```

Response:
```json
{
  "success": true,
  "application_id": "LOAN-2024-AB1C2D",
  "loan_type": "home_loan"
}
```

### Step 3 — Upload Documents

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-API-Key: your_api_key" \
  -F "application_id=LOAN-2024-AB1C2D" \
  -F "files=@aadhaar_front.jpg" \
  -F "files=@pan_card.jpg" \
  -F "files=@salary_slip.pdf"
```

Response:
```json
{
  "success": true,
  "application_id": "LOAN-2024-AB1C2D",
  "processed": 3,
  "results": [
    {
      "filename": "aadhaar_front.jpg",
      "doc_key": "aadhaar_front",
      "doc_display_name": "Aadhaar Card (Front)",
      "confidence": 0.97,
      "extracted_fields": {
        "name": "Ramesh Kumar",
        "dob": "15/08/1985",
        "gender": "Male",
        "address": "123, Main Street, Indore, MP",
        "aadhaar_number": "XXXX-XXXX-5678"
      }
    }
  ],
  "checklist": {
    "is_complete": false,
    "total_required": 11,
    "total_uploaded": 3,
    "missing_documents": [...]
  }
}
```

### Step 4 — Check Missing Documents

```bash
curl http://localhost:8000/api/v1/applications/LOAN-2024-AB1C2D/missing \
  -H "X-API-Key: your_api_key"
```

### Step 5 — Download Final PDF

```bash
curl -o loan_package.pdf \
  http://localhost:8000/api/v1/applications/LOAN-2024-AB1C2D/pdf \
  -H "X-API-Key: your_api_key"
```

---

## 🛠️ Admin Panel Guide

### Loan Types
- Add new loan types (Vehicle Loan, Gold Loan, etc.)
- Edit the required documents checklist
- Activate/deactivate loan types

### Document Master
- Add new document types
- Edit what fields to extract from each document
- Set serial order in final PDF

### Applications
- View all applications
- Filter by status (incomplete/complete/exported)
- Delete applications

### API Keys
- Generate keys for different integrations
- Revoke compromised keys

---

## 📁 Project Structure

```
loan-docs-system/
├── main.py                     # App entry point
├── config.py                   # Environment config
├── database.py                 # SQLite models + seed data
├── requirements.txt
├── .env.example
├── api/
│   ├── upload.py               # Document upload + app endpoints
│   ├── admin.py                # Admin REST API
│   └── admin_panel.py          # Serves admin HTML
├── services/
│   ├── ai_service.py           # Claude Vision recognition + extraction
│   ├── telegram_service.py     # File storage via Telegram
│   ├── image_service.py        # Auto crop, deskew, enhance
│   ├── checklist_service.py    # Missing doc checker
│   └── pdf_service.py          # PDF generation
└── admin/
    └── templates/
        └── index.html          # Full admin panel SPA
```

---

## 🔧 Supported Document Types (Default)

| Doc Key | Document |
|---|---|
| `aadhaar_front` | Aadhaar Card Front |
| `aadhaar_back` | Aadhaar Card Back |
| `pan_card` | PAN Card |
| `passport` | Passport |
| `voter_id` | Voter ID |
| `driving_license` | Driving License |
| `salary_slip_1/2/3` | Salary Slips |
| `bank_statement` | Bank Statement |
| `itr_1/2` | Income Tax Returns |
| `property_papers` | Property Documents |
| `business_proof` | Business Registration |
| `gst_certificate` | GST Certificate |
| `form_16` | Form 16 |
| `photo` | Passport Photo |

> Add more anytime via Admin Panel → Document Master

---

## 🏦 Default Loan Types

| Loan Type | Code | Required Documents |
|---|---|---|
| Home Loan | `home_loan` | Aadhaar, PAN, 3 Salary Slips, Bank Statement, Form 16, ITR, Property Papers |
| Personal Loan | `personal_loan` | Aadhaar, PAN, 3 Salary Slips, Bank Statement, Form 16 |
| Business Loan | `business_loan` | Aadhaar, PAN, Bank Statement, 2 ITRs, Business Proof, GST Certificate |

> Customize or add more via Admin Panel → Loan Types

---

## ⚠️ Important Notes

- **Telegram file limit**: 20MB per file via Bot API (free)
- **Claude API**: Pay-per-use, ~$0.003 per document processed
- **SQLite**: Perfect for development; migrate to PostgreSQL for high production load
- **Change default admin password** in `.env` before deploying

---

## 🔒 Security Checklist Before Production

- [ ] Change `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env`
- [ ] Set a strong random `SECRET_KEY`
- [ ] Restrict CORS origins in `main.py`
- [ ] Use HTTPS (nginx reverse proxy)
- [ ] Rotate API keys regularly
