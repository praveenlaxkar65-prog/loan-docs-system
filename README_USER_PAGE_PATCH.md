# LoanDocs User Page Patch

Adds a simple user/operator page at `/user/`.

Files changed/added:
- `main.py`
- `api/user_panel.py`
- `admin/templates/user.html`

Apply from project root:

```bash
unzip -o loan-docs-user-page-patch.zip -d .
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `/user/`, paste an API key, create application, upload documents, check status, download PDF.
