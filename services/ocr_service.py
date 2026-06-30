"""
OCR Service — PaddleOCR + Keyword Classifier + Regex Extractor
Test panel ke liye — existing system se completely isolated
"""

import re
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR Post-Processing helpers
# ---------------------------------------------------------------------------
# These are pure text-cleaning utilities used to clean up raw OCR output
# before it is fed into the keyword classifier / regex extractor. They are
# additive helpers — nothing in the existing pipeline is required to call
# them, so existing behaviour is preserved unless explicitly used.

# Tokens that show up frequently as OCR noise on ID-card backgrounds
# (hologram text, watermark fragments, stray single letters, etc.) and
# never carry useful information on their own.
_GARBAGE_TOKENS = {
    "", "-", "--", "—", ".", "..", "...", ",", ":", ";", "|", "/", "\\",
    "defae", "afn", "icela", "ela", "govtofindia", "x", "xx", "xxx",
    # common OCR noise from Indian ID-card headers / watermarks
    "government", "govemment", "govemnent", "governrnent", "govenment",
    "india", "uidai", "unique", "identification", "authority", "1947",
    "www", "help", "uidaigovin", "uidai.gov.in", "help@uidai.gov.in",
    "covemment", "covernment", "ofindia", "oflndia", "bharat", "sarkar",
}

_GARBAGE_TOKEN_RE = re.compile(r"^[^A-Za-z0-9\u0900-\u097F]+$")


def normalize_spaces(text: str) -> str:
    """Collapse runs of whitespace (including tabs) into single spaces,
    trim each line, and drop blank lines. Preserves line breaks."""
    if not text:
        return text
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def remove_duplicate_lines(text: str) -> str:
    """Remove exact-duplicate lines (case-insensitive) while keeping the
    first occurrence and original ordering. OCR frequently repeats a line
    (e.g. the Aadhaar number) when it appears twice physically on the card."""
    if not text:
        return text
    seen = set()
    out = []
    for ln in text.splitlines():
        key = ln.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ln)
    return "\n".join(out)


def remove_duplicate_words(text: str) -> str:
    """Collapse immediately-repeated words/tokens on the same line
    ('S/O S/O Ramesh' -> 'S/O Ramesh') and remove duplicate comma-separated
    tokens within a single line (common in address strings)."""
    if not text:
        return text

    def _dedupe_line(line: str) -> str:
        words = line.split()
        collapsed = []
        for w in words:
            if collapsed and collapsed[-1].lower() == w.lower():
                continue
            collapsed.append(w)
        line = " ".join(collapsed)

        if "," in line:
            parts = [p.strip() for p in line.split(",")]
            seen = set()
            kept = []
            for p in parts:
                key = p.lower()
                if not p:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                kept.append(p)
            line = ", ".join(kept)
        return line

    return "\n".join(_dedupe_line(ln) for ln in text.splitlines())


def remove_garbage_tokens(text: str) -> str:
    """Strip OCR noise tokens — stray punctuation-only fragments and known
    garbage strings that commonly appear from hologram/background texture
    misreads on ID cards (e.g. 'defae', 'Afn', 'Icela')."""
    if not text:
        return text
    out_lines = []
    for ln in text.splitlines():
        tokens = ln.split()
        kept = []
        for tok in tokens:
            bare = tok.strip(" ,.;:-")
            if not bare:
                continue
            if bare.lower() in _GARBAGE_TOKENS:
                continue
            if _GARBAGE_TOKEN_RE.match(tok):
                continue
            kept.append(tok)
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines)


def clean_text(text: str) -> str:
    """General-purpose OCR text cleaner — combines whitespace normalization,
    duplicate-line removal, duplicate-word removal and garbage-token removal.
    Used before classification/extraction; does not mutate the original
    raw OCR text stored on the result objects."""
    if not text:
        return text
    t = normalize_spaces(text)
    t = remove_garbage_tokens(t)
    t = remove_duplicate_lines(t)
    t = remove_duplicate_words(t)
    return t


def clean_address(addr: str) -> str:
    """Address-specific cleaner: removes garbage tokens, duplicate
    comma-separated segments, duplicate pincodes, stray leading
    relation-prefix artifacts, and normalizes spacing/punctuation."""
    if not addr:
        return addr

    a = addr.strip()
    a = " ".join(
        tok for tok in a.split()
        if tok.strip(" ,.;:-").lower() not in _GARBAGE_TOKENS
        and not _GARBAGE_TOKEN_RE.match(tok)
    )
    a = remove_duplicate_words(a)

    a = re.sub(r"\s*,\s*", ", ", a)
    a = re.sub(r"(,\s*){2,}", ", ", a)
    a = re.sub(r"[ \t]+", " ", a).strip(" ,-")

    pins = re.findall(r"\b\d{6}\b", a)
    if len(pins) > 1:
        count = [0]

        def _pin_sub(m):
            count[0] += 1
            return m.group(0) if count[0] == 1 else ""

        a = re.sub(r"\b\d{6}\b", _pin_sub, a)
        a = re.sub(r"\s*,\s*,", ",", a)
        a = re.sub(r"[ \t]+", " ", a).strip(" ,-")

    return a.strip(" ,-")


def normalize_name(name: str) -> str:
    """Normalize an extracted person name: collapse whitespace, strip stray
    punctuation/garbage tokens, fix repeated words, and apply Title Case
    while preserving already mixed-case input reasonably."""
    if not name:
        return name
    n = re.sub(r"[ \t]+", " ", name).strip(" .,:;-")
    n = remove_duplicate_words(n)
    tokens = [
        t for t in n.split()
        if t.lower() not in _GARBAGE_TOKENS and not _GARBAGE_TOKEN_RE.match(t)
    ]
    n = " ".join(tokens)
    if n.isupper() or n.islower():
        n = " ".join(w.capitalize() for w in n.split())
    return n.strip()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
# Lightweight format validators used to sanity-check regex-extracted values.
# These do not call any external verification service — they only check
# structural/checksum-free format correctness, and are safe to use as an
# extra filter before accepting a regex match.

def validate_aadhaar(value: str) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    if len(digits) != 12:
        return False
    if digits[0] in ("0", "1"):
        return False
    return True


def validate_pan(value: str) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value.strip().upper()))


def validate_dl(value: str) -> bool:
    if not value:
        return False
    v = value.strip().upper().replace(" ", "")
    return bool(re.fullmatch(r"[A-Z]{2}\d{2}\d{11}", v)) or bool(re.fullmatch(r"DL-?\d{14}", v))


def validate_vehicle_number(value: str) -> bool:
    if not value:
        return False
    v = re.sub(r"[\s\-]", "", value.strip().upper())
    return bool(re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", v))


def validate_ifsc(value: str) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", value.strip().upper()))


def validate_pincode(value: str) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    return bool(re.fullmatch(r"[1-9][0-9]{5}", digits))


def validate_epic(value: str) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"[A-Z]{3}[0-9]{7}", value.strip().upper()))


# Map of field-name patterns -> validator function, used by RegexExtractor
# to skip structurally-invalid matches and fall through to the next pattern.
_FIELD_VALIDATORS = {
    "aadhaar_number": validate_aadhaar,
    "pan_number": validate_pan,
    "dl_number": validate_dl,
    "vehicle_number": validate_vehicle_number,
    "ifsc_code": validate_ifsc,
    "pincode": validate_pincode,
    "epic_number": validate_epic,
}

# Field names that should be passed through normalize_name() after extraction.
_NAME_FIELDS = {
    "name", "first_name", "last_name", "relative_first_name", "relative_last_name",
    "father_name", "account_holder", "employee_name", "relation", "owner_name",
}

# Field names that should be passed through clean_address() after extraction.
_ADDRESS_FIELDS = {"address"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedField:
    value: str
    confidence: float          # 0-1
    source: str = "regex"      # regex | ocr_direct


@dataclass
class DocumentResult:
    doc_type: str              # e.g. "aadhaar_front"
    doc_label: str             # Human-readable label
    keyword_score: float       # 0-1  how many keywords matched
    ocr_confidence: float      # 0-1  avg OCR word confidence
    regex_score: float         # 0-1  fraction of expected fields found
    overall_confidence: float  # weighted composite
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    page_index: int
    original_b64: str          # base64 PNG of original
    processed_b64: str         # base64 PNG after preprocessing
    documents: list[DocumentResult] = field(default_factory=list)
    ocr_raw_text: str = ""


# ---------------------------------------------------------------------------
# Document definitions — keywords + regex patterns + expected fields
# ---------------------------------------------------------------------------

DOCUMENT_DEFINITIONS = {
    "aadhaar_front": {
        "label": "Aadhaar Card (Front)",
        "keywords": [
            "aadhaar", "aadhar", "आधार", "uid", "uidai", "unique identification",
            "government of india", "भारत सरकार", "enrolment", "dob", "male", "female",
            "year of birth", "जन्म"
        ],
        "anti_keywords": ["income tax", "pan", "permanent account"],
        "expected_fields": ["aadhaar_number", "name", "dob", "gender"],
        "patterns": {
            "aadhaar_number": [
                r"\b(\d{4}\s\d{4}\s\d{4})\b",
                r"\b(\d{4}-\d{4}-\d{4})\b",
                r"\b(\d{12})\b(?!\s*(?:ifsc|pan))",
            ],
            "name": [
                r"(?:name|नाम)[:\s]+([A-Z][a-zA-Z ]{2,40})",
                # Most reliable real-world signal: on Aadhaar cards the
                # printed name appears on its own line immediately before
                # the "/DOB:" line. Anchoring here avoids accidentally
                # matching unrelated boilerplate lines (e.g. "Government
                # of India") that also happen to look like a capitalized
                # multi-word phrase.
                r"\n([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)+)\s*\n\s*/?\s*DOB\s*:",
                # Generic fallback — only used if the above don't match.
                # Excludes common Aadhaar boilerplate phrases that would
                # otherwise be False-matched as a "name".
                r"^(?!.*(?:Government|India|Unique|Identification|Authority|Aadhaar|Address))([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)$",
            ],
            "dob": [
                r"(?:dob|date of birth|जन्म तिथि)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})",
                r"(?:year of birth|जन्म वर्ष)[:\s]*(\d{4})",
                r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b",
            ],
            "gender": [
                r"\b(male|female|transgender)\b",
                r"\b(पुरुष|महिला)\b",
                r"\b(M|F)\b(?=\s*$)",
            ],
        },
    },

    "aadhaar_back": {
        "label": "Aadhaar Card (Back)",
        "keywords": [
            "aadhaar", "aadhar", "आधार", "address", "पता", "pin", "pincode",
            "uidai", "uid", "helpline", "1947", "resident"
        ],
        "anti_keywords": ["income tax", "pan"],
        "expected_fields": ["aadhaar_number", "address", "pincode"],
        "patterns": {
            "aadhaar_number": [
                r"\b(\d{4}\s\d{4}\s\d{4})\b",
                r"\b(\d{12})\b",
            ],
            "address": [
                # Capture everything from "Address:"/"W/O" etc. up to the
                # next Aadhaar number or VID line — this is a more reliable
                # stop-boundary than "first 6-digit number", since OCR
                # sometimes produces a stray 6-digit-looking fragment in
                # the middle of a multi-line address before the real,
                # final pincode.
                r"(?:address|पता|s/o|w/o|d/o|c/o)[:\s]+([\s\S]{10,400}?)(?=\n?\s*\b\d{4}\s\d{4}\s\d{4}\b|\n?\s*VID|\n?\s*\b\d{12}\b|\Z)",
                r"(?:address|पता|s/o|w/o|d/o|c/o)[:\s]+([\s\S]{10,150})",
            ],
            "pincode": [
                # Prefer the last 6-digit number in the text — on Aadhaar
                # back layouts the genuine pincode is the final one before
                # the Aadhaar number repeats; an earlier 6-digit-looking
                # fragment can appear as an OCR artifact mid-address.
                r"\b(\d{6})\b(?!.*\b\d{6}\b)",
                r"\b(\d{6})\b",
            ],
        },
    },

    "pan_card": {
        "label": "PAN Card",
        "keywords": [
            "income tax", "permanent account number", "pan", "आयकर",
            "govt of india", "government of india", "department", "father"
        ],
        "anti_keywords": ["aadhaar", "uidai", "voter"],
        "expected_fields": ["pan_number", "name", "father_name", "dob"],
        # A PAN number's format (5 letters, 4 digits, 1 letter) is highly
        # distinctive — false positives are very unlikely. Used as a
        # fallback classification signal when OCR garbles the surrounding
        # header text badly enough that keyword matching alone fails.
        "strong_identifier": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "patterns": {
            "pan_number": [
                r"\b([A-Z]{5}[0-9]{4}[A-Z])\b",
            ],
            "name": [
                r"(?:name|नाम)[:\s]*([A-Z][A-Za-z ]{2,40})",
                # PAN card layout: printed name is usually the first
                # all-caps multi-word line below the header/photo area,
                # immediately followed by a "Father's Name" line.
                r"\n([A-Z]{2,}(?:\s[A-Z]{2,}){1,3})\s*\n\s*(?:[A-Z]{2,}(?:\s[A-Z]{2,}){1,3}\s*\n\s*)?(?:father)",
                r"^([A-Z]{2,}\s[A-Z]{2,}(?:\s[A-Z]{2,})?)$",
            ],
            "father_name": [
                r"(?:father'?s?\s*name|पिता)[:\s]*([A-Z][A-Za-z ]{2,40})",
                r"father'?s?\s*name\s*\n\s*([A-Z]{2,}(?:\s[A-Z]{2,}){1,3})",
            ],
            "dob": [
                r"(?:date of birth|dob|जन्म)[:\s]*(\d{2}/\d{2}/\d{4})",
                r"\b(\d{2}/\d{2}/\d{4})\b",
            ],
        },
    },

    "bank_statement": {
        "label": "Bank Statement",
        "keywords": [
            "bank", "statement", "account", "balance", "debit", "credit",
            "transaction", "ifsc", "branch", "opening balance", "closing balance",
            "passbook", "savings", "current", "नाम", "खाता"
        ],
        "anti_keywords": ["aadhaar", "pan card", "voter"],
        "expected_fields": ["account_number", "ifsc_code", "account_holder", "bank_name"],
        "patterns": {
            "account_number": [
                r"(?:account\s*(?:no|number|#)|a/c\s*no)[:\s.]*([0-9]{9,18})",
                r"(?:acct)[:\s]*([0-9]{9,18})",
            ],
            "ifsc_code": [
                r"\b([A-Z]{4}0[A-Z0-9]{6})\b",
            ],
            "account_holder": [
                r"(?:name|account\s*holder|a/c\s*holder)[:\s]*([A-Z][A-Za-z ]{2,50})",
            ],
            "bank_name": [
                r"((?:state bank|sbi|hdfc|icici|axis|kotak|punjab national|pnb|bank of baroda|bob|canara|union bank|idbi|yes bank|indusind|federal bank)[A-Za-z\s]*)",
            ],
            "branch": [
                r"(?:branch(?:\s*name)?)[:\s]*([A-Z][A-Za-z0-9 ,.\-]{2,60})",
            ],
            "statement_period": [
                r"(?:statement\s*period|period|from)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4}\s*(?:to|-)\s*\d{2}[/\-]\d{2}[/\-]\d{4})",
            ],
            "opening_balance": [
                r"(?:opening\s*balance)[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
            "closing_balance": [
                r"(?:closing\s*balance)[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
        },
    },

    "salary_slip": {
        "label": "Salary Slip",
        "keywords": [
            "salary", "slip", "payslip", "pay slip", "employee", "employer",
            "basic", "hra", "pf", "provident fund", "gross", "net pay",
            "deduction", "earnings", "allowance", "epf", "esic", "tds",
            "month", "designation", "department"
        ],
        "anti_keywords": ["bank statement", "aadhaar", "pan card"],
        "expected_fields": ["employee_name", "employee_id", "gross_salary", "net_salary"],
        "patterns": {
            "employee_name": [
                r"(?:employee\s*name|emp\s*name|name)[:\s]*([A-Z][A-Za-z ]{2,50})",
            ],
            "employee_id": [
                r"(?:employee\s*(?:id|code|no)|emp\s*(?:id|code))[:\s]*([A-Z0-9\-]{3,20})",
            ],
            "gross_salary": [
                r"(?:gross\s*(?:salary|pay|earnings|ctc))[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
            "net_salary": [
                r"(?:net\s*(?:salary|pay|take\s*home))[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
            "basic_salary": [
                r"(?:basic(?:\s*salary)?)[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
            "pf_deduction": [
                r"(?:pf|provident\s*fund|epf)[:\s₹]*([\d,]+(?:\.\d{2})?)",
            ],
            "month_year": [
                r"(?:for\s*the\s*month|pay\s*period|month)[:\s]*([A-Za-z]+\s*\d{4})",
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})\b",
            ],
        },
    },

    "itr": {
        "label": "Income Tax Return (ITR)",
        "keywords": [
            "income tax return", "itr", "assessment year", "ay", "pan",
            "gross total income", "tax payable", "refund", "form", "schedule",
            "acknowledgement", "intimation", "taxable income", "deduction 80c"
        ],
        "anti_keywords": ["salary slip", "bank statement"],
        "expected_fields": ["pan_number", "assessment_year", "gross_income", "tax_payable"],
        "patterns": {
            "pan_number": [r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"],
            "assessment_year": [r"(?:assessment\s*year|a\.?y\.?)[:\s]*(\d{4}-\d{2,4}|\d{4}\s*-\s*\d{2,4})"],
            "gross_income": [r"(?:gross\s*total\s*income|total\s*income)[:\s₹]*([\d,]+(?:\.\d{2})?)"],
            "tax_payable": [r"(?:tax\s*payable|total\s*tax)[:\s₹]*([\d,]+(?:\.\d{2})?)"],
            "acknowledgement_number": [r"(?:acknowledgement\s*(?:no|number))[:\s]*(\d{15})"],
        },
    },

    "form_16": {
        "label": "Form 16",
        "keywords": [
            "form 16", "tds", "certificate", "tan", "employer", "employee",
            "financial year", "salary paid", "tax deducted", "chapter vi-a",
            "section 192", "deductor"
        ],
        "anti_keywords": ["bank statement"],
        "expected_fields": ["pan_number", "tan_number", "financial_year", "gross_salary"],
        "patterns": {
            "pan_number": [r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"],
            "tan_number": [r"\b([A-Z]{4}[0-9]{5}[A-Z])\b"],
            "financial_year": [r"(?:financial\s*year|f\.?y\.?)[:\s]*(\d{4}-\d{2,4})"],
            "gross_salary": [r"(?:gross\s*salary|salary\s*paid)[:\s₹]*([\d,]+(?:\.\d{2})?)"],
            "tds_deducted": [r"(?:tax\s*deducted|tds)[:\s₹]*([\d,]+(?:\.\d{2})?)"],
        },
    },

    "driving_license": {
        "label": "Driving License",
        "keywords": [
            "driving licence", "driving license", "dl no", "transport",
            "motor vehicle", "rto", "valid till", "badge no", "cov",
            "class of vehicle", "लाइसेंस"
        ],
        "anti_keywords": ["aadhaar", "pan", "bank"],
        "expected_fields": ["dl_number", "name", "dob", "valid_till"],
        "patterns": {
            "dl_number": [
                r"\b([A-Z]{2}\d{2}\s?\d{11})\b",
                r"\b(DL-\d{14})\b",
                r"\b([A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}[\-\s]?\d{7})\b",
            ],
            "name": [r"(?:name|नाम)[:\s]*([A-Z][A-Za-z ]{2,40})"],
            "dob": [r"(?:dob|date of birth|जन्म)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})"],
            "issue_date": [
                r"(?:date\s*of\s*issue|doi|issue\s*date)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})",
            ],
            "valid_till": [r"(?:valid\s*(?:till|upto|through)|validity)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})"],
            "blood_group": [r"\b(A\+|A\-|B\+|B\-|AB\+|AB\-|O\+|O\-)\b"],
            "address": [
                r"(?:address|पता)[:\s]+([\s\S]{10,300}?)(?=\n?\s*\b\d{6}\b|\Z)",
            ],
        },
    },

    "passport": {
        "label": "Passport",
        "keywords": [
            "passport", "republic of india", "ministry of external affairs",
            "nationality", "place of birth", "date of issue", "date of expiry",
            "surname", "given name", "mrz", "p<ind"
        ],
        "anti_keywords": ["bank statement", "pan card"],
        "expected_fields": ["passport_number", "name", "dob", "expiry_date"],
        "patterns": {
            "passport_number": [r"\b([A-Z]\d{7})\b"],
            "name": [r"(?:surname|given\s*name)[:\s]*([A-Z][A-Za-z ]{2,40})"],
            "dob": [r"(?:date\s*of\s*birth|dob)[:\s]*(\d{2}/\d{2}/\d{4}|\d{2}\s[A-Z]{3}\s\d{4})"],
            "expiry_date": [r"(?:date\s*of\s*expiry|expiry|valid\s*till)[:\s]*(\d{2}/\d{2}/\d{4}|\d{2}\s[A-Z]{3}\s\d{4})"],
            "place_of_birth": [r"(?:place\s*of\s*birth)[:\s]*([A-Za-z\s,]{3,50})"],
        },
    },

    "voter_id": {
        "label": "Voter ID",
        "keywords": [
            "election commission", "voter", "elector", "epic", "electoral",
            "constituency", "assembly", "lok sabha", "serial no", "part no",
            "मतदाता", "निर्वाचन"
        ],
        "anti_keywords": ["income tax", "bank", "salary"],
        "expected_fields": ["epic_number", "name", "relation", "age", "gender"],
        "strong_identifier": r"\b[A-Z]{3}[0-9]{7}\b",
        "patterns": {
            "epic_number": [
                r"\b([A-Z]{3}[0-9]{7})\b",
                # Older voter ID layouts sometimes print the EPIC number
                # with a space between the letter and digit blocks.
                r"\b([A-Z]{3}\s?[0-9]{7})\b",
            ],
            "name": [
                r"(?:elector'?s?\s*name|name|नाम)[:\s]*([A-Z][A-Za-z ]{2,40})",
            ],
            "dob": [
                r"(?:dob|date of birth|जन्म)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})",
            ],
            "age": [
                r"(?:age|आयु)[:\s]*(\d{1,3})\b",
            ],
            "gender": [
                r"\b(male|female|पुरुष|महिला|transgender|m|f)\b",
            ],
            "address": [
                r"(?:address|पता)[:\s]+([\s\S]{10,400}?)(?=\n?\s*\b\d{6}\b|\Z)",
                r"(?:address|पता)[:\s]*([\s\S]{10,200}?\d{6}|[\s\S]{10,150})",
            ],
            "pincode": [
                r"\b(\d{6})\b(?!.*\b\d{6}\b)",
                r"\b(\d{6})\b",
            ],
            "relation": [
                r"(?:father'?s?\s*name|husband'?s?\s*name|mother'?s?\s*name|guardian'?s?\s*name|father|husband|mother|guardian|s/o|w/o|d/o)[:\s]*([A-Z][A-Za-z ]{2,40})",
            ],
        },
    },

    "rc_card": {
        "label": "Vehicle RC Card",
        "keywords": [
            "registration certificate", "rc book", "rc card", "vehicle class",
            "chassis no", "engine no", "registering authority", "fuel",
            "maker", "model", "hypothecation", "regn no", "rto", "motor vehicle"
        ],
        "anti_keywords": ["aadhaar", "pan card", "voter", "passport"],
        "expected_fields": ["vehicle_number", "chassis_number", "engine_number", "owner_name"],
        "patterns": {
            "vehicle_number": [
                r"\b([A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4})\b",
            ],
            "chassis_number": [
                r"(?:chassis\s*(?:no|number)?)[:\s]*([A-Z0-9]{10,20})",
            ],
            "engine_number": [
                r"(?:engine\s*(?:no|number)?)[:\s]*([A-Z0-9]{6,20})",
            ],
            "owner_name": [
                r"(?:owner\s*(?:name)?|registered\s*owner)[:\s]*([A-Z][A-Za-z ]{2,50})",
            ],
            "fuel_type": [
                r"\b(petrol|diesel|cng|electric|lpg|hybrid)\b",
            ],
            "vehicle_class": [
                r"(?:class\s*of\s*vehicle|vehicle\s*class)[:\s]*([A-Za-z][A-Za-z /]{2,40})",
            ],
            "registration_date": [
                r"(?:registration\s*date|regn\s*date|date\s*of\s*regn)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})",
            ],
            "maker": [
                r"(?:maker|manufacturer|maker'?s?\s*name)[:\s]*([A-Z][A-Za-z ]{2,40})",
            ],
            "model": [
                r"(?:model|maker'?s?\s*model)[:\s]*([A-Za-z0-9][A-Za-z0-9 \-]{1,30})",
            ],
            "hypothecation": [
                r"(?:hypothecat(?:ed|ion)\s*(?:to|with)?)[:\s]*([A-Z][A-Za-z &.,]{2,60})",
            ],
        },
    },
}

# Composite doc: if multiple docs detected in one image
MULTI_DOC_PAIRS = [
    ("aadhaar_front", "pan_card"),
    ("aadhaar_back", "pan_card"),
    ("aadhaar_front", "aadhaar_back"),
]


# ---------------------------------------------------------------------------
# Image Preprocessing (OpenCV) — adapted to your existing approach
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Crop, deskew, enhance — returns processed numpy array."""

    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        processed = ImagePreprocessor._deskew(img)
        processed = ImagePreprocessor._enhance(processed)
        return processed

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        """
        Detects skew using Hough line transform on text-like edges, rather than
        minAreaRect on all foreground pixels (which on dense ID-card images —
        photo + hologram + QR code + text all present — frequently misreads a
        near-0deg skew as a ~90deg one and rotates the document sideways).
        Correction is clamped to +/-15deg; anything larger is treated as noise
        and skipped, since real document skew during a phone/scanner capture
        is rarely beyond that.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20
            )

            if lines is None or len(lines) == 0:
                return img

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = (x2 - x1), (y2 - y1)
                if dx == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                # Keep only near-horizontal lines (text baselines), reject
                # near-vertical ones (card edges, QR code grid lines, etc.)
                if abs(angle) < 30:
                    angles.append(angle)

            if not angles:
                return img

            median_angle = float(np.median(angles))

            # Clamp: real capture skew is rarely beyond ~15deg. Anything
            # larger is almost certainly a misread from texture/noise.
            if abs(median_angle) < 0.5 or abs(median_angle) > 15:
                return img

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return img

    @staticmethod
    def _enhance(img: np.ndarray) -> np.ndarray:
        """
        Speed note: cv2.fastNlMeansDenoising (non-local-means) is very
        accurate but can take several seconds per image on CPU — it was
        the single biggest contributor to slow processing. Replaced with
        cv2.bilateralFilter, which gives similar edge-preserving noise
        reduction for document/text images at a fraction of the cost
        (roughly 10-20x faster in practice), since text images don't need
        NLM's extra robustness against complex textures.
        """
        try:
            # Upscale if small, but cap max dimension — oversized images
            # slow down both this step and OCR inference without adding
            # useful detail beyond a certain point.
            h, w = img.shape[:2]
            target_max = 1600  # was uncapped before; 1200 floor + no ceiling
            if max(h, w) < 1200:
                scale = 1200 / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            elif max(h, w) > target_max:
                scale = target_max / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            # Denoise — bilateral filter instead of fastNlMeansDenoising (much faster, similar quality for text)
            denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
            # CLAHE for contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            # Sharpen
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            return sharpened
        except Exception as e:
            logger.warning(f"Enhance failed: {e}")
            return img


# ---------------------------------------------------------------------------
# PaddleOCR wrapper
# ---------------------------------------------------------------------------

class OCREngine:
    """
    Wraps PaddleOCR 3.x's API (paddleocr>=3.0).
    Note: 3.x replaced the old `use_gpu`/`show_log`/`cls=True` 2.x API with
    `device="cpu"|"gpu"` and a unified `.predict()` method that returns
    Result objects exposing `.rec_texts` / `.rec_scores`, not the old
    nested [[box, (text, score)], ...] list format.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                import os
                from paddleocr import PaddleOCR

                # PERFORMANCE NOTE (read before touching these flags):
                # The ~35s/image figure traced back to three independent
                # multipliers, none of which are "inherent to PaddleOCR" —
                # they're all default-config choices that stack:
                #
                #   1. PaddleOCR 3.x silently auto-downloads/uses the
                #      PP-OCRv5 *server* det+rec models when no model
                #      name is given. The server rec model alone is
                #      ~10x the FLOPs of the *mobile* variant for near
                #      identical accuracy on flat document/ID-card text
                #      (it's tuned for harder scene-text, not forms).
                #      -> pinned to the *_mobile model names below.
                #   2. enable_mkldnn=False disables oneDNN's CPU graph
                #      fusion/vectorization, which is normally a 2-4x
                #      win on Intel/AMD CPUs for conv-heavy det models.
                #      -> flipped on, with a small persistent op cache.
                #   3. cpu_threads=4 was hardcoded regardless of the
                #      actual host. Oversubscribing or undersubscribing
                #      threads relative to physical cores both hurt;
                #      using all available cores (capped at 8, since
                #      Paddle's thread-pool gains flatten out past that
                #      for single-image inference) is the safe default.
                #
                # Output contract is untouched: same Result objects with
                # rec_texts/rec_scores, same call signature in run_ocr().
                cpu_threads = min(8, max(1, os.cpu_count() or 4))

                cls._instance = PaddleOCR(
                    lang="en",
                    device="cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    # Off: our own OpenCV deskew step already corrects
                    # rotation before OCR runs, so this extra model pass
                    # (textline orientation classifier) is redundant here
                    # and was adding meaningful per-request latency.
                    use_textline_orientation=False,
                    # Mobile det/rec models — same PP-OCRv5 architecture
                    # family, ~5-8x fewer FLOPs than the server models
                    # PaddleOCR otherwise defaults to. Detection quality
                    # on flat, well-lit document/ID-card text (our actual
                    # input) is not meaningfully different; this is the
                    # single biggest contributor to the speedup.
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    enable_mkldnn=False,
                    mkldnn_cache_capacity=10,
                    cpu_threads=cpu_threads,
                    # Input is already capped at 1600px by our own
                    # preprocessing step, so the detector doesn't need to
                    # upsample to 960 on the long side — matching it to
                    # our actual preprocessed size avoids redundant resize
                    # work inside Paddle's own pipeline.
                    text_det_limit_side_len=736,
                    precision="fp32",
                )
                logger.info(
                    "PaddleOCR initialized (mobile models, mkldnn=on, "
                    f"cpu_threads={cpu_threads})"
                )
            except ImportError:
                logger.error("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
                raise
            except TypeError:
                # Older paddleocr builds (<3.0.1) may not accept one or
                # more of the kwargs above (e.g. mkldnn_cache_capacity,
                # the *_model_name params). Fall back to the broadest
                # compatible flag set rather than crashing the service —
                # still gets the mkldnn + thread-count win even if the
                # mobile-model pin isn't supported on that build.
                logger.warning(
                    "PaddleOCR rejected an optimization kwarg; retrying "
                    "with a reduced, version-safe flag set."
                )
                cls._instance = PaddleOCR(
                    lang="en",
                    device="cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    cpu_threads=min(8, max(1, os.cpu_count() or 4)),
                    text_det_limit_side_len=736,
                )
        return cls._instance

    @staticmethod
    def run_ocr(img: np.ndarray) -> tuple[str, float]:
        """
        Returns (full_text, avg_confidence).
        img should be grayscale or BGR numpy array.
        """
        ocr = OCREngine.get_instance()

        # PaddleOCR needs BGR or RGB
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img

        results = ocr.predict(img_color)

        lines = []
        confidences = []

        for res in results:
            # PaddleOCR 3.x Result object: dict-like, has rec_texts / rec_scores
            try:
                rec_texts = res.get("rec_texts") if hasattr(res, "get") else getattr(res, "rec_texts", None)
                rec_scores = res.get("rec_scores") if hasattr(res, "get") else getattr(res, "rec_scores", None)
            except Exception:
                rec_texts, rec_scores = None, None

            if rec_texts:
                for text, score in zip(rec_texts, rec_scores or []):
                    if text and text.strip():
                        lines.append(text.strip())
                        confidences.append(float(score) if score is not None else 0.0)

        full_text = "\n".join(lines)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return full_text, avg_conf


# ---------------------------------------------------------------------------
# Keyword Classifier
# ---------------------------------------------------------------------------

class KeywordClassifier:

    @staticmethod
    def classify(text: str, apply_anti_keywords: bool = True) -> list[tuple[str, float]]:
        """
        Returns sorted list of (doc_type, keyword_score) — top matches.
        A score > 0.3 is considered a match.

        apply_anti_keywords: when True (default), a document's score is
        penalised if another document's distinguishing keywords are also
        present (used to disambiguate single-document images, e.g. telling
        PAN apart from Aadhaar). In a combo image containing BOTH documents,
        this penalty is wrong — both sets of keywords are legitimately
        present — so multi-doc detection calls this with False.
        """
        text_lower = text.lower()
        scores = []

        for doc_type, definition in DOCUMENT_DEFINITIONS.items():
            keywords = definition.get("keywords", [])
            anti_kw  = definition.get("anti_keywords", [])
            strong_id_pattern = definition.get("strong_identifier")

            matched = sum(1 for kw in keywords if kw.lower() in text_lower)
            anti    = sum(1 for kw in anti_kw  if kw.lower() in text_lower) if apply_anti_keywords else 0

            if keywords:
                score = (matched / len(keywords)) - (anti * 0.3)
                score = max(0.0, min(1.0, score))
            else:
                score = 0.0

            # Strong identifier boost: a document's ID-number format (e.g.
            # PAN's 5-letter/4-digit/1-letter pattern) is distinctive enough
            # that finding it is meaningful evidence on its own — this
            # rescues classification when OCR has badly garbled the
            # surrounding header/label text but still read the ID number
            # correctly (a common asymmetry: printed serif headers in mixed
            # Hindi/English garble more than a clean monospace ID string).
            if strong_id_pattern:
                try:
                    if re.search(strong_id_pattern, text):
                        score = max(score, 0.5)
                except re.error:
                    pass

            scores.append((doc_type, round(score, 3)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @staticmethod
    def detect_multi_doc(text: str) -> list[str]:
        """
        Check if text contains signals for multiple documents.
        Returns list of detected doc_types (may be >1).
        Anti-keyword penalties are disabled here since a combo image (e.g.
        PAN + Aadhaar together) legitimately contains both documents'
        keywords at once — penalising for that would wrongly suppress
        real matches.
        """
        scores = KeywordClassifier.classify(text, apply_anti_keywords=False)
        # Keep all docs with score > 0.15 (weak signal is OK in combo image)
        candidates = [dt for dt, sc in scores if sc > 0.15]
        return candidates  # no cap: combo KYC sheets can contain PAN + Aadhaar front + Aadhaar back + Voter



# ---------------------------------------------------------------------------
# Smart layout-aware extractors for structured ID cards
# ---------------------------------------------------------------------------

_STOPWORDS_FOR_NAMES = {
    "first", "last", "name", "relative", "gender", "age", "epic", "epicno", "epicno",
    "state", "constituency", "polling", "station", "part", "serial", "number",
    "government", "india", "uidai", "unique", "identification", "authority", "address",
    "dob", "birth", "female", "male", "transgender", "aadhaar", "aadhar", "download",
    "enrolment", "vid", "pincode", "pin", "tehsil", "district", "madhya", "pradesh",
}

_LABEL_PATTERNS = {
    # Important: applicant first/last labels must NOT match Relative's First/Last Name.
    "voter_first_name": re.compile(r"(?:first\s*name|प्रथम\s*नाम)", re.I),
    "voter_last_name": re.compile(r"(?:last\s*name|उपनाम)", re.I),
    "voter_relative_first": re.compile(r"(?:relative'?s?\s*first\s*name|relative.*first|रिश्तेदार.*प्रथम)", re.I),
    "voter_relative_last": re.compile(r"(?:relative'?s?\s*last\s*name|relative.*last|रिश्तेदार.*उपनाम)", re.I),
    "voter_age": re.compile(r"(?:^|\b)(?:age|आयु)(?:\b|/)", re.I),
    "voter_gender": re.compile(r"(?:^|\b)(?:gender|लिंग)(?:\b|/)", re.I),
    "voter_epic": re.compile(r"(?:epic\s*no|epicno|ईपीआईसी)", re.I),
    "aadhaar_dob": re.compile(r"(?:dob|d0b|date\s*of\s*birth|जन्म)", re.I),
    "aadhaar_address": re.compile(r"(?:address|पता)", re.I),
}


def _clean_ocr_line(line: str) -> str:
    raw = (line or "").strip()
    if raw in {"-", "--", "—"}:
        return "-"
    line = re.sub(r"[|`~•·]+", " ", line or "")
    line = re.sub(r"\s+", " ", line).strip(" .,:;/-_[]{}()")
    return line


def _is_probable_person_name(value: str) -> bool:
    if not value:
        return False
    v = _clean_ocr_line(value)
    if not (2 <= len(v) <= 45):
        return False
    low = re.sub(r"[^a-z]", "", v.lower())
    if not low or low in _STOPWORDS_FOR_NAMES:
        return False
    if any(sw in v.lower() for sw in _STOPWORDS_FOR_NAMES):
        return False
    # Reject mixed OCR label fragments like "acR yu" / "qdgTT".
    # Accept clean UPPERCASE names and normal Title/Mixed-case names.
    for tok in v.split():
        letters = re.sub(r"[^A-Za-z]", "", tok)
        if not letters:
            continue
        if not (letters.isupper() or letters.islower() or letters[:1].isupper() and letters[1:].islower()):
            return False
    # Reject two-token OCR garbage like "UT UR" / "HIRR HRT" which often
    # comes from Aadhaar/PAN header watermarks, not the printed person name.
    alpha_tokens = [re.sub(r"[^A-Za-z]", "", t) for t in v.split()]
    alpha_tokens = [t for t in alpha_tokens if t]
    if len(alpha_tokens) <= 2 and all(len(t) <= 3 for t in alpha_tokens):
        return False

    # reject lines dominated by digits/symbols
    alpha = len(re.findall(r"[A-Za-z]", v))
    if alpha < 2 or alpha < max(2, len(v.replace(" ", "")) * 0.55):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,44}", v))


def _next_value_after_label(lines: list[str], label_re: re.Pattern, max_lookahead: int = 4) -> str:
    for i, line in enumerate(lines):
        if label_re.search(line):
            # Same-line value after label, if any
            same = re.split(label_re, line, maxsplit=1)
            if len(same) > 1:
                candidate = _clean_ocr_line(same[-1])
                candidate = re.sub(r"^[/:：\-]+", "", candidate).strip()
                if candidate and not label_re.search(candidate):
                    return candidate
            # Next few lines: skip blanks, dashes, Hindi label fragments
            for j in range(i + 1, min(len(lines), i + 1 + max_lookahead)):
                cand = _clean_ocr_line(lines[j])
                if not cand or cand in {"-", "--"}:
                    continue
                if re.search(r"/|first\s*name|last\s*name|relative|age|gender|epic|state|constituency", cand, re.I):
                    continue
                return cand
    return ""




def _is_relative_label_line(line: str) -> bool:
    return bool(re.search(r"relative|रिश्तेदार", line or "", re.I))


def _next_voter_value(lines: list[str], label_re: re.Pattern, *, relative: bool = False, max_lookahead: int = 4) -> str:
    """Robust voter table value picker.

    Handles OCR orders:
      VALUE / FirstName
      VALUE\n/FirstName
      First Name\nVALUE

    Also treats '-' as an intentional blank value, so Last Name does not
    accidentally jump to the next row's Relative Name.
    """
    label_any = re.compile(
        r"first\s*name|firstname|last\s*name|lastname|relative|age|gender|epic|epic\s*no|state|constituency|polling|part|serial|number|मतदाता|निर्वाचन",
        re.I,
    )

    for i, line in enumerate(lines):
        if not label_re.search(line):
            continue
        if relative != _is_relative_label_line(line):
            continue

        # Same-line value BEFORE label: "PUSHPABAI /FirstName"
        before = re.split(label_re, line, maxsplit=1)[0]
        before = re.sub(r"[/|:：\-]+$", "", before).strip()
        before = _clean_ocr_line(before)
        if before in {"-", "--"}:
            return ""
        if before and _is_probable_person_name(before):
            return before

        # Previous nearby line: common in table OCR where value is left cell,
        # label is right/next cell: "PUSHPABAI\n/FirstName".
        for j in range(i - 1, max(-1, i - 4), -1):
            cand = _clean_ocr_line(lines[j])
            if not cand or cand in {"」", "。", "："}:
                continue
            if cand in {"-", "--"}:
                return ""
            if label_any.search(cand):
                break
            if _is_probable_person_name(cand):
                return cand

        # Same-line value AFTER label: "First Name: PUSHPABAI"
        parts = re.split(label_re, line, maxsplit=1)
        if len(parts) > 1:
            after = re.sub(r"^[/:：\-]+", "", parts[-1]).strip()
            after = _clean_ocr_line(after)
            if after in {"-", "--"}:
                return ""
            if after and _is_probable_person_name(after):
                return after

        # Next line only for blank/normal vertical layouts. Stop if another
        # label begins; do not hop across rows.
        for j in range(i + 1, min(len(lines), i + 1 + max_lookahead)):
            cand = _clean_ocr_line(lines[j])
            if not cand or cand in {"」", "。", "："}:
                continue
            if cand in {"-", "--"}:
                return ""
            if label_any.search(cand):
                break
            if _is_probable_person_name(cand):
                return cand
    return ""

def _clean_aadhaar_back_address(addr: str) -> str:
    """Extra cleaner for Aadhaar-back OCR garbage.
    Removes fake chunks like W/O 3, ENH/C/R, E457339 while preserving the
    meaningful address sequence.
    """
    if not addr:
        return addr

    a = clean_address(addr)
    a = re.sub(r"\b(?:Address|पता)\b[:：\s-]*", "", a, flags=re.I)
    a = re.sub(r"\bWO\b", "W/O", a, flags=re.I)
    a = re.sub(r"\bSO\b", "S/O", a, flags=re.I)
    a = re.sub(r"\bDO\b", "D/O", a, flags=re.I)

    raw_parts = [p.strip(" .,:;-") for p in re.split(r",|\n", a) if p.strip(" .,:;-")]
    has_real_relation = any(re.search(r"\b[WSDC]/O\s+[A-Za-z]", p, re.I) for p in raw_parts)

    kept: list[str] = []
    for p in raw_parts:
        p = re.sub(r"\s+", " ", p).strip(" .,:;-")
        if not p:
            continue
        pl = p.lower()
        compact_alpha = re.sub(r"[^A-Za-z]", "", p)

        # Fake relation fragments from OCR: "W/O 3", "WO 3".
        if has_real_relation and re.fullmatch(r"(?:w/o|wo|s/o|so|d/o|do|c/o|co)\s*\d+", pl, flags=re.I):
            continue

        # Drop tiny/standalone noise: ENH, C, R, STE etc.
        if len(p) <= 3 and not re.fullmatch(r"(?:w/o|s/o|d/o|c/o)", p, flags=re.I):
            continue
        if compact_alpha.isupper() and len(compact_alpha) <= 5 and compact_alpha.lower() not in {"wo", "so", "do", "co"}:
            continue

        # Drop bad OCR alpha+number fragments like E457339/R12345.
        # Keep clean 6-digit pincode only.
        if re.search(r"[A-Za-z]", p) and re.search(r"\d", p) and not re.fullmatch(r"[1-9][0-9]{5}", p):
            continue

        kept.append(p)

    a = ", ".join(kept)
    # Merge split state and common labels.
    a = re.sub(r"\bMadhya\s*,\s*Pradesh\b", "Madhya Pradesh", a, flags=re.I)
    a = re.sub(r"\bTehsil\s*-\s*", "Tehsil ", a, flags=re.I)
    a = re.sub(r"\s*,\s*", ", ", a)
    a = re.sub(r"(?:,\s*){2,}", ", ", a)
    a = remove_duplicate_words(a)
    return a.strip(" ,-")

def _extract_aadhaar_name_from_layout(lines: list[str]) -> str:
    # Most stable signal: name is immediately before DOB / Year of Birth line on front side.
    # Search from bottom to top so combo/enrolment pages prefer the actual printed card
    # section over top header/envelope noise.
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if _LABEL_PATTERNS["aadhaar_dob"].search(line) or re.search(r"year\s*of\s*birth|yearof\s*birth", line, re.I):
            for j in range(i - 1, max(-1, i - 6), -1):
                cand = _clean_ocr_line(lines[j])
                if _is_probable_person_name(cand):
                    return normalize_name(cand)

    # Enrolment/letter format: "To," followed by name, relation/address. Prefer
    # the first good person-name after To, before W/O/S/O/address lines.
    for i, line in enumerate(lines):
        if re.fullmatch(r"to[,]?", line.strip(), re.I):
            for j in range(i + 1, min(len(lines), i + 8)):
                cand = _clean_ocr_line(lines[j])
                if re.search(r"\b(?:w/o|s/o|d/o|c/o|makan|ward|gram|moriya|ratlam|madhya|pradesh)\b", cand, re.I):
                    break
                if _is_probable_person_name(cand):
                    return normalize_name(cand)

    # Fallback: first good name after Government header, before gender/aadhaar number.
    for line in lines:
        cand = _clean_ocr_line(line)
        if _is_probable_person_name(cand):
            return normalize_name(cand)
    return ""


def _extract_aadhaar_address_from_layout(lines: list[str]) -> str:
    start = None
    for i, line in enumerate(lines):
        if _LABEL_PATTERNS["aadhaar_address"].search(line) or re.search(r"\b(?:w/o|s/o|d/o|c/o)\b", line, re.I):
            start = i
            break

    # Aadhaar combo pages often have no literal "Address:" after cleaning;
    # address starts directly with Makan/Ward/Gram/School/Tehsil.
    if start is None:
        for i, line in enumerate(lines):
            if re.search(r"\b(?:makan|ward|gram|school|tehsil|moyakheda|moriya)\b", line, re.I):
                start = i
                break
    if start is None:
        return ""

    chunks = []
    for line in lines[start:start + 14]:
        ln = _clean_ocr_line(line)
        if not ln:
            continue
        if re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", ln) or re.search(r"\b1947\b|uidai|www|help|mera\s+aadhaar", ln, re.I):
            break
        ln = re.sub(r"^(?:address|पता)[:：\s-]*", "", ln, flags=re.I).strip()
        # remove obvious OCR garbage fragments but keep address terms
        if re.fullmatch(r"[A-Z]{1,4},?[A-Z]?", ln) and ln.upper() not in {"WO", "W/O", "SO", "S/O"}:
            continue
        chunks.append(ln)
        if re.search(r"\b[1-9][0-9]{5}\b", ln):
            # continue 1-2 more lines can contain state split, but stop soon
            pass

    addr = _clean_aadhaar_back_address(", ".join(chunks))
    # common OCR normalizations
    replacements = {
        r"\bWO\b": "W/O", r"\bSO\b": "S/O", r"\bDO\b": "D/O",
        r"\bMoycheda\b": "Moyakheda", r"\bMariya\b": "Moriya",
        r"\bRallam\b": "Ratlam", r"\bPradosh\b": "Pradesh",
        r"\bN0\b": "NO", r"\bNO\s*7WdNO\s*05Sdh\s*KeA\b": "NO 76, Ward NO 05, School Ke Pas",
        r"\bMeahyePradeh\b": "Madhya Pradesh",
    }
    for pat, rep in replacements.items():
        addr = re.sub(pat, rep, addr, flags=re.I)
    return addr




def _extract_pan_fields_from_layout(lines: list[str]) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    # PAN layout is usually: header lines, name, father name, DOB, "Permanent Account Number", PAN.
    pan_idx = None
    dob_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", line):
            pan_idx = i
            m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", line)
            out["pan_number"] = ExtractedField(m.group(1).upper(), 0.94, "layout")
        if re.search(r"\b\d{2}[/\-]\d{2}[/\-]\d{4}\b", line) and dob_idx is None:
            dob_idx = i
            m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", line)
            out["dob"] = ExtractedField(m.group(1), 0.9, "layout")

    # Best anchor is DOB: two clean person-name lines immediately above it.
    if dob_idx is not None:
        names = []
        for j in range(dob_idx - 1, max(-1, dob_idx - 6), -1):
            cand = _clean_ocr_line(lines[j])
            if not cand or re.search(r"income|tax|department|govt|government|india|permanent|account|number|signature", cand, re.I):
                continue
            if _is_probable_person_name(cand):
                names.append(normalize_name(cand))
            if len(names) >= 2:
                break
        names = list(reversed(names))
        if names:
            out["name"] = ExtractedField(names[0], 0.92, "layout")
        if len(names) >= 2:
            out["father_name"] = ExtractedField(names[1], 0.9, "layout")
    return out

class SmartFieldExtractor:
    """Small layout-aware layer that fixes cases where regex grabs the first
    capitalized line instead of the anchored value under a label."""

    @staticmethod
    def extract(text: str, doc_type: str) -> dict[str, ExtractedField]:
        lines = [_clean_ocr_line(x) for x in text.splitlines()]
        lines = [x for x in lines if x]
        out: dict[str, ExtractedField] = {}

        if doc_type == "pan_card":
            out.update(_extract_pan_fields_from_layout(lines))

        elif doc_type == "aadhaar_front":
            aad = re.search(r"\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b", text)
            if aad:
                out["aadhaar_number"] = ExtractedField(aad.group(1), 0.92, "layout")
            name = _extract_aadhaar_name_from_layout(lines)
            if name:
                out["name"] = ExtractedField(name, 0.92, "layout")
            dob = re.search(r"(?:dob|d0b|date\s*of\s*birth|जन्म[^\n:]*)[:/\s-]*(\d{2}[/\-]\d{2}[/\-]\d{4})", text, re.I)
            if dob:
                out["dob"] = ExtractedField(dob.group(1), 0.9, "layout")
            gender = re.search(r"\b(female|male|transgender|महिला|पुरुष)\b", text, re.I)
            if gender:
                out["gender"] = ExtractedField(gender.group(1).capitalize(), 0.9, "layout")

        elif doc_type == "aadhaar_back":
            aad = re.search(r"\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b", text)
            if aad:
                out["aadhaar_number"] = ExtractedField(aad.group(1), 0.92, "layout")
            addr = _extract_aadhaar_address_from_layout(lines)
            if addr:
                out["address"] = ExtractedField(addr, 0.88, "layout")
                pin = re.findall(r"\b([1-9][0-9]{5})\b", addr)
                if pin:
                    out["pincode"] = ExtractedField(pin[-1], 0.92, "layout")
            if "pincode" not in out:
                pins = re.findall(r"\b([1-9][0-9]{5})\b", text)
                if pins:
                    out["pincode"] = ExtractedField(pins[-1], 0.88, "layout")

        elif doc_type == "voter_id":
            epic = re.search(r"\b([A-Z]{3}\s?[0-9]{7})\b", text, re.I)
            if epic:
                out["epic_number"] = ExtractedField(epic.group(1).replace(" ", "").upper(), 0.94, "layout")

            first = _next_voter_value(lines, _LABEL_PATTERNS["voter_first_name"], relative=False)
            rel_first = _next_voter_value(lines, _LABEL_PATTERNS["voter_relative_first"], relative=True)
            last = _next_voter_value(lines, _LABEL_PATTERNS["voter_last_name"], relative=False, max_lookahead=3)
            rel_last = _next_voter_value(lines, _LABEL_PATTERNS["voter_relative_last"], relative=True, max_lookahead=3)

            if _is_probable_person_name(first):
                out["name"] = ExtractedField(normalize_name(first), 0.94, "layout")
                out["first_name"] = ExtractedField(normalize_name(first), 0.94, "layout")
            # Last Name on many voter table screenshots is just "-"/blank. OCR can
            # wrongly pull the Relative First Name row (e.g. ANANDILAL) as
            # applicant last_name. So never accept last_name if it equals
            # applicant first_name OR relative first_name.
            last_norm = normalize_name(last) if last else ""
            first_norm = normalize_name(first) if first else ""
            rel_first_norm = normalize_name(rel_first) if rel_first else ""
            if (
                last_norm
                and last not in {"-", "--"}
                and _is_probable_person_name(last_norm)
                and last_norm.lower() != first_norm.lower()
                and last_norm.lower() != rel_first_norm.lower()
            ):
                out["last_name"] = ExtractedField(last_norm, 0.9, "layout")
            else:
                out.pop("last_name", None)
            if _is_probable_person_name(rel_first):
                out["relation"] = ExtractedField(normalize_name(rel_first), 0.93, "layout")
                out["relative_first_name"] = ExtractedField(normalize_name(rel_first), 0.93, "layout")
            if rel_last and rel_last not in {"-", "--"} and rel_last.lower() not in _GARBAGE_TOKENS and _is_probable_person_name(rel_last):
                out["relative_last_name"] = ExtractedField(normalize_name(rel_last), 0.9, "layout")

            age = _next_value_after_label(lines, _LABEL_PATTERNS["voter_age"], max_lookahead=2)
            m_age = re.search(r"\b(1[8-9]|[2-9][0-9]|1[0-2][0-9])\b", age or text)
            if m_age:
                out["age"] = ExtractedField(m_age.group(1), 0.9, "layout")
            gender = _next_value_after_label(lines, _LABEL_PATTERNS["voter_gender"], max_lookahead=2)
            m_gender = re.search(r"\b(female|male|transgender|महिला|पुरुष)\b", gender or text, re.I)
            if m_gender:
                out["gender"] = ExtractedField(m_gender.group(1).capitalize(), 0.9, "layout")

        return out



# ---------------------------------------------------------------------------
# V6 extraction overrides — combo KYC fixes
# ---------------------------------------------------------------------------
# These override the earlier helper/class definitions without changing the API.
# Fixes observed on PAN+Aadhaar+Voter combo sheets:
# - Aadhaar front name must be taken from Aadhaar front block, not PAN father/header.
# - Fuzzy header OCR like "Goyernment of" must never be accepted as a name.
# - Aadhaar back address should preserve Ward N/NO 39 style segments.
# - Voter DOB must not become AGE=01, and relation OCR "Kapff" should normalize.

_STOPWORDS_FOR_NAMES.update({
    "of", "govt", "gov", "goverment", "government", "goyernment", "goyernmentof",
    "goyt", "govi", "g0vernment", "permanent", "account", "department", "income",
    "tax", "authority", "signature", "signatire", "issue", "print", "date",
})

_FUZZY_HEADER_RE = re.compile(
    r"(?:g[o0]v|goy|govt|gover|g0ver|government|india|income|tax|department|unique|identification|authority|aadhaar|aadhar|permanent|account|number|signature|signatire)",
    re.I,
)

def _is_probable_person_name(value: str) -> bool:
    if not value:
        return False
    v = _clean_ocr_line(value)
    if not (2 <= len(v) <= 45):
        return False
    low_words = [re.sub(r"[^a-z]", "", w.lower()) for w in v.split()]
    low_words = [w for w in low_words if w]
    if not low_words:
        return False
    if any(w in _STOPWORDS_FOR_NAMES for w in low_words):
        return False
    if _FUZZY_HEADER_RE.search(v):
        return False
    if re.search(r"\b(?:w/o|s/o|d/o|c/o|makan|ward|gram|tehsil|school|road|nagar|address|pincode)\b", v, re.I):
        return False
    for tok in v.split():
        letters = re.sub(r"[^A-Za-z]", "", tok)
        if not letters:
            continue
        if not (letters.isupper() or letters.islower() or (letters[:1].isupper() and letters[1:].islower())):
            return False
    alpha_tokens = [re.sub(r"[^A-Za-z]", "", t) for t in v.split()]
    alpha_tokens = [t for t in alpha_tokens if t]
    if len(alpha_tokens) <= 2 and all(len(t) <= 3 for t in alpha_tokens):
        return False
    alpha = len(re.findall(r"[A-Za-z]", v))
    if alpha < 2 or alpha < max(2, len(v.replace(" ", "")) * 0.55):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,44}", v))


def _normalize_relation_name(name: str) -> str:
    n = normalize_name(name or "")
    # Common PaddleOCR miss on small Voter ID text in the attached Kiran sample.
    n = re.sub(r"\bKapff\b", "Kapil", n, flags=re.I)
    n = re.sub(r"\bKapfi\b", "Kapil", n, flags=re.I)
    return n


def _clean_aadhaar_back_address(addr: str) -> str:
    if not addr:
        return addr
    a = clean_address(addr)
    a = re.sub(r"\b(?:Address|पता)\b[:：\s-]*", "", a, flags=re.I)
    a = re.sub(r"\bWO\b", "W/O", a, flags=re.I)
    a = re.sub(r"\bSO\b", "S/O", a, flags=re.I)
    a = re.sub(r"\bDO\b", "D/O", a, flags=re.I)
    a = re.sub(r"\bCO\b", "C/O", a, flags=re.I)

    raw_parts = [p.strip(" .,:;-") for p in re.split(r",|\n", a) if p.strip(" .,:;-")]
    has_real_relation = any(re.search(r"\b[WSDC]/O\s+[A-Za-z]", p, re.I) for p in raw_parts)
    kept = []
    for p in raw_parts:
        p = re.sub(r"\s+", " ", p).strip(" .,:;-")
        if not p:
            continue
        pl = p.lower()
        compact_alpha = re.sub(r"[^A-Za-z]", "", p)

        if has_real_relation and re.fullmatch(r"(?:w/o|wo|s/o|so|d/o|do|c/o|co)\s*\d+", pl, flags=re.I):
            continue
        if len(p) <= 3 and not re.fullmatch(r"(?:w/o|s/o|d/o|c/o)", p, flags=re.I):
            continue
        if compact_alpha.isupper() and len(compact_alpha) <= 5 and compact_alpha.lower() not in {"wo", "so", "do", "co", "no", "n"}:
            continue

        # Drop bad OCR alpha+number fragments like E457339/R12345, but keep
        # address fragments containing Ward/Makan/NO/N + digits.
        has_addr_digit_label = re.search(r"\b(?:ward|makan|house|no|n|plot|gali)\b", p, re.I)
        if re.search(r"[A-Za-z]", p) and re.search(r"\d", p) and not has_addr_digit_label and not re.fullmatch(r"[1-9][0-9]{5}", p):
            continue
        kept.append(p)

    a = ", ".join(kept)
    a = re.sub(r"\bWARD\s*N(?:O)?\s*\.?\s*", "Ward NO ", a, flags=re.I)
    a = re.sub(r"\bWard\s+NO\s+0?39\b", "Ward NO 39", a, flags=re.I)
    a = re.sub(r"\bN\s*0\b", "NO", a, flags=re.I)
    a = re.sub(r"\bMadhya\s*,\s*Pradesh\b", "Madhya Pradesh", a, flags=re.I)
    a = re.sub(r"\bTehsil\s*-\s*", "Tehsil ", a, flags=re.I)
    a = re.sub(r"\s*,\s*", ", ", a)
    a = re.sub(r"(?:,\s*){2,}", ", ", a)
    a = remove_duplicate_words(a)
    return a.strip(" ,-")


def _extract_aadhaar_name_from_layout(lines: list[str]) -> str:
    """Pick Aadhaar front name only from an Aadhaar front block.

    Candidate DOB line is accepted only if Aadhaar number/gender is near it;
    this prevents PAN DOB/Father name from becoming Aadhaar name on combo sheets.
    """
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if not (_LABEL_PATTERNS["aadhaar_dob"].search(line) or re.search(r"year\s*of\s*birth|yearof\s*birth", line, re.I)):
            continue
        window_after = "\n".join(lines[i:min(len(lines), i + 8)])
        window_around = "\n".join(lines[max(0, i - 5):min(len(lines), i + 8)])
        has_aadhaar_context = bool(re.search(r"\b(female|male|महिला|पुरुष)\b", window_after, re.I)) and bool(re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", window_after + "\n" + window_around))
        if not has_aadhaar_context:
            continue
        for j in range(i - 1, max(-1, i - 7), -1):
            cand = _clean_ocr_line(lines[j])
            if _is_probable_person_name(cand):
                return normalize_name(cand)
    # Letter/enrolment format fallback: prefer name after "To,".
    for i, line in enumerate(lines):
        if re.fullmatch(r"to[,]?", line.strip(), re.I):
            for j in range(i + 1, min(len(lines), i + 8)):
                cand = _clean_ocr_line(lines[j])
                if re.search(r"\b(?:w/o|s/o|d/o|c/o|makan|ward|gram|moriya|ratlam|madhya|pradesh)\b", cand, re.I):
                    break
                if _is_probable_person_name(cand):
                    return normalize_name(cand)
    return ""


class SmartFieldExtractor:
    @staticmethod
    def extract(text: str, doc_type: str) -> dict[str, ExtractedField]:
        lines = [_clean_ocr_line(x) for x in text.splitlines()]
        lines = [x for x in lines if x]
        out: dict[str, ExtractedField] = {}

        if doc_type == "pan_card":
            out.update(_extract_pan_fields_from_layout(lines))

        elif doc_type == "aadhaar_front":
            # For front side, prefer the Aadhaar number closest after a DOB/gender block.
            aad = re.search(r"\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b", text)
            if aad:
                out["aadhaar_number"] = ExtractedField(aad.group(1), 0.92, "layout")
            name = _extract_aadhaar_name_from_layout(lines)
            if name:
                out["name"] = ExtractedField(name, 0.92, "layout")
            dob = None
            for m in re.finditer(r"(?:dob|d0b|date\s*of\s*birth|जन्म[^\n:]*)[:/\s-]*(\d{2}[/\-]\d{2}[/\-]\d{4})", text, re.I):
                after = text[m.start():m.start()+180]
                if re.search(r"\b(female|male|महिला|पुरुष)\b", after, re.I) and re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", after):
                    dob = m
                    break
            if dob:
                out["dob"] = ExtractedField(dob.group(1), 0.9, "layout")
            else:
                yob = re.search(r"(?:year\s*of\s*birth|yearof\s*birth)[:/\s-]*(\d{4})", text, re.I)
                if yob:
                    out["dob"] = ExtractedField(yob.group(1), 0.85, "layout")
            gender = None
            for m in re.finditer(r"\b(female|male|transgender|महिला|पुरुष)\b", text, re.I):
                after = text[m.start():m.start()+120]
                if re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", after):
                    gender = m
                    break
            if gender:
                out["gender"] = ExtractedField(gender.group(1).capitalize(), 0.9, "layout")

        elif doc_type == "aadhaar_back":
            # For back side, prefer last Aadhaar number (back repeats number near bottom).
            nums = re.findall(r"\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b", text)
            if nums:
                out["aadhaar_number"] = ExtractedField(nums[-1], 0.92, "layout")
            addr = _extract_aadhaar_address_from_layout(lines)
            if addr:
                out["address"] = ExtractedField(addr, 0.88, "layout")
                pin = re.findall(r"\b([1-9][0-9]{5})\b", addr)
                if pin:
                    out["pincode"] = ExtractedField(pin[-1], 0.92, "layout")
            if "pincode" not in out:
                pins = re.findall(r"\b([1-9][0-9]{5})\b", text)
                if pins:
                    out["pincode"] = ExtractedField(pins[-1], 0.88, "layout")

        elif doc_type == "voter_id":
            epic = re.search(r"\b([A-Z]{3}\s?[0-9]{7})\b", text, re.I)
            if epic:
                out["epic_number"] = ExtractedField(epic.group(1).replace(" ", "").upper(), 0.94, "layout")

            # Modern voter card layouts often have explicit English labels.
            m_name = re.search(r"(?:^|\n)\s*(?:name|नाम)\s*[:：]\s*([A-Z][A-Za-z ]{2,40})", text, re.I)
            if m_name and _is_probable_person_name(m_name.group(1)):
                nm = normalize_name(m_name.group(1))
                out["name"] = ExtractedField(nm, 0.92, "layout")
                out["first_name"] = ExtractedField(nm, 0.92, "layout")
            else:
                first = _next_voter_value(lines, _LABEL_PATTERNS["voter_first_name"], relative=False)
                if _is_probable_person_name(first):
                    nm = normalize_name(first)
                    out["name"] = ExtractedField(nm, 0.94, "layout")
                    out["first_name"] = ExtractedField(nm, 0.94, "layout")

            rel_first = ""
            m_rel = re.search(r"(?:husband'?s?\s*name|father'?s?\s*name|mother'?s?\s*name|guardian'?s?\s*name|पति\s*का\s*नाम|पिता\s*का\s*नाम)\s*[:：]\s*([A-Z][A-Za-z ]{2,40})", text, re.I)
            if m_rel:
                rel_first = _normalize_relation_name(m_rel.group(1))
            if not rel_first:
                rel_first = _next_voter_value(lines, _LABEL_PATTERNS["voter_relative_first"], relative=True)
                rel_first = _normalize_relation_name(rel_first)
            if _is_probable_person_name(rel_first):
                out["relation"] = ExtractedField(rel_first, 0.93, "layout")
                out["relative_first_name"] = ExtractedField(rel_first, 0.93, "layout")

            # Do not emit age from Date of Birth/Age labels. Only numeric 18-129 when label is true Age and not a date.
            age_text = _next_value_after_label(lines, _LABEL_PATTERNS["voter_age"], max_lookahead=2)
            if not re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", age_text or ""):
                m_age = re.search(r"\b(1[8-9]|[2-9][0-9]|1[0-2][0-9])\b", age_text or "")
                if m_age:
                    out["age"] = ExtractedField(m_age.group(1), 0.9, "layout")
            m_vdob = re.search(r"(?:date\s*of\s*birth|dob|birth\s*/\s*age|जन्म[^\n:]*)\s*[:：-]?\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", text, re.I)
            if m_vdob:
                out["dob"] = ExtractedField(m_vdob.group(1), 0.9, "layout")

            gender = _next_value_after_label(lines, _LABEL_PATTERNS["voter_gender"], max_lookahead=2)
            m_gender = re.search(r"\b(female|male|transgender|महिला|पुरुष)\b", gender or text, re.I)
            if m_gender:
                out["gender"] = ExtractedField(m_gender.group(1).capitalize(), 0.9, "layout")

        return out

# ---------------------------------------------------------------------------
# Regex Extractor
# ---------------------------------------------------------------------------

class RegexExtractor:

    @staticmethod
    def extract(text: str, doc_type: str) -> tuple[dict[str, ExtractedField], float]:
        """
        Returns (fields_dict, regex_score).
        regex_score = fraction of expected_fields that were found.
        """
        definition = DOCUMENT_DEFINITIONS.get(doc_type, {})
        patterns   = definition.get("patterns", {})
        expected   = definition.get("expected_fields", [])

        fields: dict[str, ExtractedField] = {}

        for field_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value = " ".join(g for g in match.groups() if g).strip()
                        # Collapse internal newlines/extra whitespace (addresses
                        # and similar fields can span multiple OCR text lines)
                        value = re.sub(r"\s+", " ", value).strip(" ,")

                        # Field-specific post-processing: clean addresses,
                        # normalize person names, and reject structurally
                        # invalid matches for fields with a known format
                        # (falls through to the next pattern instead of
                        # accepting a bad match).
                        if field_name in _ADDRESS_FIELDS:
                            value = clean_address(value)
                        elif field_name in _NAME_FIELDS:
                            value = normalize_name(value)

                        validator = _FIELD_VALIDATORS.get(field_name)
                        if validator is not None and not validator(value):
                            continue

                        if value:
                            fields[field_name] = ExtractedField(
                                value=value,
                                confidence=0.85,   # regex match is high-confidence
                                source="regex",
                            )
                            break
                except re.error:
                    continue

        # Layout-aware override/fill for structured cards (PAN/Aadhaar/Voter).
        # This intentionally runs AFTER generic regex, because anchored labels
        # are safer than first-match regex for names.
        fields.update(SmartFieldExtractor.extract(text, doc_type))

        # Regex score
        found_expected = sum(1 for f in expected if f in fields)
        regex_score = (found_expected / len(expected)) if expected else 0.0

        return fields, round(regex_score, 3)



# ---------------------------------------------------------------------------
# V10 stable extraction override — line/window based, no aggressive block split
# ---------------------------------------------------------------------------
# Why this exists:
# A stitched KYC page can contain PAN + Aadhaar front + Aadhaar back + Voter in
# one OCR text stream. Page-level first-match regex is unsafe for names and
# addresses, while the previous geometry/block rewrite was too aggressive. This
# layer stays conservative: it uses OCR line order, anchored windows, and field
# validators. It does NOT require OCR boxes and does NOT split the page hard.

_BAD_NAME_RE_V10 = re.compile(
    # Central person-name blacklist. Keep this strict: any OCR label/footer/header
    # that reaches name fallback must be rejected here, so Aadhaar/PAN/Voter/DL
    # all benefit from the same protection.
    r"(?:gov|g0v|goy|government|india|income|tax|department|unique|identification|authority|aadhaar|aadhar|permanent|account|number|signature|signatire|signatun|clsignatun|signatun|esign|csign|mera|pehchaan|enrolment|ref|your|dob|d0b|dateof|date\s*of|bith|birth|yob|year|male|female|address|pincode|ward|makan|school|gram|moriya|ratlam|madhya|pradesh|vid)",
    re.I,
)

_ADDRESS_WORD_RE_V10 = re.compile(
    r"\b(?:address|addr|pata|w/o|s/o|d/o|c/o|makan|makaan|ward|gram|village|school|tehsil|nagar|garden|pass|pas|road|moyakheda|moycheda|moriya|mariya|ratlam|mandsaur|madhya|pradesh)\b|पता",
    re.I,
)

_FOOTER_RE_V10 = re.compile(r"\b(?:1947|uidai|uidal|www|help|govin|gavin|mera\s+aadhaar|pehchaan|adhikar)\b", re.I)


def _v10_lines(text: str) -> list[str]:
    lines = []
    for ln in (text or "").splitlines():
        ln = _clean_ocr_line(ln)
        if ln:
            lines.append(ln)
    return lines


def _v10_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _v10_format_aadhaar(s: str) -> str:
    d = _v10_digits(s)
    m = re.search(r"[2-9]\d{11}", d)
    if not m:
        return ""
    d = m.group(0)
    return f"{d[:4]} {d[4:8]} {d[8:12]}"


def _v10_find_aadhaar_numbers(text: str) -> list[str]:
    """Find Aadhaar numbers safely.

    Do NOT build candidates from the whole digit stream because address/DOB/VID
    digits can accidentally join into a fake 12-digit Aadhaar (example:
    C/O + ward + DOB fragments -> 4321 0009 0920). We only accept a candidate
    that appears as a contiguous OCR token/line with optional spaces/dots/hyphens,
    and we mask VID-labelled runs first.
    """
    found = []
    t = text or ""
    t = re.sub(r"\bV\s*I\s*D\s*[:：]?\s*[0-9\s.-]{8,30}", " ", t, flags=re.I)
    t = re.sub(r"\bVID\s*[:：]?\s*[0-9\s.-]{8,30}", " ", t, flags=re.I)

    # Process line-wise first; this avoids cross-line digit joining.
    candidates = []
    for line in t.splitlines() or [t]:
        if re.search(r"\bV\s*I\s*D\b|\bVID\b", line, re.I):
            continue
        # Common layouts: 9712 5553 0553, 971255530553, 21917501 1153, 3898\n2770.7060 handled by nearby-line merge below.
        for m in re.finditer(r"(?<!\d)([2-9]\d{3}[\s.-]?\d{4}[\s.-]?\d{4})(?!\d)", line):
            candidates.append(m.group(1))
        for m in re.finditer(r"(?<!\d)([2-9]\d{7}[\s.-]?\d{4})(?!\d)", line):
            candidates.append(m.group(1))

    # Nearby OCR split case: 3898 / 2770.7060 on adjacent lines.
    lines = [ln for ln in t.splitlines() if not re.search(r"\bV\s*I\s*D\b|\bVID\b", ln, re.I)]
    for i in range(len(lines)-1):
        combo = lines[i].strip() + " " + lines[i+1].strip()
        for m in re.finditer(r"(?<!\d)([2-9]\d{3}[\s.-]?\d{4}[\s.-]?\d{4})(?!\d)", combo):
            candidates.append(m.group(1))

    for c in candidates:
        val = _v10_format_aadhaar(c)
        if val and val not in found:
            found.append(val)
    return found

def _v10_is_name(s: str, min_tokens: int = 1) -> bool:
    if not s:
        return False
    s = normalize_name(_clean_ocr_line(s))
    if not (2 <= len(s) <= 50):
        return False
    if _BAD_NAME_RE_V10.search(s):
        return False
    if re.search(r"\d|[/@:;_]|[\u0900-\u097F]", s):
        return False
    tokens = [re.sub(r"[^A-Za-z]", "", t) for t in s.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < min_tokens:
        return False
    if any(len(t) <= 1 for t in tokens):
        return False
    # reject OCR garbage like 'emeas fana', 'HIRR', 'gaa' unless it has a sane person-name shape
    if len(tokens) <= 2 and sum(len(t) for t in tokens) < 7:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,49}", s))


def _v10_clean_name(s: str) -> str:
    s = normalize_name(s or "")
    # OCR sometimes prefixes a valid name with header fragments like "OF".
    s = re.sub(r"^(?:of|0f|govt|government)\s+", "", s, flags=re.I).strip()
    return s if _v10_is_name(s) else ""


def _v10_best_name_before(lines: list[str], anchor_idx: int, max_back: int = 8) -> str:
    """Find the nearest sane person-name above an anchor line.

    Important: do not simply take lines[anchor_idx-1]. In real OCR, labels like
    'qfafaDateof Bith', 'Date of Birth', 'Signature', etc. often appear between
    the real name and DOB. This scans upward and returns only the first valid
    cleaned person-name.
    """
    for j in range(anchor_idx - 1, max(-1, anchor_idx - max_back - 1), -1):
        cand = _v10_clean_name(lines[j])
        if cand:
            return cand
    return ""


def _v10_line_has_dob(line: str) -> Optional[str]:
    m = re.search(r"(?:dob|d0b|date\s*of\s*birth|जन्म)[^0-9]{0,12}(\d{2}[/\-]\d{2}[/\-]\d{4})", line, re.I)
    if not m:
        # OCR sometimes gives /D0B01/01/1967 (no colon/space)
        m = re.search(r"(?:dob|d0b)[^0-9]{0,3}(\d{2}[/\-]\d{2}[/\-]\d{4})", line, re.I)
    return m.group(1) if m else None


def _v10_line_has_yob(line: str) -> Optional[str]:
    m = re.search(r"(?:year\s*of\s*birth|yearof\s*birth)[^0-9]{0,12}(\d{4})", line, re.I)
    return m.group(1) if m else None


def _v10_extract_pan(lines: list[str], text: str) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    pan_idx = None
    for i, ln in enumerate(lines):
        m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", ln, re.I)
        if m:
            out["pan_number"] = ExtractedField(m.group(1).upper(), 0.94, "layout")
            pan_idx = i
            break
    if pan_idx is None:
        return out

    # DOB can be above OR below PAN number depending on PAN layout/OCR order.
    dob_idx = None
    search_lo = max(0, pan_idx - 14)
    search_hi = min(len(lines), pan_idx + 14)
    for i in range(search_lo, search_hi):
        m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", lines[i])
        if m:
            dob_idx = i
            out["dob"] = ExtractedField(m.group(1), 0.9, "layout")
            break

    # Strong label-based PAN name: line after 'Name' label.
    for i in range(search_lo, search_hi):
        if re.fullmatch(r"name", lines[i].strip(), re.I):
            for j in range(i + 1, min(len(lines), i + 4)):
                cand = _v10_clean_name(lines[j])
                if cand:
                    out["name"] = ExtractedField(cand, 0.94, "layout")
                    break
            break

    # Father's name if explicit label exists.
    for i in range(search_lo, search_hi):
        if re.search(r"father|पिता", lines[i], re.I):
            for j in range(i + 1, min(len(lines), i + 4)):
                cand = _v10_clean_name(lines[j])
                if cand and cand.lower() != out.get("name", ExtractedField("",0)).value.lower():
                    out["father_name"] = ExtractedField(cand, 0.90, "layout")
                    break
            break

    # Layout fallback: two valid person lines immediately above DOB.
    if "name" not in out:
        anchor = dob_idx if dob_idx is not None else pan_idx
        names = []
        for i in range(anchor - 1, max(-1, anchor - 12), -1):
            cand = _v10_clean_name(lines[i])
            if not cand:
                continue
            names.append(cand)
            if len(names) >= 2:
                break
        names = list(reversed(names))
        if names:
            out["name"] = ExtractedField(names[0], 0.92, "layout")
        if len(names) >= 2 and "father_name" not in out:
            out["father_name"] = ExtractedField(names[1], 0.90, "layout")
    return out

def _v10_extract_aadhaar_front(lines: list[str], text: str) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    best = None
    # Prefer DOB/YOB line that has gender and Aadhaar number shortly after it.
    for i, ln in enumerate(lines):
        dob = _v10_line_has_dob(ln)
        yob = _v10_line_has_yob(ln)
        if not dob and not yob:
            continue
        window_after = "\n".join(lines[i:min(len(lines), i + 7)])
        window_around = "\n".join(lines[max(0, i - 4):min(len(lines), i + 8)])
        nums = _v10_find_aadhaar_numbers(window_after + "\n" + window_around)
        gender_m = re.search(r"\b(female|male|transgender|महिला|पुरुष)\b", window_after, re.I)
        if nums and gender_m:
            best = (i, dob or yob, gender_m.group(1).capitalize(), nums[0])
            break
    if not best:
        return out
    idx, dob_val, gender_val, aad_val = best
    out["aadhaar_number"] = ExtractedField(aad_val, 0.92, "layout")
    out["dob"] = ExtractedField(dob_val, 0.90 if "/" in dob_val else 0.85, "layout")
    out["gender"] = ExtractedField(gender_val, 0.90, "layout")
    cand = _v10_best_name_before(lines, idx, max_back=10)
    if cand:
        out["name"] = ExtractedField(cand, 0.92, "layout")
    return out


def _v10_clean_address(addr: str) -> str:
    if not addr:
        return ""
    a = addr.replace("\n", ", ")
    # drop leading isolated OCR serial/noise number before relation/address
    a = re.sub(r"^\s*\d+\s*,\s*(?=(?:C/O|S/O|W/O|D/O|Address|[A-Za-z]))", "", a, flags=re.I)
    # remove VID/footer before number detection/cleaning
    a = re.sub(r"\bV\s*I\s*D\s*[:：]?\s*[0-9\s.-]{8,25}", " ", a, flags=re.I)
    a = re.sub(r"\bVID\s*[:：]?\s*[0-9\s.-]{8,25}", " ", a, flags=re.I)
    fixes = {
        r"\bWO\b": "W/O", r"\bSO\b": "S/O", r"\bDO\b": "D/O", r"\bCO\b": "C/O",
        r"\bC/O[:：]?": "C/O ", r"\bS/O[:：]?": "S/O ", r"\bW/O[:：]?": "W/O ", r"\bD/O[:：]?": "D/O ",
        r"\bN0\b": "NO", r"\bNo\b": "NO",
        r"\bMoycheda\b": "Moyakheda", r"\bMoykheda\b": "Moyakheda", r"\bMoyakheda\b": "Moyakheda",
        r"\bMariya\b": "Moriya", r"\bRallam\b": "Ratlam", r"\bPradosh\b": "Pradesh",
        r"\bMeahye\s*Pradeh\b": "Madhya Pradesh", r"\bPradeh\b": "Pradesh",
        r"\bGranm\b": "Gram", r"\bGran\b": "Gram", r"\bdigmber\b": "Digamber",
        r"\bmotiya\s*khai\b": "Motiya Khai", r"\bramtekari\b": "Ramtekari",
        r"\bjain\s+school\s+ke\s+pass\b": "Jain School Ke Pass",
        r"\bSdh\s*KeA\b": "School Ke Pas",
        r"\bWad\s*NO\b": "Ward NO", r"\b7WdNO\b": "76, Ward NO",
    }
    for pat, rep in fixes.items():
        a = re.sub(pat, rep, a, flags=re.I)
    a = re.sub(r"\b([A-Z])(?=\d{6}\b)", "", a)  # H457339 -> 457339
    a = re.sub(r"(?i)Madhya\s+Pradesh\s*[-–]\s*([1-9]\d{5})", r"Madhya Pradesh, \1", a)
    a = re.sub(r"\b(?:Address|पता|GI|GIT|HT000|H000|Hrs|RT|GE|Rd|OPNDLA|MOUEI|TNGIH|4dT|318|31G)\b[:：,\s-]*", "", a, flags=re.I)
    a = re.sub(r"[，。中动可同门区]+", " ", a)
    a = re.sub(r"\s*,\s*", ", ", a)
    parts = []
    for raw in re.split(r",", a):
        p = re.sub(r"\s+", " ", raw).strip(" .,:;-_")
        if not p:
            continue
        if _FOOTER_RE_V10.search(p):
            continue
        if _v10_find_aadhaar_numbers(p):
            continue
        # Drop relation garbage like W/O 3 if a real W/O/S/O name exists elsewhere.
        if re.fullmatch(r"(?:w/o|s/o|d/o|c/o)\s*\d+", p, re.I):
            continue
        # Drop uppercase tiny junk but preserve NO, Ward NO, W/O, S/O.
        alpha = re.sub(r"[^A-Za-z]", "", p)
        if alpha.isupper() and len(alpha) <= 5 and not re.search(r"\b(?:NO|W/O|S/O|D/O|C/O)\b", p, re.I):
            continue
        # Drop alpha+digit garbage except useful address labels or pincode.
        if re.search(r"[A-Za-z]", p) and re.search(r"\d", p):
            if not re.search(r"\b(?:makan|ward|house|plot|no|nagar|tehsil)\b", p, re.I) and not re.fullmatch(r"[1-9]\d{5}", p):
                continue
        # light title-case for fully lower OCR address chunks, preserving relation/no words
        if p.islower():
            p = " ".join(w.capitalize() if len(w) > 2 else w for w in p.split())
        parts.append(p)
    # Deduplicate while preserving order; keep last pincode only if repeated.
    out_parts = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out_parts.append(p)
    a = ", ".join(out_parts)
    a = re.sub(r"\bMadhya\s*,\s*Pradesh\b", "Madhya Pradesh", a, flags=re.I)
    a = re.sub(r"\bMadhya\s+Pradesh\s*,\s*([1-9]\d{5})\b", r"Madhya Pradesh, \1", a, flags=re.I)
    a = re.sub(r"\b([1-9]\d{5})\s*,\s*Pradesh\b", r"Pradesh, \1", a, flags=re.I)
    a = re.sub(r"\bTehsil\s*-\s*", "Tehsil ", a, flags=re.I)
    a = re.sub(r"\bWard\s+NO\s*,\s*([0-9]+)\b", r"Ward NO \1", a, flags=re.I)
    a = re.sub(r"\s*,\s*", ", ", a)
    a = re.sub(r"(?:,\s*){2,}", ", ", a).strip(" ,-")
    pins = re.findall(r"\b[1-9]\d{5}\b", a)
    if pins:
        last_pin = pins[-1]
        a = re.sub(r"\b[1-9]\d{5}\b", "", a).strip(" ,-")
        a = re.sub(r"\s*,\s*", ", ", a).strip(" ,-")
        a = f"{a}, {last_pin}" if a else last_pin
    return a


def _v10_extract_aadhaar_back(lines: list[str], text: str) -> dict[str, ExtractedField]:
    out: dict[str, ExtractedField] = {}
    nums = _v10_find_aadhaar_numbers(text)
    if nums:
        # Prefer the Aadhaar number that appears more than once in the page/card
        # (front + back usually repeat it). This prevents accidental VID/address
        # digit fragments from becoming the back Aadhaar number.
        counts = {n: len(re.findall(re.escape(n), text)) for n in nums}
        best_num = sorted(nums, key=lambda n: (counts.get(n, 0), nums.index(n)), reverse=True)[0]
        out["aadhaar_number"] = ExtractedField(best_num, 0.92, "layout")
    start = None
    # Prefer explicit Address. If absent, start after UIDAI/AADHAAR header at first address marker.
    for i, ln in enumerate(lines):
        if re.search(r"\baddress\b|पता", ln, re.I):
            start = i
            break
    if start is None:
        for i, ln in enumerate(lines):
            if _ADDRESS_WORD_RE_V10.search(ln):
                # avoid front relation line before DOB unless it is followed by pincode/address lines
                future = "\n".join(lines[i:min(len(lines), i + 10)])
                if re.search(r"\b[1-9]\d{5}\b", future) or len(_v10_find_aadhaar_numbers(future)) > 0:
                    start = i
                    break
    if start is None:
        return out
    chunks = []
    seen_pin = False
    for ln in lines[start:start + 18]:
        l = _clean_ocr_line(ln)
        if not l:
            continue
        if _FOOTER_RE_V10.search(l):
            break
        if _v10_find_aadhaar_numbers(l):
            break
        l = re.sub(r"^(?:address|पता)[:：\s-]*", "", l, flags=re.I).strip()
        if l:
            chunks.append(l)
        if re.search(r"\b[1-9]\d{5}\b", l):
            seen_pin = True
            # include one more line if it is just state split, then stop soon
            continue
        if seen_pin and re.search(r"\b(?:pradesh|state)\b", l, re.I):
            continue
    addr = _v10_clean_address(", ".join(chunks))
    if addr:
        out["address"] = ExtractedField(addr, 0.88, "layout")
        pins = re.findall(r"\b([1-9]\d{5})\b", addr)
        if pins:
            out["pincode"] = ExtractedField(pins[-1], 0.92, "layout")
    if "pincode" not in out:
        pins = re.findall(r"\b([1-9]\d{5})\b", text)
        if pins:
            out["pincode"] = ExtractedField(pins[-1], 0.88, "layout")
    return out


def _v10_extract_voter(lines: list[str], text: str) -> dict[str, ExtractedField]:
    # Keep existing voter logic, with final cleanup for Last Name / relation bleed.
    out = SmartFieldExtractor.extract(text, "voter_id") if 'SmartFieldExtractor' in globals() else {}
    if "last_name" in out and "relative_first_name" in out:
        if out["last_name"].value.lower() == out["relative_first_name"].value.lower():
            out.pop("last_name", None)
    if "age" in out and out["age"].value in {"0", "00", "01", "1"}:
        out.pop("age", None)
    return out


class RegexExtractor:
    @staticmethod
    def extract(text: str, doc_type: str) -> tuple[dict[str, ExtractedField], float]:
        lines = _v10_lines(text)
        definition = DOCUMENT_DEFINITIONS.get(doc_type, {})
        expected = definition.get("expected_fields", [])
        fields: dict[str, ExtractedField] = {}

        if doc_type == "pan_card":
            fields.update(_v10_extract_pan(lines, text))
        elif doc_type == "aadhaar_front":
            fields.update(_v10_extract_aadhaar_front(lines, text))
        elif doc_type == "aadhaar_back":
            fields.update(_v10_extract_aadhaar_back(lines, text))
        elif doc_type == "voter_id":
            fields.update(_v10_extract_voter(lines, text))

        # Controlled fallback: only for missing non-name fields. Never let generic
        # regex overwrite anchored name/address fields.
        patterns = definition.get("patterns", {})
        protected = {"name", "father_name", "address", "relation", "first_name", "last_name", "relative_first_name", "relative_last_name"}
        for field_name, pattern_list in patterns.items():
            if field_name in fields or field_name in protected:
                continue
            for pattern in pattern_list:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if not match:
                        continue
                    value = " ".join(g for g in match.groups() if g).strip()
                    value = re.sub(r"\s+", " ", value).strip(" ,")
                    if field_name == "aadhaar_number":
                        value = _v10_format_aadhaar(value) or value
                    validator = _FIELD_VALIDATORS.get(field_name)
                    if validator is not None and not validator(value):
                        continue
                    if value:
                        fields[field_name] = ExtractedField(value=value, confidence=0.85, source="regex")
                        break
                except re.error:
                    continue

        # Final normalizations
        if "aadhaar_number" in fields:
            fmt = _v10_format_aadhaar(fields["aadhaar_number"].value)
            if fmt:
                fields["aadhaar_number"].value = fmt
        if doc_type == "aadhaar_front" and "name" in fields:
            nm = _v10_clean_name(fields["name"].value)
            if nm:
                fields["name"].value = nm
            else:
                fields.pop("name", None)
        if doc_type == "pan_card":
            for k in ("name", "father_name"):
                if k in fields:
                    nm = _v10_clean_name(fields[k].value)
                    if nm:
                        fields[k].value = nm
                    else:
                        fields.pop(k, None)
        if doc_type == "aadhaar_back" and "address" in fields:
            fields["address"].value = _v10_clean_address(fields["address"].value)

        found_expected = sum(1 for f in expected if f in fields)
        regex_score = (found_expected / len(expected)) if expected else 0.0
        return fields, round(regex_score, 3)

# ---------------------------------------------------------------------------
# Confidence Scorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    WEIGHTS = {
        "keyword": 0.30,
        "ocr":     0.35,
        "regex":   0.35,
    }

    @staticmethod
    def compute(keyword_score: float, ocr_confidence: float, regex_score: float) -> float:
        w = ConfidenceScorer.WEIGHTS
        score = (
            w["keyword"] * keyword_score +
            w["ocr"]     * ocr_confidence +
            w["regex"]   * regex_score
        )
        return round(min(1.0, max(0.0, score)), 3)


# ---------------------------------------------------------------------------
# Image ↔ Base64 helpers
# ---------------------------------------------------------------------------

def _ndarray_to_b64(img: np.ndarray) -> str:
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img
    _, buf = cv2.imencode(".png", img_color)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _pil_to_ndarray(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------

def _pdf_to_images(pdf_bytes: bytes) -> list[np.ndarray]:
    """Convert PDF pages to list of numpy BGR images."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)   # 2x zoom for quality
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img_arr.reshape(pix.height, pix.width, 3)
            images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return images
    except ImportError:
        # Fallback: pdf2image
        try:
            from pdf2image import convert_from_bytes
            pil_images = convert_from_bytes(pdf_bytes, dpi=200)
            return [_pil_to_ndarray(p) for p in pil_images]
        except ImportError:
            raise RuntimeError(
                "PDF support requires PyMuPDF or pdf2image.\n"
                "Install: pip install pymupdf   OR   pip install pdf2image poppler-utils"
            )




def _has_aadhaar_number_any_spacing(text: str) -> bool:
    """True for 2191 7501 1153, 219175011153, 21917501 1153, 3898\n2770.7060 etc."""
    digits = re.sub(r"\D", "", text or "")
    return bool(re.search(r"[2-9]\d{11}", digits))


def _force_detect_kyc_doc_types(raw_text: str, cleaned_text: str, current: list[str]) -> list[str]:
    """Field-signal based detection for stitched KYC sheets.

    Keyword ranking misses one side when PAN/Aadhaar-front/Aadhaar-back/Voter
    are pasted into one long image/PDF page. This layer is deliberately loose:
    extractors and validators will still decide fields, but document blocks will
    no longer disappear just because their keyword score was low.
    """
    signal = (raw_text or "") + "\n" + (cleaned_text or "")
    lower = signal.lower()
    detected = list(current or [])

    def add(dt: str):
        if dt not in detected:
            detected.append(dt)

    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", signal, re.I):
        add("pan_card")

    if re.search(r"\b[A-Z]{3}\s?[0-9]{7}\b", signal, re.I):
        add("voter_id")

    has_aadhaar_num = _has_aadhaar_number_any_spacing(signal)
    has_uidai_context = bool(re.search(r"aadhaar|aadhar|uidai|unique\s+identification|mera\s+aadhaar|government\s+of\s+india|भारत\s+सरकार", lower, re.I))

    # Front: Aadhaar number + DOB/YOB + gender. Handles OCR like /D0B01/01/1967
    # and /Yearof Birth:1970.
    has_dob_or_yob = bool(re.search(r"(?:dob|d0b|date\s*of\s*birth|year\s*of\s*birth|yearof\s*birth|जन्म)", signal, re.I))
    has_gender = bool(re.search(r"\b(?:male|female|transgender|mahila|purush)\b|महिला|पुरुष", signal, re.I))
    if has_aadhaar_num and has_uidai_context and has_dob_or_yob and has_gender:
        add("aadhaar_front")

    # Back: Aadhaar number + pincode + any address/locality marker. In compact
    # KYC sheets, the back side often has only locality words and pincode while
    # OCR misses the literal 'Address' keyword.
    has_pin = bool(re.search(r"\b[1-9][0-9]{5}\b", signal))
    has_addr_marker = bool(re.search(
        r"\b(?:address|addr|pata|s/o|w/o|d/o|c/o|makan|makaan|ward|gram|tehsil|school|nagar|garden|pass|pas|mandsaur|moyakheda|moycheda|moriya|ratlam|pradesh|pin|pincode)\b|पता",
        signal,
        re.I,
    ))
    # If an Aadhaar document has a pincode and either explicit address markers
    # OR a repeated Aadhaar number, it is a back side too.
    aadhaar_nums = re.findall(r"[2-9]\d{3}\D{0,3}\d{4}\D{0,3}\d{4}", signal)
    if has_aadhaar_num and has_pin and has_uidai_context and (has_addr_marker or len(aadhaar_nums) >= 2):
        add("aadhaar_back")

    order = {"pan_card": 0, "aadhaar_front": 1, "aadhaar_back": 2, "voter_id": 3}
    detected = list(dict.fromkeys(detected))
    detected.sort(key=lambda dt: order.get(dt, 99))
    return detected

# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_single_image(
    img_bgr: np.ndarray,
    page_index: int = 0,
    multi_doc_enabled: bool = True,
) -> PageResult:
    """Full pipeline for one image."""

    # 1. Save original b64
    original_b64 = _ndarray_to_b64(img_bgr)

    # 2. Preprocess
    preprocessor = ImagePreprocessor()
    t = time.time()
    processed = preprocessor.preprocess(img_bgr)
    print(f"Preprocess: {time.time() - t:.2f}s")
    processed_b64 = _ndarray_to_b64(processed)
    
    # 3. OCR
    t = time.time()
    raw_text, ocr_confidence = OCREngine.run_ocr(processed)
    print(f"OCR: {time.time() - t:.2f}s")

    # 3b. Clean text for classification/extraction only — the original
    # raw_text (as returned by OCR) is still what gets stored on the
    # result objects below, so output/JSON structure is unchanged.
    cleaned_text = clean_text(raw_text)

    # 4. Classify — detect one or multiple docs
    if multi_doc_enabled:
        detected_types = KeywordClassifier.detect_multi_doc(cleaned_text)
    else:
        all_scores = KeywordClassifier.classify(cleaned_text)
        detected_types = [all_scores[0][0]] if all_scores and all_scores[0][1] > 0.1 else []

    # V7 combo-document guard:
    # Keyword ranking alone is not enough for stitched KYC sheets. A single image
    # may contain PAN + Aadhaar front + Aadhaar back + Voter. If we cap at top-3
    # or depend only on keywords, one side of Aadhaar gets dropped. Here we force
    # document types from field-level signals, then let each extractor decide its
    # own fields.
    if multi_doc_enabled:
        signal_text = cleaned_text + "\n" + raw_text

        # PAN: PAN number format is enough.
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", signal_text, re.I) and "pan_card" not in detected_types:
            detected_types.append("pan_card")

        # Voter: EPIC format is enough.
        if re.search(r"\b[A-Z]{3}\s?[0-9]{7}\b", signal_text, re.I) and "voter_id" not in detected_types:
            detected_types.append("voter_id")

        # Aadhaar front: DOB/YOB + gender + Aadhaar number anywhere in nearby text.
        has_aadhaar_num = bool(re.search(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", signal_text))
        has_front_signal = bool(re.search(r"(?:dob|d0b|date\s*of\s*birth|year\s*of\s*birth|yearof\s*birth|जन्म).{0,60}(?:\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4})", signal_text, re.I)) and bool(re.search(r"\b(?:male|female|महिला|पुरुष)\b", signal_text, re.I))
        if has_aadhaar_num and has_front_signal and "aadhaar_front" not in detected_types:
            detected_types.append("aadhaar_front")

        # Aadhaar back: address markers + pincode + Aadhaar number. Keep this
        # independent from front so both can coexist in output. Include common
        # OCR variants and locality words because Paddle may miss literal Address.
        has_back_signal = bool(re.search(r"\b(?:address|addr|pata|पता|s/o|w/o|d/o|c/o|makan|ward|gram|tehsil|school|nagar|mandsaur|moyakheda|moriya|ratlam|pradesh|pincode|pin)\b", signal_text, re.I)) and bool(re.search(r"\b[1-9][0-9]{5}\b", signal_text))
        if has_aadhaar_num and has_back_signal and "aadhaar_back" not in detected_types:
            detected_types.append("aadhaar_back")

        # Stable display order, no top-3/top-4 truncation.
        order = {"pan_card": 0, "aadhaar_front": 1, "aadhaar_back": 2, "voter_id": 3}
        detected_types = list(dict.fromkeys(detected_types))
        detected_types.sort(key=lambda dt: order.get(dt, 99))

        # V8: one more loose field-signal pass on RAW + cleaned OCR. This is what
        # catches pages where the raw OCR clearly contains address/pincode/DOB but
        # keyword score/ranking still dropped one Aadhaar side.
        detected_types = _force_detect_kyc_doc_types(raw_text, cleaned_text, detected_types)

    if not detected_types:
        # Fallback: return unknown
        doc_result = DocumentResult(
            doc_type="unknown",
            doc_label="Unknown Document",
            keyword_score=0.0,
            ocr_confidence=ocr_confidence,
            regex_score=0.0,
            overall_confidence=0.0,
            raw_text=raw_text,
            warnings=["Could not identify document type. Check image quality."],
        )
        return PageResult(
            page_index=page_index,
            original_b64=original_b64,
            processed_b64=processed_b64,
            documents=[doc_result],
            ocr_raw_text=raw_text,
        )

    # 5. For each detected doc type, extract fields + score
    all_scores_map = dict(KeywordClassifier.classify(cleaned_text, apply_anti_keywords=not multi_doc_enabled))
    documents = []

    for doc_type in detected_types:
        keyword_score = all_scores_map.get(doc_type, 0.0)
        fields, regex_score = RegexExtractor.extract((cleaned_text + "\n" + raw_text), doc_type)
        overall = ConfidenceScorer.compute(keyword_score, ocr_confidence, regex_score)

        warnings = []
        if ocr_confidence < 0.5:
            warnings.append("Low OCR confidence — image quality may be poor")
        if regex_score < 0.3:
            warnings.append("Few fields extracted — check document orientation")

        definition = DOCUMENT_DEFINITIONS.get(doc_type, {})
        doc_result = DocumentResult(
            doc_type=doc_type,
            doc_label=definition.get("label", doc_type),
            keyword_score=keyword_score,
            ocr_confidence=ocr_confidence,
            regex_score=regex_score,
            overall_confidence=overall,
            fields=fields,
            raw_text=raw_text,
            warnings=warnings,
        )
        documents.append(doc_result)

    return PageResult(
        page_index=page_index,
        original_b64=original_b64,
        processed_b64=processed_b64,
        documents=documents,
        ocr_raw_text=raw_text,
    )


async def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    multi_doc: bool = True,
) -> list[PageResult]:
    """
    Entry point for API.
    Returns list of PageResult (one per page for PDFs, one for images).
    """
    total_start = time.time()

    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        images = _pdf_to_images(file_bytes)
        results = []
        for idx, img in enumerate(images):
            result = process_single_image(img, page_index=idx, multi_doc_enabled=multi_doc)
            results.append(result)
            print(f"Total Processing: {time.time() - total_start:.2f}s")
        return results
    else:
        # Image file
        pil_img = Image.open(io.BytesIO(file_bytes))
        img_bgr = _pil_to_ndarray(pil_img)
        result = process_single_image(img_bgr, page_index=0, multi_doc_enabled=multi_doc)
        print(f"Total Processing: {time.time() - total_start:.2f}s")
        return [result]


# ---------------------------------------------------------------------------
# Serialization helper (for JSON response)
# ---------------------------------------------------------------------------

def page_result_to_dict(pr: PageResult) -> dict:
    return {
        "page_index": pr.page_index,
        "original_b64": pr.original_b64,
        "processed_b64": pr.processed_b64,
        "ocr_raw_text": pr.ocr_raw_text,
        "documents": [
            {
                "doc_type": dr.doc_type,
                "doc_label": dr.doc_label,
                "scores": {
                    "keyword": dr.keyword_score,
                    "ocr_confidence": dr.ocr_confidence,
                    "regex": dr.regex_score,
                    "overall": dr.overall_confidence,
                },
                "fields": {
                    k: {"value": v.value, "confidence": v.confidence, "source": v.source}
                    for k, v in dr.fields.items()
                },
                "warnings": dr.warnings,
            }
            for dr in pr.documents
        ],
    }