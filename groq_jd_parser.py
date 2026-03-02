"""
Groq JD Parser — Job Description Parser using Llama 3.1 8B via Groq
Two-pass LLM extraction with confidence scoring and provenance tracking.
Optimized for accuracy over speed.
"""

import json
import os
import re
import time
import uuid
import requests
from datetime import datetime, timezone


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert job description parser for an Applicant Tracking System. "
    "You extract structured data from job descriptions with perfect accuracy. "
    "You ONLY return valid JSON. No explanations, no markdown, no extra text."
)

EXTRACTION_PROMPT = r"""Extract ALL information from the job description below into the EXACT JSON structure shown. Follow every rule carefully.

RULES:
- Use null for any field whose value is NOT explicitly stated in the text. NEVER guess or fabricate values.
- Use empty arrays [] where a list field has no items found in the text.
- Be thorough — capture EVERY skill, EVERY requirement, EVERY responsibility mentioned in the text.
- All string list fields (requirements, responsibilities, benefits, preferred_experience, preferred_technologies, certifications) must be arrays of PLAIN STRINGS. Do NOT use objects.

FIELD-BY-FIELD INSTRUCTIONS:

1. title — Object: {"text": "exact job title", "seniority_level": "Senior"|"Lead"|"Junior"|"Mid"|"Entry"|null, "domain": "data"|"engineering"|"security"|"management"|"IT"|etc}

2. company — Plain string of company name. null if not mentioned.

3. location — Object: {"city": str|null, "region": str|null, "country": str|null, "remote": "remote"|"hybrid"|"onsite", "formatted_address": "City, Region, Country"}

4. employment_type — Array of NORMALIZED strings:
   "Full-time"/"Full time" → "full_time", "Part-time" → "part_time", "Contract"/"C2C" → "contract", "Internship" → "internship", "Temporary" → "temporary", "Freelance" → "freelance"
   Return [] if not stated.

5. salary — Object or null: {"min": number, "max": number, "currency": "USD"|"INR"|"EUR"|"GBP", "period": "year"|"month"|"hour", "ote": false}
   Convert: "12-18 LPA" → min:1200000, max:1800000, currency:"INR". "$80K-120K" → min:80000, max:120000, currency:"USD". null if not mentioned.

6. requirements — Array of PLAIN STRINGS. Each string is one requirement/qualification bullet copied EXACTLY from the text.

7. responsibilities — Array of PLAIN STRINGS. Each string is one responsibility bullet copied EXACTLY from the text.

8. skills — Array of objects: [{"name": "skill_name", "category": "category"}]
   Categories: "programming_language", "framework", "database", "cloud", "devops", "tool", "methodology", "domain", "soft_skill", "networking", "os", "other"
   Extract ALL skills from everywhere in the JD. Do NOT include "source" or any extra keys.

8b. technical_skills — Array of PLAIN STRINGS of technical/hard skills only (e.g., ["SQL", "Power BI", "Excel", "Python"]). Extract from the skills list above, including only those with categories: programming_language, framework, database, cloud, devops, tool, networking, os.

8c. soft_skills — Array of PLAIN STRINGS of soft/interpersonal skills only (e.g., ["Communication", "Leadership", "Problem Solving"]). Extract from the skills list above, including only those with category: soft_skill.

9. education — Object or null: {"level": "Bachelor's"|"Master's"|"PhD"|etc, "field": "field of study"}

10. experience_years — Object or null: {"min_years": number, "max_years": number, "requirement_type": "required"|"preferred"}

11. benefits — Array of PLAIN STRINGS like ["health insurance", "401k", "bonus"]. [] if none.

12. work_authorization — String or null.

13. job_domain — String like "Information Technology", "Healthcare", etc. Infer from context if clear.

14. job_summary — The brief job summary paragraph (usually 1-3 sentences at the top describing the role). This is the SHORT overview.

15. description — The FULL detailed job description text. This is the longer, more detailed paragraph(s) that describe the role in depth, including the full "Job Description" or "About the Role" section. If there is no separate detailed description beyond the summary, set to null.

16. job_id — String or null.

16. work_mode — One of: "remote", "hybrid", "onsite". Default "onsite" if not stated.

17. job_posted_date — "YYYY-MM-DD" format string or null.

18. job_expiry_date — "YYYY-MM-DD" format string or null.

19. reporting_to — String or null.

20. team_size — String or null.

21. travel_requirement — String or null.

22. application_link — URL string or null.

23. equal_opportunity_statement — String or null.

24. company_website — URL string or null.

25. industry — String or null.

26. company_size — String or null.

27. company_overview — String or null.

28. preferred_experience — Array of PLAIN STRINGS. [] if none.

29. preferred_technologies — Array of PLAIN STRINGS. [] if none.

30. certifications — Array of PLAIN STRINGS. [] if none.

CRITICAL:
- requirements, responsibilities, benefits, preferred_experience, preferred_technologies, certifications MUST be arrays of plain strings. NOT arrays of objects.
- skills MUST be [{"name": "...", "category": "..."}] with ONLY those two keys per skill. No "source" key.
- company MUST be a plain string, NOT an object.

{
  "title": {"text": "", "seniority_level": null, "domain": ""},
  "company": "",
  "location": {"city": "", "region": null, "country": "", "remote": "", "formatted_address": ""},
  "employment_type": [],
  "salary": null,
  "requirements": [],
  "responsibilities": [],
  "skills": [{"name": "", "category": ""}],
  "technical_skills": [],
  "soft_skills": [],
  "education": null,
  "experience_years": null,
  "benefits": [],
  "work_authorization": null,
  "job_domain": null,
  "job_summary": null,
  "description": null,
  "job_id": null,
  "work_mode": "onsite",
  "job_posted_date": null,
  "job_expiry_date": null,
  "reporting_to": null,
  "team_size": null,
  "travel_requirement": null,
  "application_link": null,
  "equal_opportunity_statement": null,
  "company_website": null,
  "industry": null,
  "company_size": null,
  "company_overview": null,
  "preferred_experience": [],
  "preferred_technologies": [],
  "certifications": []
}

JOB DESCRIPTION:
---
JD_TEXT_HERE
---

Return ONLY the JSON object. No other text."""


VALIDATION_PROMPT = r"""You are verifying a parsed job description. Below is the original JD text and the extracted JSON. CHECK every field and CORRECT any errors. Return the corrected JSON.

VERIFICATION CHECKLIST:
1. Every value MUST be traceable to the original text. Set fabricated values to null.
2. Skills categories must be correct (programming_language vs framework vs tool etc).
3. Salary numbers must be correctly converted (LPA=lakhs per annum, K=thousands).
4. employment_type values must be normalized: "full_time", "part_time", "contract", "internship", "temporary", "freelance".
5. work_mode must be: "remote", "hybrid", or "onsite".
6. ALL responsibility and requirement bullets must be captured. Add any missing ones.
7. Dates must be YYYY-MM-DD format.
8. No duplicate skills.
9. requirements, responsibilities, benefits MUST be arrays of plain strings, NOT objects.
10. company MUST be a plain string, NOT an object.
11. skills MUST have ONLY "name" and "category" keys per item.

ORIGINAL JD:
---
JD_TEXT_HERE
---

EXTRACTED (verify and correct):
EXTRACTED_JSON_HERE

Return ONLY the corrected JSON. No explanations."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def is_groq_configured():
    return bool(GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext == ".docx":
        return _extract_docx(filepath)
    elif ext == ".doc":
        return _extract_doc(filepath)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        return _extract_image_ocr(filepath)
    elif ext in (".txt", ".html", ".htm"):
        with open(filepath, "r", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(filepath):
    import fitz
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    if len(text.strip()) < 50:
        try:
            return _extract_image_ocr(filepath)
        except Exception:
            pass
    return text


def _extract_docx(filepath):
    import docx2txt
    return docx2txt.process(filepath)


def _extract_doc(filepath):
    import subprocess
    try:
        result = subprocess.run(
            ["antiword", filepath], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass
    try:
        import olefile
        ole = olefile.OleFileIO(filepath)
        if ole.exists("WordDocument"):
            stream = ole.openstream("WordDocument")
            data = stream.read()
            text = data.decode("latin-1", errors="ignore")
            clean = "".join(
                c if c.isprintable() or c in "\n\r\t" else " " for c in text
            )
            ole.close()
            if len(clean.strip()) > 50:
                return clean
    except Exception:
        pass
    raise ValueError("Cannot extract text from DOC file. Install antiword.")


def _extract_image_ocr(filepath):
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ValueError("OCR not available. Install: pip install pytesseract Pillow")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        import fitz
        from io import BytesIO
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            full_text += pytesseract.image_to_string(img) + "\n"
        doc.close()
        return full_text
    else:
        img = Image.open(filepath)
        return pytesseract.image_to_string(img)


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------

def _call_groq(messages, model=None, api_key=None, temperature=0.1, max_tokens=8192):
    key = api_key or GROQ_API_KEY
    mdl = model or GROQ_MODEL
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    finish = data["choices"][0].get("finish_reason", "unknown")
    return content, usage, finish


# ---------------------------------------------------------------------------
# JSON extraction (robust, 3-strategy)
# ---------------------------------------------------------------------------

def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Normalize LLM output — fix common format issues
# ---------------------------------------------------------------------------

def _normalize_llm_output(parsed):
    """Clean up LLM output to ensure consistent types."""
    if not isinstance(parsed, dict):
        return parsed

    # Fix company: if it's an object like {"text": "X", "source": "Y"}, extract the string
    company = parsed.get("company")
    if isinstance(company, dict):
        parsed["company"] = company.get("text") or company.get("name") or company.get("value") or str(company)

    # Fix string list fields: if items are objects, extract the text
    for list_field in ("requirements", "responsibilities", "benefits",
                       "preferred_experience", "preferred_technologies", "certifications",
                       "technical_skills", "soft_skills"):
        items = parsed.get(list_field)
        if isinstance(items, list):
            cleaned = []
            for item in items:
                if isinstance(item, str):
                    cleaned.append(item)
                elif isinstance(item, dict):
                    # Extract text from object
                    text = item.get("text") or item.get("value") or item.get("name") or ""
                    if text:
                        cleaned.append(text)
                    else:
                        # Use first string value
                        for v in item.values():
                            if isinstance(v, str) and len(v) > 5:
                                cleaned.append(v)
                                break
            parsed[list_field] = cleaned

    # Fix skills: strip any extra keys beyond name and category
    skills = parsed.get("skills")
    if isinstance(skills, list):
        cleaned_skills = []
        for skill in skills:
            if isinstance(skill, dict):
                cleaned_skills.append({
                    "name": skill.get("name", ""),
                    "category": skill.get("category", "other"),
                })
            elif isinstance(skill, str):
                cleaned_skills.append({"name": skill, "category": "other"})
        parsed["skills"] = cleaned_skills

    # Fix salary: strip any extra keys
    salary = parsed.get("salary")
    if isinstance(salary, dict):
        parsed["salary"] = {
            "min": salary.get("min"),
            "max": salary.get("max"),
            "currency": salary.get("currency"),
            "period": salary.get("period", "year"),
            "ote": salary.get("ote", False),
        }

    # Fix education: strip extra keys
    edu = parsed.get("education")
    if isinstance(edu, dict):
        parsed["education"] = {
            "level": edu.get("level"),
            "field": edu.get("field"),
        }

    # Fix experience_years: strip extra keys
    exp = parsed.get("experience_years")
    if isinstance(exp, dict):
        parsed["experience_years"] = {
            "min_years": exp.get("min_years"),
            "max_years": exp.get("max_years"),
            "requirement_type": exp.get("requirement_type", "required"),
        }

    # Fix title: ensure it's the expected object format
    title = parsed.get("title")
    if isinstance(title, str):
        parsed["title"] = {"text": title, "seniority_level": None, "domain": None}
    elif isinstance(title, dict):
        parsed["title"] = {
            "text": title.get("text", ""),
            "seniority_level": title.get("seniority_level"),
            "domain": title.get("domain"),
        }

    # Fix location: strip extra keys
    loc = parsed.get("location")
    if isinstance(loc, dict):
        parsed["location"] = {
            "city": loc.get("city"),
            "region": loc.get("region"),
            "country": loc.get("country"),
            "remote": loc.get("remote", "onsite"),
            "formatted_address": loc.get("formatted_address", ""),
        }

    # Remove _source if present (we handle provenance ourselves)
    parsed.pop("_source", None)

    return parsed


# ---------------------------------------------------------------------------
# Provenance: find character spans in original text
# ---------------------------------------------------------------------------

def _find_spans(original_text, search_text):
    """Find character offset spans of search_text in original_text."""
    if not search_text or not original_text:
        return []
    search_clean = str(search_text).strip()
    if not search_clean or len(search_clean) < 2:
        return []

    # Exact match
    idx = original_text.find(search_clean)
    if idx != -1:
        return [[idx, idx + len(search_clean)]]

    # Case-insensitive
    lower_orig = original_text.lower()
    lower_search = search_clean.lower()
    idx = lower_orig.find(lower_search)
    if idx != -1:
        return [[idx, idx + len(search_clean)]]

    # Try first 80 chars (for long texts)
    if len(search_clean) > 80:
        short = search_clean[:80]
        idx = lower_orig.find(short.lower())
        if idx != -1:
            return [[idx, idx + len(search_clean)]]

    # Try first significant line
    for line in search_clean.split("\n"):
        line = line.strip()
        if len(line) > 10:
            idx = lower_orig.find(line.lower())
            if idx != -1:
                return [[idx, idx + len(line)]]

    return []


def _get_search_text_for_field(field_name, value):
    """Derive a text to search for in the original text based on field value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # For structured fields, pick the most searchable text
        if field_name == "title":
            return value.get("text", "")
        if field_name == "location":
            return value.get("formatted_address", "") or value.get("city", "")
        if field_name == "salary":
            return ""  # Salary text varies too much
        if field_name == "experience_years":
            mn = value.get("min_years")
            mx = value.get("max_years")
            if mn is not None and mx is not None:
                return f"{mn}" if mn == mx else f"{mn}"
            return ""
        if field_name == "education":
            return value.get("level", "")
        # Generic: try "text" or "name" keys
        return value.get("text", "") or value.get("name", "")
    if isinstance(value, list) and len(value) > 0:
        # For lists, use the first item
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("name", "") or first.get("text", "")
    return ""


def _build_provenance(original_text, field_name, value, extractor_name):
    """Build provenance dict by searching for the value in the original text."""
    search_text = _get_search_text_for_field(field_name, value)
    spans = _find_spans(original_text, search_text)

    # For list fields, try to find the section span
    if isinstance(value, list) and len(value) > 0 and not spans:
        # Find first and last items to get section bounds
        first_item = value[0] if isinstance(value[0], str) else (value[0].get("name", "") if isinstance(value[0], dict) else "")
        last_item = value[-1] if isinstance(value[-1], str) else (value[-1].get("name", "") if isinstance(value[-1], dict) else "")
        first_spans = _find_spans(original_text, first_item)
        last_spans = _find_spans(original_text, last_item)
        if first_spans and last_spans:
            spans = [[first_spans[0][0], last_spans[0][1]]]

    return {
        "spans": spans,
        "extractor": extractor_name,
        "extractor_version": "1.0.0",
        "rule_id": None,
        "extracted_text": search_text[:200] if search_text else "",
    }


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

_BASE_CONFIDENCE = {
    "title": 0.95, "company": 0.95, "location": 0.90,
    "employment_type": 0.95, "salary": 0.80, "requirements": 0.90,
    "responsibilities": 0.95, "skills": 0.90, "technical_skills": 0.90,
    "soft_skills": 0.90, "education": 0.90, "experience_years": 0.95,
    "benefits": 0.85, "work_authorization": 0.85, "job_domain": 0.85,
    "job_summary": 0.95, "description": 0.95, "job_id": 0.95,
    "work_mode": 0.90, "job_posted_date": 0.95, "job_expiry_date": 0.95,
    "reporting_to": 0.95, "team_size": 0.95, "travel_requirement": 0.95,
    "application_link": 0.95, "source_type": 0.95, "language_detected": 0.95,
}

_EXTRACTOR_NAMES = {
    "title": "job_title_extractor", "company": "company_label_extractor",
    "location": "location_extractor", "employment_type": "employment_type_extractor",
    "salary": "salary_extractor", "requirements": "sectionizer",
    "responsibilities": "sectionizer", "skills": "skills_extractor",
    "technical_skills": "skills_extractor", "soft_skills": "skills_extractor",
    "education": "education_extractor", "experience_years": "experience_extractor",
    "benefits": "benefits_extractor", "work_authorization": "metadata_extractor",
    "job_domain": "domain_extractor", "job_summary": "sectionizer",
    "description": "sectionizer", "job_id": "metadata_extractor",
    "work_mode": "location_extractor", "job_posted_date": "date_extractor",
    "job_expiry_date": "date_extractor", "reporting_to": "metadata_extractor",
    "team_size": "metadata_extractor", "travel_requirement": "metadata_extractor",
    "application_link": "metadata_extractor", "source_type": "metadata_extractor",
    "language_detected": "metadata_extractor",
}


def _calc_confidence(field_name, value, spans, original_text):
    base = _BASE_CONFIDENCE.get(field_name, 0.85)
    if value is None:
        return 0.0
    # Boost if we found the value in the original text (span match)
    if spans:
        base = min(base + 0.05, 1.0)
    return round(base, 2)


def _field_status(confidence):
    return "ok" if confidence >= 0.80 else "low_confidence"


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

_ALL_FIELDS = [
    "title", "company", "location", "employment_type", "salary",
    "requirements", "responsibilities", "skills", "technical_skills",
    "soft_skills", "education", "experience_years", "benefits",
    "work_authorization", "job_domain", "job_summary", "description",
    "job_id", "work_mode", "job_posted_date", "job_expiry_date",
    "reporting_to", "team_size", "travel_requirement", "application_link",
    "equal_opportunity_statement", "company_website", "industry",
    "company_size", "company_overview", "preferred_experience",
    "preferred_technologies", "certifications",
    "source_type", "language_detected",
]


def _wrap_field(field_name, value, original_text):
    """Wrap a value into the output format with confidence + provenance."""
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 0:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None

    extractor = _EXTRACTOR_NAMES.get(field_name, "generic_extractor")
    prov = _build_provenance(original_text, field_name, value, extractor)
    confidence = _calc_confidence(field_name, value, prov["spans"], original_text)

    return {
        "value": value,
        "confidence": confidence,
        "provenance": prov,
        "status": _field_status(confidence),
    }


def _build_output(original_text, parsed, source_filename, metadata):
    fields = {}
    for fname in _ALL_FIELDS:
        val = parsed.get(fname)
        wrapped = _wrap_field(fname, val, original_text)
        if wrapped is not None:
            fields[fname] = wrapped
        else:
            fields[fname] = None

    # Global confidence = average of non-null field confidences
    confidences = [f["confidence"] for f in fields.values() if isinstance(f, dict)]
    global_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "id": str(uuid.uuid4()),
        "source": {
            "type": "file",
            "filename": source_filename,
            "url": None,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        },
        "detected_language": "en",
        "global_confidence": global_conf,
        "fields": fields,
        "_metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _post_process(parsed):
    if not isinstance(parsed, dict):
        return parsed, []
    applied = []

    try:
        _fix_employment_type(parsed)
        applied.append("employment_type_normalize")
    except Exception:
        pass
    try:
        _fix_work_mode(parsed)
        applied.append("work_mode_normalize")
    except Exception:
        pass
    try:
        _fix_salary(parsed)
        applied.append("salary_numbers")
    except Exception:
        pass
    try:
        _fix_skill_dedup(parsed)
        applied.append("skill_dedup")
    except Exception:
        pass
    try:
        _fix_location_consistency(parsed)
        applied.append("location_consistency")
    except Exception:
        pass

    try:
        _fix_derive_skill_splits(parsed)
        applied.append("derive_skill_splits")
    except Exception:
        pass

    return parsed, applied


def _fix_derive_skill_splits(parsed):
    """Derive technical_skills and soft_skills from skills if not already set."""
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return

    tech_categories = {
        "programming_language", "framework", "database", "cloud",
        "devops", "tool", "networking", "os", "other", "methodology", "domain",
    }

    # Only derive if not already populated
    if not parsed.get("technical_skills"):
        parsed["technical_skills"] = [
            s["name"] for s in skills
            if isinstance(s, dict) and s.get("category") in tech_categories
        ]
    if not parsed.get("soft_skills"):
        parsed["soft_skills"] = [
            s["name"] for s in skills
            if isinstance(s, dict) and s.get("category") == "soft_skill"
        ]


def _fix_employment_type(parsed):
    emp = parsed.get("employment_type")
    if not isinstance(emp, list):
        return
    mapping = {
        "full-time": "full_time", "full time": "full_time", "fulltime": "full_time",
        "part-time": "part_time", "part time": "part_time",
        "contract": "contract", "c2c": "contract", "corp-to-corp": "contract",
        "internship": "internship", "intern": "internship",
        "temporary": "temporary", "temp": "temporary",
        "freelance": "freelance",
    }
    parsed["employment_type"] = [
        mapping.get(i.strip().lower(), i.strip().lower())
        for i in emp if isinstance(i, str)
    ]


def _fix_work_mode(parsed):
    wm = parsed.get("work_mode")
    if not isinstance(wm, str):
        return
    mapping = {
        "remote": "remote", "work from home": "remote", "wfh": "remote",
        "hybrid": "hybrid", "flexible": "hybrid",
        "onsite": "onsite", "on-site": "onsite", "on site": "onsite",
        "in-office": "onsite", "office": "onsite",
    }
    parsed["work_mode"] = mapping.get(wm.strip().lower(), wm.strip().lower())


def _fix_salary(parsed):
    sal = parsed.get("salary")
    if not isinstance(sal, dict):
        return
    for key in ("min", "max"):
        val = sal.get(key)
        if isinstance(val, str):
            clean = re.sub(r"[^\d.]", "", val)
            try:
                sal[key] = float(clean) if "." in clean else int(clean)
            except ValueError:
                pass


def _fix_skill_dedup(parsed):
    skills = parsed.get("skills")
    if not isinstance(skills, list):
        return
    seen = set()
    deduped = []
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name", "").strip().lower()
            if name and name not in seen:
                seen.add(name)
                deduped.append(skill)
    parsed["skills"] = deduped


def _fix_location_consistency(parsed):
    loc = parsed.get("location")
    wm = parsed.get("work_mode")
    if isinstance(loc, dict) and isinstance(wm, str):
        loc["remote"] = wm


# ---------------------------------------------------------------------------
# Main parse function — two-pass for accuracy
# ---------------------------------------------------------------------------

def parse_jd(jd_text, filename="unknown", model=None, api_key=None):
    """Parse a job description using two-pass LLM extraction.

    Pass 1: Extract all fields.
    Pass 2: Validate and correct against original text.
    """
    key = api_key or GROQ_API_KEY
    mdl = model or GROQ_MODEL

    if not key:
        return {"error": "GROQ_API_KEY not set."}

    total_start = time.time()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finish_reason = "unknown"

    # --- Pass 1: Extraction ---
    prompt1 = EXTRACTION_PROMPT.replace("JD_TEXT_HERE", jd_text)
    try:
        content1, usage1, finish1 = _call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt1},
            ],
            model=mdl, api_key=key, temperature=0.1, max_tokens=8192,
        )
        for k in total_usage:
            total_usage[k] += usage1.get(k, 0)
        finish_reason = finish1
    except Exception as e:
        return {"error": f"Pass 1 failed: {str(e)}"}

    extracted = _extract_json(content1)
    if extracted is None:
        return {"error": "Failed to parse JSON from model response", "raw": content1[:500]}

    # Normalize pass 1 output
    extracted = _normalize_llm_output(extracted)

    # --- Pass 2: Validation (with brief delay to avoid rate limit) ---
    time.sleep(1)
    extracted_str = json.dumps(extracted, indent=2, ensure_ascii=False)
    prompt2 = VALIDATION_PROMPT.replace("JD_TEXT_HERE", jd_text).replace(
        "EXTRACTED_JSON_HERE", extracted_str
    )
    try:
        content2, usage2, finish2 = _call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt2},
            ],
            model=mdl, api_key=key, temperature=0.0, max_tokens=8192,
        )
        for k in total_usage:
            total_usage[k] += usage2.get(k, 0)
        finish_reason = finish2

        validated = _extract_json(content2)
        if validated is not None:
            validated = _normalize_llm_output(validated)
        else:
            validated = extracted
            finish_reason = "pass2_parse_failed"
    except Exception:
        validated = extracted
        finish_reason = "pass2_failed"

    # --- Post-processing ---
    validated, pp_applied = _post_process(validated)

    elapsed_ms = int((time.time() - total_start) * 1000)

    metadata = {
        "parser": "groq_jd_parser",
        "model": mdl,
        "processing_time_ms": elapsed_ms,
        "finish_reason": finish_reason,
        "prompt_tokens": total_usage["prompt_tokens"],
        "completion_tokens": total_usage["completion_tokens"],
        "total_tokens": total_usage["total_tokens"],
        "passes": 2,
        "_post_processed": pp_applied,
    }

    # Set metadata fields that are derived, not extracted
    validated["source_type"] = "file"
    validated["language_detected"] = "en"

    return _build_output(jd_text, validated, filename, metadata)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY in .env file first")
        exit(1)

    test_jd = """
Job Title: Business Analyst
Company Name: Arytic Inc.
Location: Hyderabad, India (Hybrid)
Employment Type: Full-time
Job Posted Date: 2025-10-15
Job Expiry Date: 2025-11-30

Job Summary:
The Business Analyst will bridge the gap between business needs and technical implementation
by gathering, analyzing, and documenting requirements to drive product enhancements and
operational excellence.

Responsibilities:
- Conduct detailed business analysis
- Gather and document functional and non-functional requirements
- Prepare BRDs, user stories, and acceptance criteria
- Collaborate with developers, testers, and stakeholders

Required Qualifications:
- Bachelor's degree in Business Administration or Computer Science
- 3-5 years of experience as a Business Analyst
- Strong knowledge of Agile and SDLC
- Proficiency in Excel, SQL, Power BI or Tableau

Salary: 12-18 LPA
Reporting To: Product Manager
Team Size: 6
Travel: Minimal
Apply: https://arytic.com/careers/business-analyst

Arytic is an equal opportunity employer.
"""
    result = parse_jd(test_jd, filename="test_jd.txt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
