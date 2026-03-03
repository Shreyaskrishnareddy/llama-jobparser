"""
Groq JD Parser — Job Description Parser using Llama 3.1 8B via Groq
Single-pass LLM extraction with confidence scoring and provenance tracking.
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

1. title — Object: {"text": "exact job title", "seniority_level": "Senior"|"Lead"|"Junior"|"Mid"|"Entry"|null, "domain": "cloud"|"data"|"engineering"|"security"|"management"|"IT"|"networking"|"devops"|"support"|"operations"|etc}
   Choose domain based on the job title itself. "Cloud Architect" → "cloud", "Data Analyst" → "data", "Security Officer" → "security", "Network Engineer" → "networking", "Project Manager" → "management", "Project Lead" → "management", "Support Analyst" → "support", "Operations Analyst" → "operations", "Systems Analyst" → "IT", "Full Stack Developer" → "engineering".

2. company — Plain string of company name. null if not mentioned. If "Client:" field contains both an ID and a name (e.g., "DAG2605-Austin Energy"), extract ONLY the company name ("Austin Energy"), not the ID. The ID goes in job_id.

3. location — Object: {"city": str|null, "region": str|null, "country": str|null, "remote": "remote"|"hybrid"|"onsite", "formatted_address": "City, Region, Country"}
   IMPORTANT: "country" is the NATION (e.g., "India", "USA", "UK"). "region" is the STATE/PROVINCE (e.g., "Telangana", "California"). Do NOT put the country name in region. Example: "Hyderabad, India" → city:"Hyderabad", region:null, country:"India".

4. employment_type — Array of NORMALIZED strings:
   "Full-time"/"Full time" → "full_time", "Part-time" → "part_time", "Contract"/"C2C" → "contract", "Internship" → "internship", "Temporary" → "temporary", "Freelance" → "freelance"
   IMPORTANT: If a "Duration" is mentioned (e.g., "12 months", "22+ Months", "6 month contract"), the employment type is "contract". If the text says "C2C", "Corp-to-Corp", "W2", or mentions a staffing client, the employment type is "contract". Do NOT default to "full_time" — only use "full_time" if the text explicitly says "Full-time" or "Full time".
   Return [] if employment type is not stated and no duration is mentioned.

5. salary — Object or null: {"min": number, "max": number, "currency": "USD"|"INR"|"EUR"|"GBP", "period": "year"|"month"|"hour", "ote": false}
   Convert ONLY when abbreviations are used: "12-18 LPA" → min:1200000, max:1800000. "$80K-120K" → min:80000, max:120000. "K"=thousands, "LPA"/"L"=lakhs.
   Do NOT multiply plain numbers: "$70/hr" → min:70, max:70, period:"hour". "$150,000/year" → min:150000, max:150000, period:"year". null if not mentioned.

6. requirements — Array of PLAIN STRINGS. Each string is one requirement/qualification bullet copied EXACTLY from the text. ONLY include items from the "Required" qualifications/skills section. Do NOT include items from the "Preferred" or "Nice-to-Have" section — those go in preferred_experience.

7. responsibilities — Array of PLAIN STRINGS. Each string is one responsibility bullet copied EXACTLY from the text.

8. skills — Array of objects: [{"name": "skill_name", "category": "category"}]
   Categories: "programming_language", "framework", "database", "cloud", "devops", "tool", "methodology", "domain", "soft_skill", "networking", "os", "other"
   Extract ALL skills from everywhere in the JD — required skills, preferred skills, responsibilities, and description. Include alternatives separated by "or"/"/" (e.g., "Power BI or Tableau" → extract BOTH). Extract specific technologies mentioned inside parentheses too (e.g., "Oracle RAC (Real Application Clusters)" → extract "Oracle RAC"). Be exhaustive — a technical JD may have 20-30+ skills. Do NOT include "source" or any extra keys.

8b. technical_skills — Array of PLAIN STRINGS of technical/hard skills only (e.g., ["SQL", "Power BI", "Excel", "Python"]). Extract from the skills list above, including only those with categories: programming_language, framework, database, cloud, devops, tool, networking, os.

8c. soft_skills — Array of PLAIN STRINGS of soft/interpersonal skills only (e.g., ["Communication", "Leadership", "Problem Solving"]). ONLY include soft skills if the EXACT word appears in the JD text. For example, only include "Communication" if the word "communication" literally appears in the text. If the JD does not explicitly name any soft skills, return []. NEVER guess or infer soft skills from job responsibilities.

9. education — Object or null: {"level": "Bachelor's"|"Master's"|"PhD"|etc, "field": "field of study"}

10. experience_years — Object or null: {"min_years": number, "max_years": number, "requirement_type": "required"|"preferred"}
    Look for patterns like "X years of experience", "X+ years", "X-Y years". If only one number given (e.g., "8 years"), use it as both min and max.

11. benefits — Array of PLAIN STRINGS like ["health insurance", "401k", "bonus"]. ONLY include benefits that are explicitly named in the text. If the JD says something vague like "as per agreement" or "standard benefits", return []. Do NOT invent specific benefits.

12. work_authorization — String or null. Include residency or location restrictions like "LOCAL ONLY", "must reside in...", "US citizens only", "work permit required", etc.

13. job_domain — String like "Information Technology", "Healthcare", etc. Infer from context if clear.

14. job_summary — The brief job summary paragraph (usually 1-3 sentences at the top describing the role). This is the SHORT overview. Look for sections like "Role Summary", "Objective", "Scope/Description of Services", "Overview", or the first paragraph describing the role.

15. description — The FULL detailed job description text. This is the longer, more detailed paragraph(s) that describe the role in depth. Look for sections titled "Job Description", "About the Role", "Scope/Description of Services", "Description of Services", or any paragraph that describes the role duties in detail. If the JD has only one descriptive paragraph, use it as description and use its first 1-2 sentences as job_summary.

16. job_id — String or null. Look for fields like "Job ID", "Reference ID", "Requisition ID", "Client ID", "Client:", or any alphanumeric code identifying the position. If "Client:" has "DAG2605-Austin Energy", the job_id is "DAG2605". Extract the code/ID portion.

16. work_mode — One of: "remote", "hybrid", "onsite". Default "onsite" if not stated.

17. job_posted_date — "YYYY-MM-DD" format string or null.

18. job_expiry_date — "YYYY-MM-DD" format string or null.

19. reporting_to — String or null. ONLY include if the JD explicitly says "Reports to", "Reporting To", or "Manager:". Do NOT guess from company name or client name.

20. team_size — String or null.

21. travel_requirement — String or null.

22. application_link — URL string or null.

23. equal_opportunity_statement — String or null.

24. company_website — URL string or null. Can be inferred from application_link domain if clearly a company domain (e.g., "https://arytic.com/careers/..." → "https://arytic.com").

25. industry — String or null. Infer from company overview or job context if clearly identifiable (e.g., "recruitment platform" → "Human Resources / Technology").

26. company_size — String or null.

27. company_overview — String or null.

28. preferred_experience — Array of PLAIN STRINGS. Capture ALL preferred/nice-to-have experience items. [] if none.

29. preferred_technologies — Array of PLAIN STRINGS. Extract ONLY technologies mentioned in the "Preferred Skills", "Nice-to-Have", "Familiarity with", or "Valuable skills" section. Do NOT include technologies from "Required Qualifications" or "Responsibilities" sections. Examples: "Lambda", "Azure Functions", "Databricks". [] if none.

30. certifications — Array of PLAIN STRINGS. [] if none.

CRITICAL — MUST FOLLOW:
- requirements, responsibilities, benefits, preferred_experience, preferred_technologies, certifications MUST be arrays of plain strings. NOT arrays of objects.
- skills MUST be [{"name": "...", "category": "..."}] with ONLY those two keys per skill. No "source" key.
- company MUST be a plain string, NOT an object.
- employment_type: If "Duration" is mentioned (e.g., "22+ Months", "12 months", "6 month"), this is a CONTRACT role → use ["contract"]. Only use "full_time" if the JD explicitly says "Full-time" or "Full time".
- soft_skills: NEVER infer or guess. Only include a soft skill if the exact word (e.g., "communication", "leadership") literally appears in the JD text. If no soft skills are explicitly written, return [].
- All fields: If data is not present in the JD text, use null or []. NEVER fabricate values.

{
  "title": {"text": "", "seniority_level": null, "domain": ""},
  "company": "",
  "location": {"city": "", "region": null, "country": "", "remote": "", "formatted_address": ""},
  "employment_type": [],
  "salary": null,
  "education": null,
  "experience_years": null,
  "work_mode": "onsite",
  "job_id": null,
  "job_domain": null,
  "job_summary": null,
  "work_authorization": null,
  "job_posted_date": null,
  "job_expiry_date": null,
  "reporting_to": null,
  "team_size": null,
  "travel_requirement": null,
  "application_link": null,
  "industry": null,
  "company_size": null,
  "company_website": null,
  "company_overview": null,
  "equal_opportunity_statement": null,
  "certifications": [],
  "requirements": [],
  "responsibilities": [],
  "description": null,
  "preferred_experience": [],
  "preferred_technologies": [],
  "benefits": [],
  "skills": [{"name": "", "category": ""}],
  "technical_skills": [],
  "soft_skills": []
}

JOB DESCRIPTION:
---
JD_TEXT_HERE
---

Return ONLY the JSON object. No other text."""



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

def _call_groq(messages, model=None, api_key=None, temperature=0.1, max_tokens=32768, frequency_penalty=0.0):
    key = api_key or GROQ_API_KEY
    mdl = model or GROQ_MODEL

    max_retries = 5
    for attempt in range(max_retries):
        payload = {
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if frequency_penalty > 0:
            payload["frequency_penalty"] = frequency_penalty
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code == 429:
            # Rate limited — extract wait time and retry
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
                wait_match = re.search(r"try again in (\d+\.?\d*)s", err_msg)
                wait_time = float(wait_match.group(1)) + 1 if wait_match else 20
            except Exception:
                wait_time = 20
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        finish = data["choices"][0].get("finish_reason", "unknown")

        # If JSON parsing fails and finish_reason is 'length', retry is handled by caller
        return content, usage, finish

    raise RuntimeError("Groq API: max retries exceeded due to rate limiting")


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

    # Fix string "null" → actual None for all string fields
    _NULL_STRINGS = {"null", "none", "n/a", ""}
    for key, val in parsed.items():
        if isinstance(val, str) and val.strip().lower() in _NULL_STRINGS:
            parsed[key] = None

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

    # Fix location: strip extra keys and normalize string "null" to actual None
    loc = parsed.get("location")
    if isinstance(loc, dict):
        def _null_str(v):
            """Convert string 'null'/'None'/'' to actual None."""
            if isinstance(v, str) and v.strip().lower() in ("null", "none", "n/a", ""):
                return None
            return v
        parsed["location"] = {
            "city": _null_str(loc.get("city")),
            "region": _null_str(loc.get("region")),
            "country": _null_str(loc.get("country")),
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

def _post_process(parsed, original_text=""):
    if not isinstance(parsed, dict):
        return parsed, []
    applied = []

    try:
        _fix_employment_type(parsed, original_text)
        applied.append("employment_type_normalize")
    except Exception:
        pass
    try:
        _fix_work_mode(parsed, original_text)
        applied.append("work_mode_normalize")
    except Exception:
        pass
    try:
        _fix_salary(parsed, original_text)
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

    try:
        _fix_soft_skills_hallucination(parsed, original_text)
        applied.append("soft_skills_validation")
    except Exception:
        pass

    try:
        _fix_benefits_hallucination(parsed, original_text)
        applied.append("benefits_validation")
    except Exception:
        pass

    try:
        _fix_clean_list_artifacts(parsed)
        applied.append("clean_list_artifacts")
    except Exception:
        pass

    try:
        _fix_reporting_to_hallucination(parsed, original_text)
        applied.append("reporting_to_validation")
    except Exception:
        pass

    try:
        _fix_company_is_id(parsed, original_text)
        applied.append("company_id_fix")
    except Exception:
        pass

    try:
        _fix_job_id_validation(parsed, original_text)
        applied.append("job_id_validation")
    except Exception:
        pass

    try:
        _fix_job_id_from_client(parsed, original_text)
        applied.append("job_id_from_client")
    except Exception:
        pass

    try:
        _fix_education_format(parsed, original_text)
        applied.append("education_format")
    except Exception:
        pass

    try:
        _fix_experience_from_text(parsed, original_text)
        applied.append("experience_from_text")
    except Exception:
        pass

    try:
        _fix_expiry_date_from_text(parsed, original_text)
        applied.append("expiry_date_from_text")
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


def _fix_employment_type(parsed, original_text=""):
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

    # If LLM said full_time but original text has duration/contract signals, fix it
    lower_text = original_text.lower()
    has_duration = bool(re.search(r'duration\s*[:\-]\s*\d+', lower_text))
    has_contract_signal = any(kw in lower_text for kw in ["c2c", "corp-to-corp", "w2", "corp to corp"])
    has_explicit_fulltime = any(kw in lower_text for kw in ["full-time", "full time", "fulltime"])

    has_contract_text = "contract" in lower_text

    if (has_duration or has_contract_signal or has_contract_text) and not has_explicit_fulltime:
        # Replace full_time with contract
        parsed["employment_type"] = [
            "contract" if t == "full_time" else t
            for t in parsed["employment_type"]
        ]
        # If it was empty or only had full_time, ensure contract is there
        if "contract" not in parsed["employment_type"]:
            parsed["employment_type"].append("contract")
    elif (has_duration or has_contract_signal or has_contract_text) and has_explicit_fulltime:
        # Both contract and full_time are explicitly stated — include both
        if "contract" not in parsed["employment_type"]:
            parsed["employment_type"].append("contract")

    # Dedup
    seen = set()
    parsed["employment_type"] = [
        t for t in parsed["employment_type"]
        if t not in seen and not seen.add(t)
    ]


def _fix_work_mode(parsed, original_text=""):
    wm = parsed.get("work_mode")
    mapping = {
        "remote": "remote", "work from home": "remote", "wfh": "remote",
        "hybrid": "hybrid", "flexible": "hybrid",
        "onsite": "onsite", "on-site": "onsite", "on site": "onsite",
        "in-office": "onsite", "office": "onsite",
    }
    if isinstance(wm, str):
        parsed["work_mode"] = mapping.get(wm.strip().lower(), wm.strip().lower())

    # Override from original text if location line has explicit (REMOTE) or (HYBRID)
    if original_text:
        loc_match = re.search(r'location\s*[:\-]\s*[^(\n]*\((remote|hybrid|onsite|on-site)\)', original_text, re.IGNORECASE)
        if loc_match:
            detected = loc_match.group(1).strip().lower()
            parsed["work_mode"] = mapping.get(detected, detected)
        # Also check "Work Mode:" line
        elif not isinstance(parsed.get("work_mode"), str) or not parsed.get("work_mode"):
            wm_match = re.search(r'work\s*mode\s*[:\-]\s*(remote|hybrid|onsite|on-site)', original_text, re.IGNORECASE)
            if wm_match:
                detected = wm_match.group(1).strip().lower()
                parsed["work_mode"] = mapping.get(detected, detected)


def _fix_salary(parsed, original_text=""):
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

    # Fix: LLM sometimes multiplies hourly rates by 1000 (e.g., $70/hr → 70000)
    # If period is "hour" and values are unreasonably high, divide by 1000
    period = sal.get("period", "year")
    if period == "hour":
        for key in ("min", "max"):
            val = sal.get(key)
            if isinstance(val, (int, float)) and val > 500:
                sal[key] = val / 1000
                # Convert to int if it's a whole number
                if sal[key] == int(sal[key]):
                    sal[key] = int(sal[key])


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


def _fix_soft_skills_hallucination(parsed, original_text=""):
    """Remove soft skills that don't actually appear in the original JD text."""
    soft = parsed.get("soft_skills")
    if not isinstance(soft, list) or not original_text:
        return
    lower_text = original_text.lower()
    # Known soft skill keywords to validate against
    _VALID_SOFT_SKILLS = {
        "communication", "leadership", "teamwork", "problem-solving", "problem solving",
        "analytical", "interpersonal", "collaboration", "presentation", "negotiation",
        "mentoring", "coaching", "adaptability", "creativity", "critical thinking",
        "decision making", "decision-making", "time management", "organizational",
        "attention to detail", "troubleshooting", "documentation", "customer-focused",
        "self-starter", "multitasking", "conflict resolution", "strategic thinking",
    }
    validated = []
    for s in soft:
        if not isinstance(s, str):
            continue
        s_lower = s.lower().strip()
        # Must appear in text AND be a recognized soft skill
        if s_lower in lower_text and any(kw in s_lower for kw in _VALID_SOFT_SKILLS):
            validated.append(s)
    parsed["soft_skills"] = validated


def _fix_benefits_hallucination(parsed, original_text=""):
    """Remove benefits that don't actually appear in the original JD text."""
    benefits = parsed.get("benefits")
    if not isinstance(benefits, list) or not original_text:
        return
    lower_text = original_text.lower()
    validated = [b for b in benefits if isinstance(b, str) and b.lower() in lower_text]
    parsed["benefits"] = validated


def _fix_reporting_to_hallucination(parsed, original_text=""):
    """Remove reporting_to if it's not actually stated with 'Reports to' or 'Reporting To' in the text."""
    reporting = parsed.get("reporting_to")
    if not reporting or not original_text:
        return
    lower_text = original_text.lower()
    # Check if the JD explicitly has a reporting_to section
    has_reporting_section = any(kw in lower_text for kw in [
        "reporting to", "reports to", "report to:", "manager:",
        "reporting to:", "reports to:",
    ])
    if not has_reporting_section:
        parsed["reporting_to"] = None


def _fix_clean_list_artifacts(parsed):
    """Clean form artifacts like 'Yes/No' prefix from list items."""
    for field in ("preferred_experience", "preferred_technologies", "requirements",
                  "responsibilities", "benefits", "certifications"):
        items = parsed.get(field)
        if not isinstance(items, list):
            continue
        cleaned = []
        for item in items:
            if isinstance(item, str):
                # Strip "Yes/No " prefix from form fields
                item = re.sub(r'^(Yes/No\s+)', '', item, flags=re.IGNORECASE).strip()
                # Strip leading "X years of " or "X-Y years of " for preferred items
                # (keep the content, strip the year prefix only if it's redundant)
                if item:
                    cleaned.append(item)
        parsed[field] = cleaned


def _fix_location_consistency(parsed):
    loc = parsed.get("location")
    wm = parsed.get("work_mode")
    if isinstance(loc, dict) and isinstance(wm, str):
        loc["remote"] = wm

    # Fix country/region confusion: if country is empty but region looks like a country, swap
    if isinstance(loc, dict):
        country = (loc.get("country") or "").strip()
        region = (loc.get("region") or "").strip()
        _COUNTRIES = {
            "india", "usa", "us", "united states", "united kingdom", "uk",
            "canada", "australia", "germany", "france", "singapore", "japan",
            "china", "brazil", "uae", "netherlands", "ireland", "sweden",
            "switzerland", "israel", "south korea", "new zealand", "spain",
            "italy", "mexico", "poland", "denmark", "norway", "finland",
        }
        if region and not country and region.lower() in _COUNTRIES:
            loc["country"] = region
            loc["region"] = None
        if not country and not region:
            # Try to extract from formatted_address
            addr = loc.get("formatted_address", "")
            parts = [p.strip() for p in addr.split(",") if p.strip()]
            if len(parts) >= 2 and parts[-1].lower() in _COUNTRIES:
                loc["country"] = parts[-1]


def _fix_company_is_id(parsed, original_text=""):
    """If company is a numeric/short alphanumeric ID, move it to job_id and null out company."""
    company = parsed.get("company")
    if not isinstance(company, str):
        return
    clean = company.strip()
    # Remove "Client:" prefix if present
    if clean.lower().startswith("client:"):
        clean = clean[7:].strip()
        parsed["company"] = clean

    # Check if it looks like an ID rather than a company name
    is_id = False
    # Pure digits
    if clean.isdigit():
        is_id = True
    # Short alphanumeric with no spaces and contains digits (e.g., "17783PSA3", "537601527", "DAG2610")
    elif re.match(r'^[A-Za-z0-9]{4,20}$', clean) and re.search(r'\d', clean) and " " not in clean:
        is_id = True

    if is_id:
        if not parsed.get("job_id"):
            parsed["job_id"] = clean
        parsed["company"] = None


def _fix_job_id_validation(parsed, original_text=""):
    """If job_id looks like a title or description (too long), null it out."""
    jid = parsed.get("job_id")
    if not isinstance(jid, str):
        return
    # Job IDs are typically short alphanumeric codes (<30 chars)
    if len(jid) > 30:
        parsed["job_id"] = None
    # If it contains only common words with spaces, it's likely a title
    if " " in jid and not re.search(r'\d', jid):
        parsed["job_id"] = None


def _fix_education_format(parsed, original_text=""):
    """Clean malformed education level strings."""
    edu = parsed.get("education")
    if not isinstance(edu, dict):
        return
    level = edu.get("level")
    if isinstance(level, str):
        # Fix patterns like 'Bachelor's"|"Master's' → "Bachelor's or Master's"
        if '"|"' in level or '"|' in level or '|"' in level:
            level = re.sub(r'["\|]+', ' or ', level).strip()
            level = re.sub(r'\s+', ' ', level)
            edu["level"] = level


def _fix_experience_from_text(parsed, original_text=""):
    """If experience_years is missing, try to extract from text."""
    if parsed.get("experience_years"):
        return
    if not original_text:
        return
    # Look for "X+ years", "X-Y years", "X years" patterns
    patterns = [
        r'(\d+)\s*[\-–to]+\s*(\d+)\s*(?:\+\s*)?years?\s+(?:of\s+)?experience',
        r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
        r'minimum\s+(?:of\s+)?(\d+)\s*years',
    ]
    for pat in patterns:
        match = re.search(pat, original_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                mn, mx = int(groups[0]), int(groups[1])
            else:
                mn = mx = int(groups[0])
            parsed["experience_years"] = {
                "min_years": mn,
                "max_years": mx,
                "requirement_type": "required",
            }
            return


def _fix_job_id_from_client(parsed, original_text=""):
    """Extract job_id from Client: field patterns like 'Client: CompanyName -(ID)' or 'Client: CompanyName (ID)'."""
    if parsed.get("job_id"):
        return
    if not original_text:
        return
    # Pattern: Client: ... -(ID) or Client: ... (ID)
    match = re.search(r'client\s*:\s*[^(\n]*[\-–]\s*\(([^)]+)\)', original_text, re.IGNORECASE)
    if match:
        parsed["job_id"] = match.group(1).strip()
        return
    # Pattern: Client: ... (ID) where ID is alphanumeric
    match = re.search(r'client\s*:\s*\w[^(\n]*\((\w+)\)', original_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        # Only use if it looks like an ID (not a word like "Hybrid")
        if re.match(r'^[A-Z0-9]{4,}', candidate):
            parsed["job_id"] = candidate


def _fix_expiry_date_from_text(parsed, original_text=""):
    """Extract job_expiry_date from 'Due date:' field if not already set."""
    if parsed.get("job_expiry_date"):
        return
    if not original_text:
        return
    match = re.search(r'due\s*date\s*:\s*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', original_text, re.IGNORECASE)
    if match:
        m, d, y = match.group(1), match.group(2), match.group(3)
        parsed["job_expiry_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"


# ---------------------------------------------------------------------------
# Main parse function — single-pass extraction + post-processing
# ---------------------------------------------------------------------------

def parse_jd(jd_text, filename="unknown", model=None, api_key=None):
    """Parse a job description using single-pass LLM extraction.

    Single LLM call with detailed prompt, followed by robust
    normalization and post-processing in Python.
    """
    key = api_key or GROQ_API_KEY
    mdl = model or GROQ_MODEL

    if not key:
        return {"error": "GROQ_API_KEY not set."}

    start = time.time()

    # --- LLM Extraction ---
    prompt = EXTRACTION_PROMPT.replace("JD_TEXT_HERE", jd_text)
    try:
        content, usage, finish_reason = _call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=mdl, api_key=key, temperature=0.1, max_tokens=32768,
        )
    except Exception as e:
        return {"error": f"LLM extraction failed: {str(e)}"}

    # If output was truncated (model stuck in repetition loop), retry with
    # temperature=0 and frequency_penalty to break the loop.
    if finish_reason == "length":
        concise_note = (
            "\n\nIMPORTANT: Be CONCISE. For the skills array, list each skill ONCE "
            "with its single best category. Maximum 30 skills. Do NOT repeat any skill."
        )
        try:
            content, usage, finish_reason = _call_groq(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt + concise_note},
                ],
                model=mdl, api_key=key, temperature=0.0, max_tokens=32768,
                frequency_penalty=0.5,
            )
        except Exception:
            pass  # Fall through to parse whatever we got from the first attempt

    extracted = _extract_json(content)
    if extracted is None:
        return {"error": "Failed to parse JSON from model response", "raw": content}

    # --- Normalize + Post-process ---
    extracted = _normalize_llm_output(extracted)
    extracted, pp_applied = _post_process(extracted, jd_text)

    elapsed_ms = int((time.time() - start) * 1000)

    metadata = {
        "parser": "groq_jd_parser",
        "model": mdl,
        "processing_time_ms": elapsed_ms,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "_post_processed": pp_applied,
    }

    # Set metadata fields that are derived, not extracted
    extracted["source_type"] = "file"
    extracted["language_detected"] = "en"

    return _build_output(jd_text, extracted, filename, metadata)


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
