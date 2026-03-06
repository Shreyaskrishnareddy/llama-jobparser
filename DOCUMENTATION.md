# Llama JD Parser — Technical Documentation

> LLM-powered job description parser using Llama 3.1 8B via Groq for structured data extraction.
> **Version**: 1.0.0
> **Last Updated**: 2026-03-02
> **Repository**: [github.com/Shreyaskrishnareddy/llama-jobparser](https://github.com/Shreyaskrishnareddy/llama-jobparser)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [API Reference](#4-api-reference)
5. [Data Schema](#5-data-schema)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Model Selection & Tradeoffs](#7-model-selection--tradeoffs)
8. [Rate Limits & Constraints](#8-rate-limits--constraints)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Known Limitations](#11-known-limitations)
12. [Configuration Reference](#12-configuration-reference)
13. [Project Structure](#13-project-structure)
14. [Changelog](#14-changelog)

---

## 1. Overview

### What It Does

This application parses job descriptions (PDF, DOCX, DOC, TXT, images) into structured JSON using Llama 3.1 8B running on Groq's LPU inference engine. It extracts 35 data fields across 7 categories: job details, company information, location, compensation, requirements, skills, and metadata. Each extracted field includes a confidence score and provenance (character spans tracing back to the original text).

### Why This Approach

Traditional JD parsers rely on regex and keyword matching, which break on non-standard formatting, varying section headers, and inconsistent layouts. LLM-based parsing handles arbitrary JD structures because the model understands context. Groq's hardware inference makes this fast enough for production (~3-8 seconds per JD).

### How It Works

```
JD File (PDF / DOCX / DOC / TXT / Image)
    |
    v
Text Extraction (PyMuPDF / docx2txt / antiword / Tesseract OCR)
    |
    v
Full text sent to Llama 3.1 8B via Groq API (single-pass, no chunking)
    |
    v
Robust JSON Extraction (handles markdown fences, malformed output)
    |
    v
Normalization (fix types, strip extra keys, string "null" → None)
    |
    v
Post-Processing (10 deterministic fixes for known LLM error patterns)
    |
    v
Confidence Scoring + Provenance (character spans in original text)
    |
    v
Structured JSON Response + Metadata
```

---

## 2. Architecture

### System Design

```
                    +------------------+
                    |   Web UI (HTML)  |
                    |  Drag & Drop     |
                    +--------+---------+
                             |
                             v
+----------------------------+----------------------------+
|                     Flask API Server (app.py)            |
|                                                          |
|  /parse          Single file upload                      |
|  /parse/text     Raw text input                          |
|  /parse/bulk     Up to 50 files (3 concurrent workers)   |
|  /health         Health check                            |
+----------------------------+----------------------------+
                             |
                             v
+----------------------------+----------------------------+
|             groq_jd_parser.py (Core Logic)               |
|                                                          |
|  extract_text_from_file()  Text extraction dispatcher    |
|  parse_jd()                Groq API call + prompt        |
|  _extract_json()           Robust JSON parser            |
|  _normalize_llm_output()   Type fixing + key stripping   |
|  _post_process()           10 deterministic corrections  |
|  _build_output()           Confidence + provenance       |
+----------------------------+----------------------------+
                             |
                             v
              +-----------------------------+
              |   Groq API (External)       |
              |   Model: llama-3.1-8b       |
              |   Endpoint: groq.com/openai |
              +-----------------------------+
```

### Request Flow

1. Client uploads file to `/parse`
2. `extract_text_from_file()` converts to raw text using the appropriate library
3. Full text is injected into the prompt template (no truncation, no chunking)
4. Single API call to Groq with the system prompt + user prompt (temperature 0.1)
5. `_extract_json()` extracts JSON from the LLM response (handles edge cases)
6. `_normalize_llm_output()` fixes types (company as string, skills schema, string "null" → None)
7. `_post_process()` applies 10 deterministic corrections for known LLM error patterns
8. `_build_output()` wraps each field with confidence scores and provenance spans
9. Metadata (model, timing, tokens, post-processing flags) is attached
10. JSON response returned to client

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-pass (no chunking) | JDs are shorter than resumes and always fit in context. Full text gives the LLM complete picture. |
| Prompt-first, code-second | Accuracy issues are addressed via prompt engineering first. Code-level post-processing is only added when the 8B model proves stubbornly wrong after multiple prompt attempts. |
| Post-processing over re-prompting | The 8B model consistently makes the same mistakes (employment type, soft skills hallucination, salary math). Fixing them in Python is faster and more reliable than multi-turn prompting. |
| Confidence + provenance | Each field carries a confidence score and character spans into the original text, enabling downstream systems to verify extractions. |
| No database | Stateless API. The parser doesn't store JDs or results. Keeps it simple and avoids data storage concerns. |
| Gunicorn with 2 workers | Matches Groq's rate limits. More workers would just hit 429 errors. |
| Vanilla JS frontend | No build step, no dependencies. The UI is a single HTML file served by Flask. |
| Retry with backoff | Unlike the resume parser, the JD parser includes automatic retry (up to 5 attempts) with wait time extraction from Groq 429 responses. |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Llama 3.1 8B Instant | JD text → structured JSON |
| **Inference** | Groq API (LPU) | Fast inference (~3-8s per JD) |
| **Backend** | Flask 3.x + Gunicorn | API server |
| **Frontend** | Vanilla HTML/CSS/JS | Drag-and-drop web UI |
| **PDF Parsing** | PyMuPDF (fitz) | PDF text extraction |
| **DOCX Parsing** | docx2txt | DOCX text extraction |
| **DOC Parsing** | antiword + olefile | Legacy .doc support |
| **OCR** | Tesseract + Pillow | Scanned PDFs and images |
| **Taxonomy** | groq-taxonomy (shared library) | Skill/title/cert/degree/industry normalization |
| **Deployment** | Docker / Render | Production hosting |

### Dependencies (requirements.txt)

```
flask>=3.0
flask-cors>=4.0
python-dotenv>=1.0
requests>=2.31
PyMuPDF>=1.23
python-docx>=1.0
docx2txt>=0.8
pytesseract>=0.3
Pillow>=10.0
olefile>=0.47
gunicorn>=21.2
openpyxl>=3.1
groq-taxonomy @ git+https://github.com/Shreyaskrishnareddy/monorepo.git#subdirectory=packages/taxonomy
```

Post-processing uses only stdlib (`re`, `uuid`, `datetime`, `time`, `json`) — no additional dependencies.

### Shared Library: groq-taxonomy

The project uses `groq-taxonomy`, a shared normalization library from the team monorepo, to enrich parsed JD data with canonical vocabularies.

**Source:** `git+https://github.com/Shreyaskrishnareddy/monorepo.git#subdirectory=packages/taxonomy`
**Version:** 0.1.0
**Dependencies:** None (zero external dependencies)

#### What It Does

After the LLM parses a JD, post-processing runs, and confidence/provenance is built, `enrich_jd()` normalizes free-text fields to canonical IDs. This enables structured matching between job descriptions and resumes. The import is wrapped in `try/except ImportError` so the parser works without it installed.

```python
from groq_taxonomy import enrich_jd
output = enrich_jd(output)  # adds _taxonomy key, leaves original fields untouched
```

`enrich_jd()` handles the JD parser's `{value, confidence, provenance}` envelope format, unwrapping values before normalization.

#### Taxonomy Modules

| Module | Data File | Entries | Key Functions |
|--------|-----------|---------|---------------|
| **skills** | `skills.json` | 329 skills | `normalize_skill(name)` → canonical ID; `classify_skill(id)` → display name, category, subcategory, domain, type, related skills |
| **titles** | `titles.json` | 45 titles + 11 seniority patterns | `normalize_title(text)` → canonical ID, display name, seniority level + weight, function |
| **education** | `degrees.json` + `fields_of_study.json` | 6 degree levels + 33 fields | `normalize_degree(degree, field)` → level, weight, display name, field |
| **certifications** | `certifications.json` | 56 certs | `normalize_cert(text)` → canonical ID, display name, issuer, domain, category |
| **industries** | `industries.json` | 14 industries | `classify_industry(text)` → industry ID; `classify_industry_multi(text)` → ranked list with scores |

#### Text Normalization

The library uses multi-tier text matching (`_text.py`):

- **Standard key** (`make_lookup_key`): lowercased, preserves `#`, `+`, `.` (so "C#" and "C++" stay distinct)
- **Aggressive key** (`make_alias_key`): strips dots/hyphens so "React.js", "reactjs", "react-js" all unify
- **Version stripping**: "Python 3.9" → "python", "Angular 11" → "angular"
- **Suffix variations**: tries "js", ".js", "lang" suffixes for broader matching

#### Skill Categories

Each skill record includes: `id`, `display_name`, `aliases` (e.g., "k8s" → kubernetes), `category` (programming_language, framework, cloud, database, etc.), `subcategory`, `domain` (software_engineering, data_science, etc.), `type` (technical/soft), and `related` skill IDs.

#### Title Seniority Detection

Titles are decomposed into a base role (e.g., `software_engineer`) plus a seniority level with numeric weights:

| Seniority | Weight |
|-----------|--------|
| Intern | 0 |
| Junior | 1 |
| Mid | 2 |
| Senior | 3 |
| Staff | 4 |
| Principal | 5 |
| Distinguished | 6 |
| VP | 7 |
| Director | 8 |
| CTO/CIO | 9 |

#### Degree Levels

| Level | Weight |
|-------|--------|
| High School | 0 |
| Certificate | 1 |
| Associate | 2 |
| Bachelors | 3 |
| Masters | 4 |
| Doctorate | 5 |

#### Enrichment Output

All enrichment data is placed in a `_taxonomy` key on the parsed result, leaving original fields untouched. The enrichment includes:

1. **Skills**: Deduplicated by canonical ID with category/subcategory/domain/type/related metadata
2. **Job title**: Normalized with seniority detection
3. **Education**: Degree level + field of study normalization
4. **Certifications**: Canonical names with issuer and domain
5. **Industry**: Classification from the job_domain field
6. **Skill summary**: Counts by category and domain

---

## 4. API Reference

### Security Headers

All responses include:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store
```

---

### `GET /health`

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "parser": "groq_jd_parser",
  "groq_configured": true,
  "model": "llama-3.1-8b-instant",
  "supported_formats": ["bmp","doc","docx","htm","html","jpeg","jpg","pdf","png","tiff","txt"],
  "max_bulk_files": 50,
  "timestamp": 1772335017.85
}
```

---

### `POST /parse`

Parse a single uploaded JD file.

**Request:** `multipart/form-data` with key `file`

```bash
curl -X POST http://localhost:8001/parse -F "file=@job_description.pdf"
```

**Response (200):**
```json
{
  "filename": "job_description.pdf",
  "processing_time_ms": 5200,
  "result": {
    "id": "a1b2c3d4-...",
    "source": {
      "type": "file",
      "filename": "job_description.pdf",
      "url": null,
      "uploaded_at": "2026-03-02T12:00:00+00:00"
    },
    "detected_language": "en",
    "global_confidence": 0.93,
    "fields": {
      "title": {
        "value": {"text": "Business Analyst", "seniority_level": "Mid", "domain": "data"},
        "confidence": 0.95,
        "provenance": {"spans": [[0, 16]], "extractor": "job_title_extractor", ...},
        "status": "ok"
      },
      "company": {
        "value": "Arytic Inc.",
        "confidence": 0.95,
        ...
      }
    },
    "_metadata": {
      "parser": "groq_jd_parser",
      "model": "llama-3.1-8b-instant",
      "processing_time_ms": 5100,
      "finish_reason": "stop",
      "prompt_tokens": 2500,
      "completion_tokens": 3200,
      "total_tokens": 5700,
      "_post_processed": ["employment_type_normalize", "work_mode_normalize", "salary_numbers", "skill_dedup", "location_consistency", "derive_skill_splits", "soft_skills_validation", "benefits_validation", "clean_list_artifacts", "reporting_to_validation"]
    }
  }
}
```

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | No file, empty filename, unsupported format |
| 502 | Groq API error (rate limit, model error, timeout) |

**Supported Formats:** PDF, DOC, DOCX, TXT, HTML, HTM, JPG, JPEG, PNG, TIFF, BMP
**Max File Size:** 10 MB

---

### `POST /parse/text`

Parse raw JD text (no file upload).

**Request:** `application/json`

```bash
curl -X POST http://localhost:8001/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Job Title: Business Analyst\nCompany: Arytic Inc.\n..."}'
```

**Response:** Same structure as `/parse`, without `filename` wrapper.

**Constraints:** Text must be at least 30 characters.

---

### `POST /parse/bulk`

Parse up to 50 JD files concurrently.

**Request:** `multipart/form-data` with key `files` (multiple)

```bash
curl -X POST http://localhost:8001/parse/bulk \
  -F "files=@jd1.pdf" \
  -F "files=@jd2.docx"
```

**Response (200):**
```json
{
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "total_processing_time_ms": 12000,
  "results": [ ... ]
}
```

**Constraints:** Max 50 files, 50 MB total. Uses 3 concurrent workers.

---

## 5. Data Schema

### Output Wrapper

Every parsed JD is wrapped in a standard envelope:

```json
{
  "id": "uuid",
  "source": {
    "type": "file",
    "filename": "string",
    "url": null,
    "uploaded_at": "ISO 8601 timestamp"
  },
  "detected_language": "en",
  "global_confidence": 0.93,
  "fields": { ... },
  "_metadata": { ... }
}
```

### Field Wrapper

Each field in `fields` is either `null` (not found) or wrapped with confidence and provenance:

```json
{
  "value": "the extracted value",
  "confidence": 0.95,
  "provenance": {
    "spans": [[start_char, end_char]],
    "extractor": "extractor_name",
    "extractor_version": "1.0.0",
    "rule_id": null,
    "extracted_text": "matched text snippet"
  },
  "status": "ok"
}
```

### All 35 Fields

```json
{
  "title": {
    "text": "string (exact job title)",
    "seniority_level": "Senior | Lead | Junior | Mid | Entry | null",
    "domain": "cloud | data | engineering | security | management | IT | networking | devops | support | operations"
  },
  "company": "string | null",
  "location": {
    "city": "string | null",
    "region": "string (state/province) | null",
    "country": "string (nation) | null",
    "remote": "remote | hybrid | onsite",
    "formatted_address": "string"
  },
  "employment_type": ["full_time", "contract", "part_time", "internship", "temporary", "freelance"],
  "salary": {
    "min": "number",
    "max": "number",
    "currency": "USD | INR | EUR | GBP",
    "period": "year | month | hour",
    "ote": false
  },
  "requirements": ["string (each requirement bullet)"],
  "responsibilities": ["string (each responsibility bullet)"],
  "skills": [{"name": "string", "category": "programming_language | framework | database | cloud | devops | tool | methodology | domain | soft_skill | networking | os | other"}],
  "technical_skills": ["string"],
  "soft_skills": ["string"],
  "education": {
    "level": "Bachelor's | Master's | PhD | etc",
    "field": "string"
  },
  "experience_years": {
    "min_years": "number",
    "max_years": "number",
    "requirement_type": "required | preferred"
  },
  "benefits": ["string"],
  "work_authorization": "string | null",
  "job_domain": "string (e.g. Information Technology)",
  "job_summary": "string (1-3 sentence overview)",
  "description": "string (full detailed description)",
  "job_id": "string | null",
  "work_mode": "remote | hybrid | onsite",
  "job_posted_date": "YYYY-MM-DD | null",
  "job_expiry_date": "YYYY-MM-DD | null",
  "reporting_to": "string | null",
  "team_size": "string | null",
  "travel_requirement": "string | null",
  "application_link": "URL | null",
  "equal_opportunity_statement": "string | null",
  "company_website": "URL | null",
  "industry": "string | null",
  "company_size": "string | null",
  "company_overview": "string | null",
  "preferred_experience": ["string"],
  "preferred_technologies": ["string"],
  "certifications": ["string"],
  "source_type": "file",
  "language_detected": "en"
}
```

### Field Rules (enforced via prompt + post-processing)

| Rule | Details |
|------|---------|
| Location splitting | "country" = nation (India, USA), "region" = state/province (TX, Telangana). Post-processing swaps if wrong. |
| Employment type | Detected from "Duration" fields, "C2C"/"W2" keywords. Never defaults to "full_time" unless explicitly stated. |
| Salary | Plain numbers kept as-is ($70/hr = 70). Only abbreviations expanded (K=thousands, LPA=lakhs). Hourly rates >500 auto-corrected. |
| Skills | Exhaustive extraction from all sections. Alternatives split ("Power BI or Tableau" → 2 skills). Parenthetical tech extracted. |
| Soft skills | Anti-hallucination: must literally appear in JD text AND match known soft skill keywords. |
| Benefits | Anti-hallucination: must literally appear in JD text. Vague phrases ("as per agreement") → empty array. |
| Reporting to | Anti-hallucination: only included if explicit "Reports to"/"Reporting To" label found in text. |
| Requirements vs Preferred | Strict section separation. Required items → requirements, preferred items → preferred_experience/preferred_technologies. |
| Job ID | Extracted from "Client:", "Job ID:", "Reference ID:" fields. If combined with company name, ID is separated. |

---

## 6. Post-Processing Pipeline

### Why Post-Processing Is Needed

The Llama 3.1 8B model produces good structured JSON but consistently makes certain types of errors that cannot be fixed through prompt engineering alone. After testing all 18 sample JDs, we identified 10 error patterns requiring code-level fixes.

### Approach

Prompt engineering is tried first for every issue. Code-level post-processing is only added when the same issue persists after multiple prompt iterations. This keeps the code minimal and the prompt doing most of the work.

### Fix 1: `_fix_employment_type()`

**What:** Detects contract roles from text signals and normalizes employment type values.

**How:**
1. Normalizes all values to standard form ("Full-time" → "full_time", "C2C" → "contract")
2. Searches original text for duration patterns (`duration: 12 months`), contract keywords ("C2C", "W2", "corp-to-corp"), and "contract" mentions
3. If contract signals found without explicit "full-time" text → replaces "full_time" with "contract"
4. If both contract signals AND explicit "full-time" found → includes both
5. Deduplicates the array

**Why prompt failed:** The 8B model stubbornly defaults to "full_time" even when "Duration: 22+ Months" is clearly stated. Tested 3 prompt iterations before adding code fix.

### Fix 2: `_fix_work_mode()`

**What:** Normalizes work mode values and overrides from location line patterns.

**How:**
1. Maps variations to standard values ("work from home" → "remote", "in-office" → "onsite")
2. Searches original text for `Location: ... (REMOTE)` or `Location: ... (Hybrid)` patterns
3. If found, overrides the LLM's work_mode with the detected value

**Why needed:** LLM sometimes returns "hybrid" when the location line explicitly says "(Remote)".

### Fix 3: `_fix_salary()`

**What:** Fixes salary value types and corrects hourly rate multiplication errors.

**How:**
1. Converts string salary values to numbers (strips "$", ",", etc.)
2. If period is "hour" and value > 500, divides by 1000 (e.g., 70000 → 70)

**Why needed:** The LLM sometimes interprets "$70/hr" as 70000 instead of 70, applying thousand-multiplication that the prompt says not to do.

### Fix 4: `_fix_skill_dedup()`

**What:** Removes duplicate skills (case-insensitive).

### Fix 5: `_fix_location_consistency()`

**What:** Fixes country/region confusion and syncs work_mode with location.remote.

**How:**
1. Syncs `location.remote` with `work_mode` value
2. If region contains a country name (e.g., "India" in region field) and country is empty, swaps them
3. If no country set, tries to extract from formatted_address
4. Checks against a known list of 25+ country names

**Why needed:** The LLM frequently puts country names in the region field (e.g., "India" in region instead of country).

### Fix 6: `_fix_derive_skill_splits()`

**What:** Derives `technical_skills` and `soft_skills` arrays from the `skills` array if not already populated by the LLM.

### Fix 7: `_fix_soft_skills_hallucination()`

**What:** Removes hallucinated soft skills that don't appear in the original JD text.

**How:** Two-layer validation:
1. The soft skill text must appear (case-insensitive) in the original JD text
2. AND it must match at least one keyword in a curated set of 25+ known soft skills (communication, leadership, teamwork, problem-solving, analytical, etc.)

**Why prompt failed:** Even with explicit "NEVER guess or infer soft skills" instructions, the 8B model fabricates skills like "Problem Solving" from job responsibility descriptions. Tested 3 prompt iterations before adding code fix.

### Fix 8: `_fix_benefits_hallucination()`

**What:** Removes hallucinated benefits that don't appear in the original JD text.

**How:** Simple text-presence check — each benefit string must appear (case-insensitive) in the original JD. Benefits not found in text are removed.

**Why needed:** When a JD says "benefits as per agreement", the LLM invents specific benefits like "health insurance", "401k", etc.

### Fix 9: `_fix_clean_list_artifacts()`

**What:** Strips form artifacts from list items.

**How:** Removes "Yes/No " prefixes that appear in form-based JDs where "Yes/No" is a column header that bleeds into the content.

### Fix 10: `_fix_reporting_to_hallucination()`

**What:** Removes hallucinated reporting_to values when no explicit label exists.

**How:** Checks if the original text contains any of: "reporting to", "reports to", "report to:", "manager:", "reporting to:", "reports to:". If none found, sets reporting_to to null.

**Why needed:** The LLM guesses reporting_to from company names or client names (e.g., "Austin Energy (AE)" as reporting_to).

### Normalization Layer

Before post-processing, `_normalize_llm_output()` handles structural issues:

| Fix | Details |
|-----|---------|
| String "null" → None | Converts literal string "null", "none", "n/a", "" to actual JSON null for all fields |
| Company type | If company is an object `{"text": "X"}`, extracts the string |
| List items type | If list items are objects instead of strings, extracts text values |
| Skills schema | Strips extra keys, keeps only `name` and `category` |
| Salary schema | Strips extra keys beyond min/max/currency/period/ote |
| Location schema | Strips extra keys, converts string "null" values in city/region/country |
| Title schema | Ensures `{text, seniority_level, domain}` structure |

### Metadata

All post-processing results are tracked in `_metadata._post_processed`:

```json
"_metadata": {
  "_post_processed": [
    "employment_type_normalize",
    "work_mode_normalize",
    "salary_numbers",
    "skill_dedup",
    "location_consistency",
    "derive_skill_splits",
    "soft_skills_validation",
    "benefits_validation",
    "clean_list_artifacts",
    "reporting_to_validation"
  ]
}
```

If a fix fails (bad data), it's silently skipped and omitted from the list.

---

## 7. Model Selection & Tradeoffs

### Why llama-3.1-8b-instant

The same model was chosen as the companion resume parser project, for consistency and proven performance:

| Factor | 8B Model | Larger Models (70B, Scout 17B) |
|--------|----------|-------------------------------|
| **Rate limits (RPD)** | 14,400 | 1,000 |
| **Tokens/day** | 500,000 | 100,000-500,000 |
| **Speed** | ~3-8s per JD | ~5-10s per JD |
| **Extraction completeness** | High (with post-processing) | Higher baseline but limited requests |
| **Post-processing needed** | Yes (10 fixes) | Fewer fixes needed |

**Verdict:** The 8B model extracts sufficient data for all 35 fields. Its consistent error patterns (employment type defaults, hallucination of soft skills/benefits) are fully corrected by deterministic post-processing. The higher rate limits (14x more daily requests than 70B) make it practical for production.

### Tradeoff Summary

```
Chose: Completeness + high throughput (8B) + deterministic post-processing
Over:  Smarter model (70B) with fewer errors but severe rate limits

Reasoning:
- JDs are simpler than resumes — fewer edge cases in text structure
- 10 post-processing fixes handle all identified error patterns
- Post-processing is cheap (microseconds, no API calls)
- 8B has 14.4K RPD vs 70B's 1K RPD (14x more daily requests)
```

---

## 8. Rate Limits & Constraints

### Groq Free Tier Limits (llama-3.1-8b-instant)

| Limit | Value | Impact |
|-------|-------|--------|
| **TPM** (tokens/min) | 6,000 | Main bottleneck. A single JD uses ~3K-6K tokens. |
| **RPM** (requests/min) | 30 | Not a bottleneck in practice. |
| **RPD** (requests/day) | 14,400 | Sufficient for moderate workloads. |
| **TPD** (tokens/day) | 500,000 | ~80-160 JDs/day depending on length. |

### Request Size Limits

| Constraint | Value |
|-----------|-------|
| Max tokens per request | ~6K prompt (JDs are typically shorter than resumes) |
| Max file size | 10 MB |
| Max bulk files | 50 per request |
| Bulk total size | 50 MB all files combined |
| API timeout | 120 seconds per Groq API call |
| Retry attempts | 5 (with backoff from 429 responses) |

### Rate Limit Handling

Unlike the resume parser, the JD parser includes automatic retry logic:

1. On 429 (rate limited), extracts wait time from error message (`try again in Xs`)
2. Waits the specified time + 1 second buffer
3. Retries up to 5 times before failing
4. Timeout per request is 120 seconds (vs 60s in resume parser)

---

## 9. Deployment

### Render (Current)

The project deploys to Render via `render.yaml`:

```yaml
services:
  - type: web
    name: llama-jobparser
    runtime: python
    pythonVersion: "3.11.12"
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 180
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GROQ_MODEL
        value: llama-3.1-8b-instant
      - key: PYTHON_VERSION
        value: "3.11.12"
    healthCheckPath: /health
```

**Steps:**
1. Push code to GitHub
2. Create Web Service on [dashboard.render.com](https://dashboard.render.com)
3. Connect repository, select **Python** runtime
4. Set `GROQ_API_KEY` environment variable
5. Set Python version to 3.11.12 (Render defaults to 3.14 which has compatibility issues)
6. Change health check path to `/health`
7. Deploy

**Note:** Free tier instances spin down after inactivity. First request after idle will take 30-60 seconds (cold start). Python version must be pinned to 3.11.x — Render's default 3.14 causes `ModuleNotFoundError` with some dependencies.

**Live URL:** https://llama-jobparser.onrender.com/

### Docker

```bash
docker build -t llama-jobparser .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key llama-jobparser
```

The Dockerfile includes system dependencies (antiword for .doc, tesseract for OCR) and uses Python 3.11-slim.

### Local Development

```bash
pip install -r requirements.txt

# Create .env file with your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

python app.py
# Server starts on http://localhost:8001 with debug mode
```

---

## 10. Testing

### Test Suite

The parser was tested against **18 unique job descriptions** covering diverse formats and domains:

| JD# | Title | Fields Extracted | Format |
|-----|-------|-----------------|--------|
| 1 | Business Analyst (Mid Senior) | 17/35 | PDF + DOCX |
| 2 | Information Security Officer | 32/35 | PDF + DOCX |
| 3 | Production/Operations Support Analyst | 15/35 | PDF + DOCX |
| 4 | Senior Network Engineer | 28/35 | PDF + DOCX |
| 5 | Full Stack Developer | 19/35 | PDF + DOCX |
| 6 | Cloud Solutions Architect | 28/35 | PDF + DOCX |
| 7 | BA Job Description | 26/35 | PDF + DOCX |
| 8 | Technical Architect | 26/35 | PDF |
| 9 | Senior Systems Analyst | 17/35 | PDF + DOCX |
| 10 | Project Lead (PMP) | 25/35 | PDF + DOCX |
| 11 | Job Description - Business Analyst | 22/35 | PDF + DOCX |
| 12 | Project Manager (Energy & Utility) | 25/35 | PDF + DOCX |
| 13 | Developer/Programmer Analyst (Power Platform) | 24/35 | PDF + DOCX |
| 14 | Senior Data Administrator | 29/35 | PDF + DOCX |
| 15 | Embedded Systems Engineer | 21/35 | PDF + DOCX |
| 16 | Support Technician | 19/35 | PDF |
| 17 | Project Manager (PMP) | 29/35 | PDF + DOCX |
| 18 | Full Stack Developer (EU) | 17/35 | PDF |

**Key:** Fields not filled (e.g., 17/35 means 18 null) are correctly null because the data is not present in the JD text. A field is only counted as a failure if data exists in the JD but was not extracted or was extracted incorrectly.

### Issues Found and Fixed During Testing

| Issue | JDs Affected | Fix Type | Resolution |
|-------|-------------|----------|------------|
| Country/region swap | #1, #2 | Prompt + Code | Prompt clarification + `_fix_location_consistency` |
| Employment type defaults to full_time | #3, #7 | Code | `_fix_employment_type` with text signal detection |
| Soft skills hallucination | #3, #7, #14 | Code | `_fix_soft_skills_hallucination` with two-layer validation |
| Benefits hallucination | #4 | Code | `_fix_benefits_hallucination` with text-presence check |
| Salary hourly rate x1000 | #4, #6 | Code | `_fix_salary` hourly rate correction |
| Preferred tech from wrong section | #4 | Prompt | Clarified section names |
| Reporting_to hallucination | #10 | Code | `_fix_reporting_to_hallucination` |
| Company/Job ID combined | #11 | Prompt | Separation instruction |
| "Yes/No" form artifacts | #10, #11 | Code | `_fix_clean_list_artifacts` |
| Work mode wrong despite location clue | #14 | Code | `_fix_work_mode` location line override |
| Skills under-extraction | #13 | Prompt | Exhaustive extraction hints |
| Employment type both contract + full_time | #18 | Code | Both-values logic in `_fix_employment_type` |
| String "null" from LLM | #18 | Code | Global string "null" → None normalization |

---

## 11. Known Limitations

### Model Limitations

| Limitation | Details | Mitigation |
|-----------|---------|------------|
| **company_overview extraction** | LLM inconsistently extracts company overview paragraphs | None — depends on JD structure |
| **job_summary vs description** | LLM sometimes uses company description as job_summary | Prompt engineering with section name hints |
| **Seniority inference** | LLM sometimes infers seniority from experience years (e.g., 2 years → "Junior") | Accepted — minor issue |
| **Preferred vs Required mixing** | LLM occasionally moves items between required and preferred sections | Prompt clarification helps but doesn't eliminate |
| **Industry inference** | LLM sometimes fails to infer industry even when context is obvious | Prompt hint to infer from company overview |

### Post-Processing Limitations

| Limitation | Details |
|-----------|---------|
| **Country list** | `_fix_location_consistency` checks against ~25 countries. Uncommon countries may not be swapped correctly. |
| **Soft skills keyword list** | `_fix_soft_skills_hallucination` validates against ~25 known soft skills. Unusual soft skills may be filtered out. |
| **Salary hourly threshold** | Hourly rates exactly between $500-$999 are ambiguous — could be legitimate high rates or multiplication errors. |
| **Benefits exact match** | `_fix_benefits_hallucination` requires exact case-insensitive substring match. Paraphrased benefits are filtered out. |

### Infrastructure Limitations

| Limitation | Details |
|-----------|---------|
| **Free tier TPM** | 6K tokens/min — effectively 1-2 JDs per minute |
| **No persistence** | Parsed results are not stored. Client must save the response. |
| **No authentication** | API is open. Add auth middleware for production. |
| **No .doc support without antiword** | Legacy .doc files require antiword system package. |
| **Render cold start** | Free tier instances spin down after inactivity. First request takes 30-60s. |
| **Python version** | Must use Python 3.11.x — Render's default 3.14 breaks dependencies. |

---

## 12. Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key from [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Model ID to use for parsing |
| `PORT` | No | `8001` | Server port |
| `PYTHON_VERSION` | No | — | Pin Python version for Render deployment (use `3.11.12`) |

### Prompt Configuration

The system prompt and extraction prompt are defined as constants in `groq_jd_parser.py`:
- `SYSTEM_PROMPT` — Sets the LLM's role as an expert JD parser for an ATS
- `EXTRACTION_PROMPT` — Contains 30 field-by-field extraction rules + JSON schema template + critical reminders section

**LLM Parameters:**
```python
temperature: 0.1    # Low randomness for consistent structured output
top_p: 0.9
max_tokens: 8192    # Max response length
timeout: 120s       # Per-request timeout (with up to 5 retries)
```

---

## 13. Project Structure

```
llama-jobparser/
|
+-- app.py                  # Flask API server (endpoints, security headers)
+-- groq_jd_parser.py       # Core logic (text extraction, LLM call, JSON parsing,
|                           #   normalization, post-processing, confidence, provenance)
+-- index.html              # Web UI (drag-drop upload, structured result display)
+-- requirements.txt        # Python dependencies
+-- Dockerfile              # Docker build (Python 3.11 + antiword + tesseract)
+-- render.yaml             # Render deployment config (Python 3.11.12 pinned)
+-- .python-version         # Python version pin file
+-- .env                    # Environment variables (not committed)
+-- .gitignore              # Git ignore rules
+-- DOCUMENTATION.md        # This file
```

### File Responsibilities

| File | Responsibility |
|------|---------------|
| `groq_jd_parser.py` | Text extraction, Groq API integration with retry, JSON extraction, LLM output normalization, 10-fix post-processing pipeline, confidence scoring, provenance tracking |
| `app.py` | HTTP endpoints (/parse, /parse/text, /parse/bulk, /health), file handling, security headers, CORS |
| `index.html` | Web UI with drag-drop, structured result rendering, JSON toggle |

---

## 14. Changelog

### v1.0.0 (2026-03-02)

**Prompt Engineering (12 improvements)**
- Location: country vs region clarification with examples
- Employment type: duration/C2C/W2 → contract detection instructions
- Salary: plain number preservation rules ($70/hr stays 70)
- Skills: exhaustive extraction with or/slash splitting, parenthetical tech
- Soft skills: anti-hallucination — only extract if exact word appears in text
- Benefits: anti-hallucination — only extract explicitly named benefits
- Reporting to: require explicit label ("Reports to", "Reporting To")
- Requirements vs preferred: strict section separation
- Title domain: expanded with cloud, networking, devops, support, operations + examples
- Company/Job ID: separation when combined in "Client:" field
- Preferred technologies: recognize "Familiarity with", "Valuable skills" sections
- Job summary/description: expanded section name recognition

**Post-Processing Pipeline (10 fixes)**
- `_fix_employment_type()` — contract detection from text signals + normalization + dedup + both contract+full_time support
- `_fix_work_mode()` — normalization + location line pattern override
- `_fix_salary()` — string → number conversion + hourly rate >500 correction
- `_fix_skill_dedup()` — case-insensitive skill deduplication
- `_fix_location_consistency()` — country/region swap detection + work_mode sync
- `_fix_derive_skill_splits()` — derive technical_skills/soft_skills from skills array
- `_fix_soft_skills_hallucination()` — two-layer validation (text presence + keyword list)
- `_fix_benefits_hallucination()` — text presence validation
- `_fix_clean_list_artifacts()` — strip form prefixes ("Yes/No")
- `_fix_reporting_to_hallucination()` — explicit label validation

**Normalization Layer**
- String "null"/"none"/"n/a" → actual None for all fields
- LLM output type fixes (company as string, skills schema, location nulls)

**Infrastructure**
- Groq API retry logic (5 attempts with backoff from 429 responses)
- Confidence scoring with per-field base scores
- Provenance tracking with character span matching
- Python 3.11.12 pinned for Render compatibility
- Render deployment with `render.yaml`
- Docker support with antiword + tesseract

**Testing**
- Tested against 18 unique JDs across diverse formats and domains
- All 18 JDs pass with correct extraction (fields correctly null when data absent)
