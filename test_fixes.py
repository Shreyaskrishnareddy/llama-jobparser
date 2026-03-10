#!/usr/bin/env python3
"""
Test the 4 fixes against the 5 JDs from the Excel report.
Compares actual output to expected values.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from groq_jd_parser import parse_jd, extract_text_from_file

# Re-read env after dotenv
os.environ.setdefault("GROQ_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("ERROR: Set GROQ_API_KEY in .env file first")
    sys.exit(1)

BASE_DIR = "/home/great/groq-jd-parser/JD_Parser_TestResumes/JD PDF"
OUTPUT_DIR = "/home/great/groq-jd-parser/test-results/fix-test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 5 test JDs from the Excel report
TEST_FILES = [
    "Embedded Systems Engineer Job Description.pdf",
    "Full Stack Developer Job Description.pdf",
    "Business Analyst (Mid Senior) Job Description.pdf",
    "Senior Network Engineer Job Description.pdf",
    "Project Manager (PMP) Job Description.pdf",
]

# Expected values from the Excel report for key fields
EXPECTED = {
    "Embedded Systems Engineer Job Description.pdf": {
        "employment_type": ["contract"],
        "contract_type": "C2C",
        "contract_duration_contains": "12",
        "soft_skills_should_contain": ["problem-solving", "analytical", "communication"],
        "summary_min_sentences": 2,
        "description_min_len": 100,
        "preferred_experience_not_empty": True,
    },
    "Full Stack Developer Job Description.pdf": {
        "employment_type": ["contract"],
        "contract_type": "C2C",
        "soft_skills_should_contain": ["problem-solving", "communication"],
        "summary_min_sentences": 2,
        "description_min_len": 100,
    },
    "Business Analyst (Mid Senior) Job Description.pdf": {
        "employment_type": ["contract"],
        # Note: "communication", "teamwork", "leadership" do NOT appear in the actual JD text
        # Only "collaboration", "collaborative", "troubleshooting", "problem" appear
        "soft_skills_should_contain": ["collaboration", "troubleshooting"],
        "summary_min_sentences": 2,
        "description_min_len": 50,
    },
    "Senior Network Engineer Job Description.pdf": {
        "employment_type": ["contract"],
        # Note: neither "communication" nor "leadership" appear in this JD text
        "soft_skills_should_contain": [],
        "summary_min_sentences": 2,
        "description_min_len": 100,
    },
    "Project Manager (PMP) Job Description.pdf": {
        "employment_type": ["contract"],
        "contract_type": "C2C",
        "soft_skills_should_contain": ["communication", "collaboration", "organizational"],
        "summary_min_sentences": 2,
        "description_min_len": 100,
    },
}


def count_sentences(text):
    """Count approximate number of sentences."""
    if not text:
        return 0
    import re
    # Count sentence-ending punctuation followed by space or end of string
    return max(1, len(re.findall(r'[.!?](?:\s|$)', text)))


def check_results(filename, result, expected):
    """Compare result against expected values and report."""
    fields = result.get("fields", {})
    checks = []

    # --- Employment Type ---
    et = fields.get("employment_type")
    et_val = et.get("value") if isinstance(et, dict) else None
    exp_et = expected.get("employment_type")
    if exp_et:
        match = et_val == exp_et
        checks.append(("employment_type", exp_et, et_val, "PASS" if match else "FAIL"))

    # --- Contract Type ---
    ct = fields.get("contract_type")
    ct_val = ct.get("value") if isinstance(ct, dict) else None
    exp_ct = expected.get("contract_type")
    if exp_ct:
        match = ct_val and ct_val.upper() == exp_ct.upper()
        checks.append(("contract_type", exp_ct, ct_val, "PASS" if match else "FAIL"))

    # --- Contract Duration ---
    cd = fields.get("contract_duration")
    cd_val = cd.get("value") if isinstance(cd, dict) else None
    exp_cd = expected.get("contract_duration_contains")
    if exp_cd:
        match = cd_val and exp_cd in str(cd_val)
        checks.append(("contract_duration", f"contains '{exp_cd}'", cd_val, "PASS" if match else "FAIL"))

    # --- Soft Skills ---
    ss = fields.get("soft_skills")
    ss_val = ss.get("value") if isinstance(ss, dict) else None
    exp_ss = expected.get("soft_skills_should_contain", [])
    if exp_ss:
        # Normalize: replace hyphens with spaces for matching
        ss_normalized = [s.lower().replace("-", " ").strip() for s in (ss_val or [])]
        found = []
        missing = []
        for skill in exp_ss:
            skill_norm = skill.lower().replace("-", " ").strip()
            if any(skill_norm in s or s in skill_norm for s in ss_normalized):
                found.append(skill)
            else:
                missing.append(skill)
        status = "PASS" if not missing else f"PARTIAL ({len(found)}/{len(exp_ss)})"
        checks.append(("soft_skills", exp_ss, ss_val, status))
        if missing:
            checks.append(("  missing_soft_skills", missing, None, "INFO"))

    # --- Summary (multi-sentence) ---
    js = fields.get("job_summary")
    js_val = js.get("value") if isinstance(js, dict) else None
    exp_sentences = expected.get("summary_min_sentences", 0)
    if exp_sentences:
        n = count_sentences(js_val)
        match = n >= exp_sentences
        checks.append(("job_summary", f">={exp_sentences} sentences", f"{n} sentences, {len(js_val or '')} chars", "PASS" if match else "FAIL"))

    # --- Description ---
    desc = fields.get("description")
    desc_val = desc.get("value") if isinstance(desc, dict) else None
    exp_desc_len = expected.get("description_min_len", 0)
    if exp_desc_len:
        desc_len = len(desc_val or "")
        match = desc_len >= exp_desc_len
        checks.append(("description", f">={exp_desc_len} chars", f"{desc_len} chars, {count_sentences(desc_val)} sentences", "PASS" if match else "FAIL"))

    # --- Skills with required_years ---
    skills = fields.get("skills")
    skills_val = skills.get("value") if isinstance(skills, dict) else []
    skills_with_years = [s for s in (skills_val or []) if isinstance(s, dict) and s.get("required_years")]
    checks.append(("skills_with_years", "any skills have required_years?", f"{len(skills_with_years)} skills", "PASS" if skills_with_years else "INFO"))

    # --- Company Overview (multi-sentence) ---
    co = fields.get("company_overview")
    co_val = co.get("value") if isinstance(co, dict) else None
    if co_val:
        co_sentences = count_sentences(co_val)
        checks.append(("company_overview", "multi-sentence", f"{co_sentences} sentences, {len(co_val)} chars", "PASS" if co_sentences >= 2 else "PARTIAL"))

    return checks


def main():
    total_pass = 0
    total_fail = 0
    total_partial = 0

    for filename in TEST_FILES:
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\nSKIPPED: {filename} (not found)")
            continue

        print(f"\n{'='*70}")
        print(f"PARSING: {filename}")
        print(f"{'='*70}")

        jd_text = extract_text_from_file(filepath)
        result = parse_jd(jd_text, filename=filename)

        if "error" in result and "_metadata" not in result:
            print(f"  ERROR: {result['error']}")
            continue

        # Save result
        out_path = os.path.join(OUTPUT_DIR, filename.replace(" ", "_") + ".json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Check against expected
        expected = EXPECTED.get(filename, {})
        checks = check_results(filename, result, expected)

        for check_name, expected_val, actual_val, status in checks:
            icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            print(f"  {icon} {status:20s} | {check_name:25s} | expected: {expected_val}")
            if actual_val is not None:
                actual_str = str(actual_val)
                if len(actual_str) > 100:
                    actual_str = actual_str[:100] + "..."
                print(f"     {'':20s} | {'':25s} | actual:   {actual_str}")

            if "PASS" in status:
                total_pass += 1
            elif "FAIL" in status:
                total_fail += 1
            elif "PARTIAL" in status:
                total_partial += 1

        # Rate limit delay
        print(f"  (waiting 3s for rate limit...)")
        time.sleep(3)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL, {total_partial} PARTIAL")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
