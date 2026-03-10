#!/usr/bin/env python3
"""
Full comparison: parse all 18 PDF JDs and compare before/after for all 4 fix areas.
"""

import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

from groq_jd_parser import parse_jd, extract_text_from_file

PDF_DIR = "/home/great/groq-jd-parser/JD_Parser_TestResumes/JD PDF"
OLD_DIR = "/home/great/groq-jd-parser/test-results/cycle3"
NEW_DIR = "/home/great/groq-jd-parser/test-results/fix-full"
os.makedirs(NEW_DIR, exist_ok=True)

import re

def count_sentences(text):
    if not text:
        return 0
    return max(1, len(re.findall(r'[.!?](?:\s|$)', text)))


def get_old_result(filename):
    """Load old cycle3 result for comparison."""
    safe = filename.replace(" ", "_").replace("(", "_").replace(")", "_")
    # Try various name patterns
    candidates = [
        safe + ".json",
        safe.replace("__", "_") + ".json",
    ]
    for c in candidates:
        path = os.path.join(OLD_DIR, c)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # Try fuzzy match
    base = os.path.splitext(filename)[0].replace(" ", "_").replace("(", "_").replace(")", "_")
    for f in os.listdir(OLD_DIR):
        if f.endswith(".pdf.json") and base[:20] in f:
            with open(os.path.join(OLD_DIR, f)) as fp:
                return json.load(fp)
    return None


def extract_field(result, field_name):
    """Extract field value from result."""
    if not result:
        return None
    fields = result.get("fields", {})
    f = fields.get(field_name)
    if isinstance(f, dict):
        return f.get("value")
    return None


def main():
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf") and not f.endswith(":Zone.Identifier")])
    print(f"Found {len(pdfs)} PDF files\n")

    improvements = {"summary": 0, "description": 0, "soft_skills": 0, "company_overview": 0,
                    "contract_type": 0, "contract_duration": 0, "skill_years": 0}
    regressions = {"summary": 0, "description": 0, "soft_skills": 0}

    for i, filename in enumerate(pdfs):
        filepath = os.path.join(PDF_DIR, filename)
        short = filename.replace(" Job Description.pdf", "").replace(" Job description.pdf", "")

        print(f"[{i+1}/{len(pdfs)}] {short}...")

        jd_text = extract_text_from_file(filepath)
        new_result = parse_jd(jd_text, filename=filename)

        if "error" in new_result and "_metadata" not in new_result:
            print(f"  ERROR: {new_result['error']}")
            continue

        # Save new result
        safe_name = filename.replace(" ", "_") + ".json"
        with open(os.path.join(NEW_DIR, safe_name), "w") as f:
            json.dump(new_result, f, indent=2, ensure_ascii=False)

        old_result = get_old_result(filename)

        # Compare fields
        old_summary = extract_field(old_result, "job_summary") or ""
        new_summary = extract_field(new_result, "job_summary") or ""
        old_ss = count_sentences(old_summary)
        new_ss = count_sentences(new_summary)

        old_desc = extract_field(old_result, "description") or ""
        new_desc = extract_field(new_result, "description") or ""
        old_ds = count_sentences(old_desc)
        new_ds = count_sentences(new_desc)

        old_soft = extract_field(old_result, "soft_skills") or []
        new_soft = extract_field(new_result, "soft_skills") or []

        old_co = extract_field(old_result, "company_overview") or ""
        new_co = extract_field(new_result, "company_overview") or ""

        new_ct = extract_field(new_result, "contract_type")
        new_cd = extract_field(new_result, "contract_duration")

        new_skills = extract_field(new_result, "skills") or []
        skills_with_years = [s for s in new_skills if isinstance(s, dict) and s.get("required_years")]

        # Print comparison
        summary_delta = f"{old_ss}→{new_ss} sentences ({len(old_summary)}→{len(new_summary)} chars)"
        desc_delta = f"{old_ds}→{new_ds} sentences ({len(old_desc)}→{len(new_desc)} chars)"
        soft_delta = f"{len(old_soft)}→{len(new_soft)} skills"
        co_delta = f"{count_sentences(old_co)}→{count_sentences(new_co)} sentences"

        summary_icon = "✅" if new_ss > old_ss else ("➡️" if new_ss == old_ss else "⚠️")
        desc_icon = "✅" if len(new_desc) > len(old_desc) else ("➡️" if len(new_desc) == len(old_desc) else "⚠️")
        soft_icon = "✅" if len(new_soft) > len(old_soft) else ("➡️" if len(new_soft) == len(old_soft) else "⚠️")

        print(f"  {summary_icon} summary:     {summary_delta}")
        print(f"  {desc_icon} description: {desc_delta}")
        print(f"  {soft_icon} soft_skills: {soft_delta} | old={old_soft} → new={new_soft}")
        print(f"  📋 contract:    type={new_ct}, duration={new_cd}")
        print(f"  🔧 skill_years: {len(skills_with_years)} skills have required_years")
        if skills_with_years:
            for s in skills_with_years[:5]:
                print(f"     - {s['name']}: {s['required_years']} years")

        if new_ss > old_ss:
            improvements["summary"] += 1
        elif new_ss < old_ss:
            regressions["summary"] += 1
        if len(new_desc) > len(old_desc):
            improvements["description"] += 1
        elif len(new_desc) < len(old_desc):
            regressions["description"] += 1
        if len(new_soft) > len(old_soft):
            improvements["soft_skills"] += 1
        elif len(new_soft) < len(old_soft):
            regressions["soft_skills"] += 1
        if new_ct:
            improvements["contract_type"] += 1
        if new_cd:
            improvements["contract_duration"] += 1
        if skills_with_years:
            improvements["skill_years"] += 1

        print()
        time.sleep(3)

    print(f"\n{'='*70}")
    print(f"OVERALL COMPARISON ({len(pdfs)} JDs)")
    print(f"{'='*70}")
    print(f"  Summary improved:      {improvements['summary']}/{len(pdfs)}")
    print(f"  Summary regressed:     {regressions['summary']}/{len(pdfs)}")
    print(f"  Description improved:  {improvements['description']}/{len(pdfs)}")
    print(f"  Description regressed: {regressions['description']}/{len(pdfs)}")
    print(f"  Soft skills improved:  {improvements['soft_skills']}/{len(pdfs)}")
    print(f"  Soft skills regressed: {regressions['soft_skills']}/{len(pdfs)}")
    print(f"  Contract type captured:{improvements['contract_type']}/{len(pdfs)}")
    print(f"  Contract dur captured: {improvements['contract_duration']}/{len(pdfs)}")
    print(f"  Skill years captured:  {improvements['skill_years']}/{len(pdfs)}")


if __name__ == "__main__":
    main()
