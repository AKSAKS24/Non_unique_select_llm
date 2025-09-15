import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import requests
import json

# --- LLM ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required.")

app = FastAPI(title="SELECT SINGLE Not Unique Remediator - LLM prompt system (GPT-4.1-nano style)")

# --- Data Models ---
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""
    findings: Optional[List[Finding]] = Field(default_factory=list)

# --- LLM SYSTEM PROMPT ---
SYSTEM_MSG = """
You are a senior ABAP and SAP expert. Always output JSON ONLY as your response.

INSTRUCTIONS:
- For EVERY item in the supplied findings list that contains BOTH a snippet (OLD code) and a suggestion (REMEDIATED code):
  - Output a distinct bullet, in the "llm_prompt" field, for EVERY finding (never summarize across more than one finding).
  - For each bullet:
     - Use the finding's `message` as a one-line summary, if present.
     - THEN output the OLD code under a fenced code block titled "Old code" (with language abap).
     - THEN output the REMEDIATED code under a fenced code block titled "Remediated code" (with language abap).
  - Omit any finding that does not provide both snippet and suggestion.

- For N findings, the output must have N separate bullets, each containing both OLD and REMEDIATED code. Do not combine or summarize findings.
- Return JSON ONLY:
{{
  "assessment": "Brief summary of risks for SELECT SINGLE not unique. State number of actionable findings.",
  "llm_prompt": "<Instructions and all before/after code. Only refer to supplied code snippets.>"
}}
"""

# --- USER PROMPT TEMPLATE, BRACES ESCAPED ---
USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

findings (JSON list of findings, each with all fields above):
{findings_json}

Instructions:
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

findings (JSON list of findings, each with all fields above):
{findings_json}

Instructions:
1. Write a brief assessment summarizing SELECT SINGLE not-unique risks.
2. For the "llm_prompt" field:
   - Output one bullet per finding, for EVERY finding with both a suggestion and a snippet.
   - For EACH:
     - Start with a summary line, ideally using the finding's "message".
     - Output the old code using: 
       Old code:
       ```abap
       <snippet from finding>
       ```
     - Output the remediated code using:
       Remediated code:
       ```abap
       <suggestion from finding>
       ```
   - DO NOT summarize across multiple findings or combine them. No general "multiple select" summaries allowed.
Return strictly JSON:
{{
  "assessment": "<concise risk summary>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}
""".strip()

# --- Prompt builder using string formatting (with escaped braces) ---
def build_prompt(unit: Unit, relevant_findings: List[Finding]) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in relevant_findings], ensure_ascii=False, indent=2)
    prompt_content = USER_TEMPLATE.format(
        pgm_name=unit.pgm_name,
        inc_name=unit.inc_name,
        unit_type=unit.type,
        unit_name=unit.name or "",
        class_implementation=unit.class_implementation or "",
        start_line=unit.start_line or 0,
        end_line=unit.end_line or 0,
        findings_json=findings_json,
    )
    return {
        "system": SYSTEM_MSG.strip(),
        "user": prompt_content.strip()
    }

# --- LLM API Integration (OpenAI/azure style, synchronous) ---
def call_llm(system_msg: str, user_prompt: str) -> Dict[str, Any]:
    url = f"{OPENAI_API_BASE}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    try:
        resp.raise_for_status()
        # Should be a valid JSON response or have a JSON "content" we can parse
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        # Expecting a JSON dict as string in the "content"
        return json.loads(content)
    except Exception as e:
        # Fallback: propagate the error as 'assessment' and empty prompt
        return {
            "assessment": f"[LLM error: {e}]",
            "llm_prompt": ""
        }

# --- Main logic ---
def llm_assess_and_prompt_llm(unit: Unit) -> Dict[str, str]:
    # Only findings with both .snippet and .suggestion
    relevant_findings = [
        f for f in (unit.findings or [])
        if f.suggestion and f.suggestion.strip() and f.snippet and f.snippet.strip()
    ]
    if not relevant_findings:
        return None
    prompt_obj = build_prompt(unit, relevant_findings)
    llm_result = call_llm(prompt_obj["system"], prompt_obj["user"])
    return llm_result

@app.post("/assess-select-single")
async def assess_select_single(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        llm_out = llm_assess_and_prompt_llm(u)
        if not llm_out:
            continue
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name,
            "class_implementation": u.class_implementation,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "assessment": llm_out.get("assessment", ""),
            "llm_prompt": prompt_out
        }
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)