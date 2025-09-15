import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import requests
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required.")

app = FastAPI(title="SELECT SINGLE Remediator")

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

# --- Strong LLM Prompt:

SYSTEM_MSG = """

You are a senior ABAP and SAP expert. Output ONLY JSON as your response.

!!! VERY IMPORTANT INSTRUCTION !!!
For every code snippet/suggestion field in your returned JSON:
- All newline characters **inside string values** must appear as two characters '\\'+'n' (like: \\\\n), *not* as literal unescaped newlines.
- Escape all double quotes in string values as \\"
- Never use literal (bare) newlines or triple-backticks or unescaped double quotes inside any JSON string value.
Your sample return should look visually like this (EXACTLY this string escaping pattern):
"""
{
  "assessment": "2 actionable findings found.",
  "llm_prompt": "- First finding message\\\\nOld code:\\\\n```abap\\\\nSELECT ...\\\\n```\\\\nRemediated code:\\\\n```abap\\\\nSELECT ...\\\\n```\\\\n\\\\n- Second finding message\\\\nOld code:\\\\n```abap\\\\nSELECT ...\\\\n```\\\\nRemediated code:\\\\n```abap\\\\nSELECT ...\\\\n```"
}
"""
Always return as:
{{
  "assessment": "...",
  "llm_prompt": "..."
}}
"""

USER_TEMPLATE = """
Unit:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

findings (json):
{findings_json}

Instructions:
- For each actionable finding (both snippet and suggestion present) emit:
  - [message]
  - Old code:
    ```abap
    [snippet]
    ```
  - Remediated code:
    ```abap
    [suggestion]
    ```
- Separate bullets with two newlines.
- llm_prompt must be a valid JSON string (escape newlines as \\n in JSON; do NOT use literal newlines).
- Do NOT combine/mix findings; never leave out any finding with snippet+suggestion.
Return JSON:
Always return as:
{{
  "assessment": "...",
  "llm_prompt": "..."
}}
""".strip()

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
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        return json.loads(content)
    except Exception as e:
        return {
            "assessment": f"[LLM error: {e}]",
            "llm_prompt": ""
        }

def llm_assess_and_prompt_llm(unit: Unit) -> Dict[str, str]:
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
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name,
            "class_implementation": u.class_implementation,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "assessment": llm_out.get("assessment", ""),
            "llm_prompt": llm_out.get("llm_prompt", "")
        }
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)