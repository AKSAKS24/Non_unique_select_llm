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

app = FastAPI(title="ABAP Remediator")

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

def json_escape_string_for_llm(s: str) -> str:
    """ Escape all backslash, quotes, newlines for safe LLM-JSON embedding """
    if not s:
        return ""
    s = s.replace("\\", "\\\\")
    s = s.replace("\"", "\\\"")
    s = s.replace("\n", "\\n")
    return s

SYSTEM_MSG = """
You are a senior ABAP and SAP expert. You ALWAYS output a single flat JSON object with exactly these two fields: "assessment" and "llm_prompt".

Instructions:
- "assessment": a brief summary, in plain English, of all the types of code transformations/remediations from the findings. Do not include any ABAP code here; just describe what you changed and why (e.g. "Converted 3 non-optimized SELECT statements to safe SELECT SINGLE patterns.").
- "llm_prompt": a single Markdown block, as a JSON-escaped string, listing for EACH finding:
  - the [finding message]
  - Old code:
    ```abap
    [snippet]
    ```
  - Remediated code:
    ```abap
    [suggestion]
    ```
- Every finding that has both suggestion and snippet must be included.
- All output must be JSON-escaped:
    - No literal newlines, use \\n
    - No raw quotes, use \\"
    - Backslash as \\\\
- 'llm_prompt' is a flat string, not a JSON array.
- Sample output:
{{
  "assessment": "Transformed all SELECT statements to SELECT SINGLE according to best practices.",
  "llm_prompt": "- [Finding A description]\\\\nOld code:\\\\n```abap\\\\n...\\\\n```\\\\nRemediated code:\\\\n```abap\\\\n...\\\\n```\\\\n\\\\n- ..."
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
- For EACH finding with both non-empty snippet and suggestion:
    - Render a section in 'llm_prompt' as described above.
    - Separate findings/blocks with two escaped newlines (\\\\n\\\\n).
    - Snippet and suggestion must be included and properly JSON-escaped (\\n for newlines, \\" for quotes, \\\\ for backslash).
- Output only a single flat object with two fields:
    - "assessment" (summary, plain English, no code)
    - "llm_prompt" (full Markdown listing, as a JSON-escaped string)
- Your output must look like:
{{
  "assessment": "summary...",
  "llm_prompt": "- ..." 
}}
"""

def build_prompt(unit: Unit, relevant_findings: List[Finding]) -> Dict[str, str]:
    findings_dicts = []
    for f in relevant_findings:
        fd = f.model_dump()
        # Escape for JSON embedding in LLM instruction (avoid any broken LLM JSON output due to code content)
        for k in ["snippet", "suggestion", "message"]:
            if fd.get(k):
                fd[k] = json_escape_string_for_llm(fd[k])
        findings_dicts.append(fd)
    findings_json = json.dumps(findings_dicts, ensure_ascii=False, indent=2)
    prompt_content = USER_TEMPLATE.format(
        pgm_name=json_escape_string_for_llm(unit.pgm_name),
        inc_name=json_escape_string_for_llm(unit.inc_name),
        unit_type=json_escape_string_for_llm(unit.type),
        unit_name=json_escape_string_for_llm(unit.name or ""),
        class_implementation=json_escape_string_for_llm(unit.class_implementation or ""),
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
        "max_tokens": 2048,
        "response_format": {"type": "json_object"}
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        result = json.loads(content)
        # Defensive: Ensure both fields are present
        if "assessment" not in result:
            result["assessment"] = "[LLM did not provide assessment summary]"
        if "llm_prompt" not in result:
            result["llm_prompt"] = "[LLM did not provide llm_prompt]"
        return result
    except Exception as e:
        content = locals().get('content', None)
        return {
            "assessment": f"[LLM error: {e}]",
            "llm_prompt": f"[{content}]"
        }

def llm_assess_and_prompt_llm(unit: Unit) -> Optional[Dict[str, str]]:
    relevant_findings = [
        f for f in (unit.findings or [])
        if f.suggestion and f.suggestion.strip() and f.snippet and f.snippet.strip()
    ]
    if not relevant_findings:
        return None
    prompt_obj = build_prompt(unit, relevant_findings)
    llm_result = call_llm(prompt_obj["system"], prompt_obj["user"])
    # Ensure both assessment and llm_prompt present for every result
    return {
        "assessment": llm_result.get("assessment", ""),
        "llm_prompt": llm_result.get("llm_prompt", "")
    }

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