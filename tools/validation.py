import os
import json
from google.adk.tools import ToolContext

from llm.gemini_client import get_client, generate_json
from llm.schemas import ADDENDUM_VALIDATION_SCHEMA
from state.session_state import (
    ADDENDUM, EXTRACTED_RESUME, GAP_ANALYSIS,
    VALIDATION, VALIDATION_PASS, VALIDATION_FLAGS,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Keywords that indicate the validator is describing an ACCEPTABLE bullet, not a real violation
ACCEPTANCE_PHRASES = [
    "acceptable",
    "this is acceptable",
    "acceptable since",
    "is acceptable",
    "not a violation",
    "no violation",
    "valid for",
    "plausible for",
    "role end date is present",
    "end date is present",
    "end_year >= 2024",
    "currently active",
    "present role",
]

SYSTEM = """You are a strict ATS Compliance Auditor.

Validate the ADDENDUM bullets.

Checks:
1) Bullets start with strong action verbs and are senior-level.
2) Bullets are specific and avoid generic fluff.
3) Keyword coverage is reasonable.
4) Timeframe compliance (MANDATORY):
   - For EVERY bullet, verify the tools/skills/methods mentioned are plausible
     for that role's timeframe.
   - Modern GenAI-era items (LLM, RAG, agents, vector DB, embeddings, LangChain/LlamaIndex,
     OpenAI/Gemini/Vertex AI, Azure OpenAI, Azure AI Search, Azure AI Foundry, Azure MCP,
     Pinecone/Weaviate/Milvus/FAISS/ChromaDB)
     must NOT appear under any role ending before 2024.
   - Roles with end_date = "Present" or end_year >= 2024 are FULLY ALLOWED to have
     any modern GenAI/Azure AI bullets. Do NOT flag these as violations under any circumstance.

STRICT RULES for timeframe_violations[]:
- ONLY add an entry to timeframe_violations[] if a modern GenAI item appears under a role
  that ENDED before 2024 (e.g., a role ending in 2022, 2021, 2020 etc.).
- If a bullet is under a currently active role (end_date = "Present") or end_year >= 2024,
  it must NOT appear in timeframe_violations[] — not even with a note saying "acceptable".
- timeframe_violations[] must contain ONLY actual violations.
  An "acceptable" item is NOT a violation and must be completely omitted from this list.

STRICT RULES for passed:
- Set passed=true if timeframe_violations=[] (regardless of issues[]).
- Set passed=false ONLY if timeframe_violations[] has at least one REAL violation
  (a modern item placed under a pre-2024 role).
- issues[] contains advisory quality feedback only — it never affects passed.

Return JSON exactly matching schema.
"""


def _is_false_violation(violation_text: str) -> bool:
    """Returns True if the violation text is actually describing an acceptable bullet."""
    lower = violation_text.lower()
    return any(phrase in lower for phrase in ACCEPTANCE_PHRASES)


def validate_bullets(tool_context: ToolContext) -> dict:
    addendum = tool_context.state.get(ADDENDUM)
    extracted_resume = tool_context.state.get(EXTRACTED_RESUME)
    gap_analysis = tool_context.state.get(GAP_ANALYSIS)

    if not addendum:
        return {"success": False, "error": "Missing addendum"}

    user = f"""
Addendum JSON:
{json.dumps(addendum, indent=2)}

Missing gaps:
{json.dumps((gap_analysis or {}).get("missing", {}), indent=2)}

Original timeline and original bullets:
{json.dumps((extracted_resume or {}).get("experience", []), indent=2)}

REMEMBER:
- Bullets under roles with end_date="Present" are ALWAYS timeframe-valid.
- Do NOT add any bullet to timeframe_violations[] if the role is currently active.
- timeframe_violations[] must be empty [] if all modern GenAI bullets are under Present roles.

Return validation results.
"""

    client = get_client()
    out = generate_json(
        client=client,
        model=MODEL,
        system=SYSTEM,
        user=user,
        response_schema=ADDENDUM_VALIDATION_SCHEMA,
        temperature=0.1,
    )

    if isinstance(out, dict) and out.get("error"):
        return {"success": False, "error": out["error"]}

    # --- Programmatic post-processing ---

    # 1) Strip false violations — entries where the LLM itself says "acceptable"
    raw_violations = out.get("timeframe_violations") or []
    real_violations = [v for v in raw_violations if not _is_false_violation(v)]
    out["timeframe_violations"] = real_violations

    # 2) Force passed=True if no real timeframe violations remain
    issues = out.get("issues") or []
    if not real_violations:
        out["passed"] = True

    # 3) Write to state
    tool_context.state[VALIDATION] = out
    tool_context.state[VALIDATION_PASS] = bool(out.get("passed", False))
    tool_context.state[VALIDATION_FLAGS] = real_violations + issues

    return {
        "success": True,
        "passed": out.get("passed", False),
        "grade": out.get("ats_grade", "?"),
    }

