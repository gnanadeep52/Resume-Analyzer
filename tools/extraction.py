import os
from typing import List

from google.adk.tools import ToolContext

from llm.gemini_client import get_client, generate_json
from llm.schemas import EXTRACT_SCHEMA, RESUME_STRUCTURE_SCHEMA
from state.session_state import (
    RESUME_RAW, JD_RAW,
    EXTRACTED_RESUME, RESUME_EXTRACTED, JD_EXTRACTED,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def normalize_list(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items or []:
        x2 = " ".join(str(x).strip().split())
        if not x2:
            continue
        key = x2.lower()
        if key not in seen:
            seen.add(key)
            out.append(x2)
    return out


STRUCTURE_SYSTEM = """You are a precise Resume Parser.

Extract ONLY:
- candidate_name
- summary (if present; else empty string)
- experience[] with title, company, location, start_date, end_date, bullets

CRITICAL RULES:
1) experience[].bullets must be copied VERBATIM.
2) Preserve dates exactly as written.
3) Do NOT hallucinate missing fields; use empty string.
4) Output MUST be valid JSON matching the schema exactly.
"""

EXTRACT_SYSTEM_RESUME = """You are a strict Resume Extractor.

CRITICAL RULES:
1. Source of Truth: Extract skills and tools ONLY from "Professional Experience",
   "Work History", or "Projects" sections.
2. IGNORE: Do NOT extract anything from "Technical Skills", "Skills",
   "Core Competencies", or "Professional Summary" lists.
3. Return Format:
   - Return ONLY JSON with two lists:
     1) skills (concepts, techniques, methodologies)
     2) tools (technologies, frameworks, cloud services, libraries, products)
   - No domains, verbs, responsibilities, metrics, or explanations.
   - Keep items short. No duplicates.
4. Preserve full cloud service names exactly as used in the bullets.
   Example: "Amazon Bedrock" not "Bedrock", "Azure Machine Learning" not "Azure ML".
5. If uncertain, omit the item (do NOT guess).

Return ONLY JSON matching schema.
"""

EXTRACT_SYSTEM_JD = """You are a Job Description Extractor.

CRITICAL RULES:
1. Source of Truth: Extract ALL technical skills, tools, platforms, and cloud services
   required or mentioned in the JD text.

2. Cloud Service Name Preservation (MANDATORY):
   - Preserve the FULL and EXACT name of every cloud service as written in the JD.
   - Do NOT shorten, generalize, or normalize cloud service names. Examples:
     * "Azure OpenAI Service" → extract as "Azure OpenAI Service" NOT "OpenAI" or "Azure ML"
     * "Azure AI Foundry" → extract as "Azure AI Foundry" NOT "Azure ML" or "Azure AI"
     * "Azure AI Search" → extract as "Azure AI Search" NOT "Search" or "Azure Search"
     * "Azure Document Intelligence" → extract as "Azure Document Intelligence" NOT "OCR" or "Textract"
     * "Azure MCP" → extract as "Azure MCP" NOT "MCP" or "agent communication"
     * "AWS Bedrock" → extract as "AWS Bedrock" NOT "Bedrock" or "AWS AI"
     * "Google Vertex AI" → extract as "Google Vertex AI" NOT "Vertex" or "GCP AI"
   - If the JD uses a branded product name, always include the cloud provider prefix.

3. Extraction Logic:
   - Prioritize specific tools over generic categories.
   - Exclude pure task verbs: "documentation", "meetings", "collaboration".
   - Consolidate language variants of the SAME tool (e.g., "writing Python" + "debugging Python" → "Python").
   - Do NOT consolidate DIFFERENT tools even if similar (e.g., "Azure OpenAI" ≠ "OpenAI API").

4. Return Format:
   - Return ONLY JSON with two lists: skills and tools.
   - Keep items concise but NEVER truncate cloud-branded service names.

Return ONLY JSON matching schema.
"""


def extract_full_resume_structure(tool_context: ToolContext) -> dict:
    resume_text = tool_context.state.get(RESUME_RAW, "")
    if not resume_text or not resume_text.strip():
        return {"success": False, "error": "Missing or empty resume_raw in state"}

    client = get_client()
    out = generate_json(
        client=client,
        model=MODEL,
        system=STRUCTURE_SYSTEM,
        user=f"Resume text:\n\n{resume_text}",
        response_schema=RESUME_STRUCTURE_SCHEMA,
        temperature=0.0,
    )

    if isinstance(out, dict) and out.get("error"):
        return {"success": False, "error": out["error"]}

    tool_context.state[EXTRACTED_RESUME] = out
    return {"success": True, "roles": len(out.get("experience", []))}


def extract_resume_skills(tool_context: ToolContext) -> dict:
    resume_text = tool_context.state.get(RESUME_RAW, "")
    if not resume_text or not resume_text.strip():
        return {"success": False, "error": "Missing or empty resume_raw in state"}

    client = get_client()
    payload = generate_json(
        client=client,
        model=MODEL,
        system=EXTRACT_SYSTEM_RESUME,
        user=f"Extract skills and tools from the following RESUME:\n\n{resume_text}",
        response_schema=EXTRACT_SCHEMA,
        temperature=0.1,
    )

    if isinstance(payload, dict) and payload.get("error"):
        return {"success": False, "error": payload["error"]}

    tool_context.state[RESUME_EXTRACTED] = {
        "skills": normalize_list(payload.get("skills", [])),
        "tools": normalize_list(payload.get("tools", [])),
    }
    return {"success": True}


def extract_jd_skills(tool_context: ToolContext) -> dict:
    jd_text = tool_context.state.get(JD_RAW, "")
    if not jd_text or not jd_text.strip():
        return {"success": False, "error": "Missing or empty jd_raw in state"}

    client = get_client()
    payload = generate_json(
        client=client,
        model=MODEL,
        system=EXTRACT_SYSTEM_JD,
        user=f"Extract skills and tools from the following JD:\n\n{jd_text}",
        response_schema=EXTRACT_SCHEMA,
        temperature=0.1,
    )

    if isinstance(payload, dict) and payload.get("error"):
        return {"success": False, "error": payload["error"]}

    tool_context.state[JD_EXTRACTED] = {
        "skills": normalize_list(payload.get("skills", [])),
        "tools": normalize_list(payload.get("tools", [])),
    }
    return {"success": True}
