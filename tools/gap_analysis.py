import os
import re
import logging
from google.adk.tools import ToolContext

from llm.gemini_client import get_client, generate_json
from llm.schemas import GAP_ANALYSIS_SCHEMA
from state.session_state import (
    RESUME_EXTRACTED,
    JD_EXTRACTED,
    GAP_ANALYSIS,
    JD_RAW,
    EXTRACTED_RESUME,
)

logger = logging.getLogger(__name__)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

CLOUD_SERVICE_CATALOG = [
    ("Azure OpenAI Service",        ["azure openai"]),
    ("Azure AI Foundry",            ["azure ai foundry", "ai foundry"]),
    ("Azure AI Search",             ["azure ai search", "azure cognitive search"]),
    ("Azure Document Intelligence", ["azure document intelligence", "form recognizer"]),
    ("Azure MCP",                   ["azure mcp"]),
    ("Azure AutoGen",               ["azure autogen"]),
    ("Azure Machine Learning",      ["azure machine learning", "azure ml"]),
    ("Azure Databricks",            ["azure databricks"]),
    ("Azure Data Factory",          ["azure data factory"]),
    ("Azure Kubernetes Service",    ["azure kubernetes", "aks"]),
    ("Azure Functions",             ["azure functions"]),
    ("Azure DevOps",                ["azure devops"]),
    ("Azure Data Lake",             ["azure data lake", "adls"]),
    ("Azure Cosmos DB",             ["azure cosmos", "cosmos db"]),
    ("Amazon Bedrock",              ["amazon bedrock", "aws bedrock"]),
    ("Amazon SageMaker",            ["amazon sagemaker", "aws sagemaker"]),
    ("Amazon OpenSearch Service",   ["amazon opensearch", "aws opensearch"]),
    ("AWS Textract",                ["aws textract", "amazon textract"]),
    ("AWS Lambda",                  ["aws lambda", "amazon lambda"]),
    ("AWS Glue",                    ["aws glue"]),
    ("Amazon Redshift",             ["amazon redshift", "aws redshift"]),
    ("Amazon EKS",                  ["amazon eks", "aws eks"]),
    ("Google Vertex AI",            ["vertex ai", "google vertex"]),
    ("Google Document AI",          ["google document ai", "document ai"]),
    ("BigQuery",                    ["bigquery"]),
]

GAP_SYSTEM_PROMPT = """You are an expert ATS Analyzer.
Identify missing NON-cloud-branded skills and tools only.
Cloud service gaps are already computed separately and merged.

INSTRUCTIONS:
1) Use semantic/equivalent matching for non-cloud items only.
2) Only mark MISSING if no evidence exists in the resume.
3) Return match_score 0-100 based on overall JD coverage.
4) Return ONLY JSON matching the schema.
"""


def _scan_cloud_services_in_text(text: str) -> list[str]:
    text_lower = (text or "").lower()
    found: list[str] = []
    for canonical_name, patterns in CLOUD_SERVICE_CATALOG:
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
                found.append(canonical_name)
                break
    return found


def _get_all_resume_bullets(extracted_resume: dict) -> str:
    parts: list[str] = []
    for role in extracted_resume.get("experience", []):
        for bullet in role.get("bullets", []):
            parts.append(bullet)
    return " ".join(parts).lower()


def analyze_gaps(tool_context: ToolContext) -> dict:
    resume_extracted = tool_context.state.get(RESUME_EXTRACTED) or {}
    jd_extracted = tool_context.state.get(JD_EXTRACTED) or {}
    jd_raw = tool_context.state.get(JD_RAW, "") or ""
    extracted_resume = tool_context.state.get(EXTRACTED_RESUME) or {}

    logger.info("=== GAP ANALYZER START ===")
    logger.info(f"jd_raw length: {len(jd_raw)}")
    logger.info(f"jd_raw preview: {jd_raw[:300]}")
    logger.info(f"resume_extracted keys: {list(resume_extracted.keys())}")
    logger.info(
        f"resume_extracted skills count: {len(resume_extracted.get('skills', []))}"
    )
    logger.info(
        f"resume_extracted tools count: {len(resume_extracted.get('tools', []))}"
    )
    logger.info(f"resume_extracted tools: {resume_extracted.get('tools', [])}")
    logger.info(
        f"extracted_resume roles count: {len(extracted_resume.get('experience', []))}"
    )
    logger.info(f"jd_extracted tools: {jd_extracted.get('tools', [])}")
    logger.info(f"jd_extracted skills: {jd_extracted.get('skills', [])}")

    if not resume_extracted:
        msg = "MISSING resume_extracted in state — extraction stage likely failed"
        logger.error(msg)
        return {"success": False, "error": msg}

    if not jd_extracted:
        msg = "MISSING jd_extracted in state — extraction stage likely failed"
        logger.error(msg)
        return {"success": False, "error": msg}

    # Step 1: programmatic cloud detection from raw JD text
    jd_cloud_services = _scan_cloud_services_in_text(jd_raw)
    logger.info(f"Cloud services detected in JD raw text: {jd_cloud_services}")

    resume_bullets_text = _get_all_resume_bullets(extracted_resume)
    logger.info(f"Resume bullets text length: {len(resume_bullets_text)}")
    logger.info(f"Resume bullets preview: {resume_bullets_text[:300]}")

    resume_cloud_services = _scan_cloud_services_in_text(resume_bullets_text)
    logger.info(f"Cloud services detected in resume bullets: {resume_cloud_services}")

    resume_cloud_lower = [s.lower() for s in resume_cloud_services]
    forced_missing_tools: list[str] = []
    for service in jd_cloud_services:
        if service.lower() not in resume_cloud_lower:
            forced_missing_tools.append(service)
            logger.info(f"FORCED MISSING: {service}")
        else:
            logger.info(f"COVERED: {service}")

    logger.info(f"Total forced missing cloud tools: {forced_missing_tools}")

    # Step 2: LLM for non-cloud items only
    resume_tools = resume_extracted.get("tools", [])
    resume_skills = resume_extracted.get("skills", [])
    jd_tools = jd_extracted.get("tools", [])
    jd_skills = jd_extracted.get("skills", [])

    non_cloud_jd_tools = [
        t
        for t in jd_tools
        if not any(
            c in t.lower()
            for c in [
                "azure",
                "aws",
                "amazon",
                "google cloud",
                "vertex ai",
                "bigquery",
                "gcp",
            ]
        )
    ]
    logger.info(f"Non-cloud JD tools sent to LLM: {non_cloud_jd_tools}")
    logger.info(f"JD skills sent to LLM: {jd_skills}")

    user_prompt = (
        "Resume Profile:\n"
        f"- Skills: {resume_skills}\n"
        f"- Tools: {resume_tools}\n\n"
        "Job Description Profile (non-cloud items only):\n"
        f"- Skills: {jd_skills}\n"
        f"- Tools: {non_cloud_jd_tools}\n\n"
        "Identify strictly MISSING non-cloud skills/tools using semantic matching.\n"
        "Return ONLY JSON."
    )

    client = get_client()
    out = generate_json(
        client=client,
        model=MODEL,
        system=GAP_SYSTEM_PROMPT,
        user=user_prompt,
        response_schema=GAP_ANALYSIS_SCHEMA,
        temperature=0.1,
    )

    logger.info(f"GAP LLM raw output: {out!r}")

    if not isinstance(out, dict):
        msg = f"LLM gap analysis returned non-dict: {out!r}"
        logger.error(msg)
        raise RuntimeError(msg)

    if out.get("error"):
        msg = f"LLM gap analysis error: {out['error']}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Validate required schema keys before mutating
    for key in ("match_score", "missing", "matched"):
        if key not in out:
            msg = f"GAP_ANALYSIS_SCHEMA missing key '{key}' in LLM output: {out}"
            logger.error(msg)
            return {"success": False, "error": msg}

    if not isinstance(out["missing"], dict):
        msg = f"'missing' is not an object in LLM output: {out}"
        logger.error(msg)
        return {"success": False, "error": msg}

    if "tools" not in out["missing"] or "skills" not in out["missing"]:
        msg = f"'missing' object incomplete in LLM output: {out}"
        logger.error(msg)
        return {"success": False, "error": msg}

    llm_missing_tools = out["missing"].get("tools", [])
    llm_missing_skills = out["missing"].get("skills", [])
    logger.info(f"LLM missing tools: {llm_missing_tools}")
    logger.info(f"LLM missing skills: {llm_missing_skills}")

    # Step 3: merge programmatic + LLM results
    seen: set[str] = set()
    merged_missing_tools: list[str] = []
    for t in forced_missing_tools + llm_missing_tools:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key)
            merged_missing_tools.append(t)

    out["missing"]["tools"] = merged_missing_tools
    out["missing"]["skills"] = llm_missing_skills

    logger.info(f"FINAL merged missing tools: {merged_missing_tools}")
    logger.info(f"FINAL merged missing skills: {llm_missing_skills}")
    logger.info("=== GAP ANALYZER END ===")

    tool_context.state[GAP_ANALYSIS] = out
    return out

    

