import os
import logging
from datetime import datetime
from google.adk.tools import ToolContext

from llm.gemini_client import get_client, generate_json
from llm.schemas import ADDENDUM_SCHEMA
from state.session_state import (
    EXTRACTED_RESUME,
    GAP_ANALYSIS,
    PLACEMENT_MAP,
    USER_INSTRUCTIONS,
    ADDENDUM,
)

logger       = logging.getLogger(__name__)
MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CURRENT_YEAR = datetime.now().year

SUGGESTION_SYSTEM = """You are a Senior Resume Strategist.
Goal: Generate high-impact resume bullets to plug skill gaps.

=== PLACEMENT RULES (ABSOLUTE — NO EXCEPTIONS) ===
1) You will receive a PLACEMENT MAP: {skill: "Title @ Company"}.
2) For EACH missing skill/tool, place bullets ONLY under the exact role specified
   in the placement map. Do NOT move bullets to any other role.
3) Every bullet must start with a strong past-tense action verb.
4) Each bullet must be ONE sentence, max 20 words.
5) Quantify impact wherever possible (%, $, time saved, scale).
6) Do NOT invent job titles, companies, or dates.
7) Output ONLY valid JSON matching the provided schema — no extra text.

=== SECTION C: Cloud Remapping Rule (CRITICAL) ===
When the JD requires a cloud-specific service and the candidate has equivalent
experience on a DIFFERENT cloud or a generic/open-source tool, follow these steps:

1) IDENTIFY the cloud service required by the JD.
2) MAP the candidate's closest equivalent experience using these EXAMPLES as patterns:
   - AWS SageMaker  ↔  Vertex AI / Azure ML / custom MLflow
   - AWS Lambda     ↔  Google Cloud Functions / Azure Functions
   - AWS S3         ↔  GCS / Azure Blob Storage
   - AWS Glue       ↔  Dataflow / Azure Data Factory / dbt
   - AWS Redshift   ↔  BigQuery / Azure Synapse
3) WRITE the bullet from the candidate's ACTUAL experience, then note the
   cloud-mapping in parentheses: e.g., "(equivalent to AWS SageMaker)".
4) Never fabricate usage of a service the candidate did not use.
"""

SUGGESTION_USER_TEMPLATE = """
=== RESUME DATA ===
{extracted_resume}

=== GAP ANALYSIS ===
{gap_analysis}

=== PLACEMENT MAP ===
{placement_map}

=== USER INSTRUCTIONS ===
{user_instructions}

Current Year: {current_year}

Generate resume addendum bullets for ALL skills listed in the placement map.
Return JSON strictly matching the provided schema.
"""

def generate_addendum_points(tool_context: ToolContext) -> dict:
    logger.info("=== ADDENDUM GENERATOR START ===")

    extracted_resume  = tool_context.state.get(EXTRACTED_RESUME, {})
    gap_analysis      = tool_context.state.get(GAP_ANALYSIS, {})
    placement_map     = tool_context.state.get(PLACEMENT_MAP, {})
    user_instructions = tool_context.state.get(USER_INSTRUCTIONS, "")

    logger.info(f"placement_map: {placement_map}")

    if not placement_map:
        logger.warning("Placement map is empty — skipping bullet generation.")
        tool_context.state[ADDENDUM] = {}
        return {"status": "skipped", "reason": "empty_placement_map"}

    user_prompt = SUGGESTION_USER_TEMPLATE.format(
        extracted_resume  = extracted_resume,
        gap_analysis      = gap_analysis,
        placement_map     = placement_map,
        user_instructions = user_instructions,
        current_year      = CURRENT_YEAR,
    )

    client = get_client()
    out    = generate_json(
        client          = client,
        model           = MODEL,
        system          = SUGGESTION_SYSTEM,
        user            = user_prompt,
        response_schema = ADDENDUM_SCHEMA,
    )

    role_entries = out.get("roles", []) if isinstance(out, dict) else []
    logger.info(f"Generated {len(role_entries)} role entries")

    tool_context.state[ADDENDUM] = out

    logger.info("=== ADDENDUM GENERATOR END ===")
    return {"status": "success", "roles_generated": len(role_entries)}



