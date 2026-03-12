import logging
from google.adk.tools import ToolContext
from state.session_state import ADDENDUM, VALIDATION, POINTS_TO_ADD

logger = logging.getLogger(__name__)

def finalize_points(tool_context: ToolContext) -> dict:
    logger.info("=== FINALIZE POINTS START ===")

    validation = tool_context.state.get(VALIDATION)
    addendum   = tool_context.state.get(ADDENDUM)

    validation_present = validation is not None
    logger.info(f"validation present: {validation_present}")
    logger.info(f"addendum present: {addendum is not None}")

    experience_additions = []
    if isinstance(addendum, dict):
        experience_additions = addendum.get("experience_additions", [])

    logger.info(f"experience_additions count: {len(experience_additions)}")

    tool_context.state[POINTS_TO_ADD] = experience_additions

    logger.info("=== FINALIZE POINTS END ===")
    return {
        "validation_present": validation_present,
        "bullets_count": len(experience_additions),
        "points_to_add": experience_additions,
    }




