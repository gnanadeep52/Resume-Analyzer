import os
from google.adk.agents import LlmAgent

from tools.addendum_generator import generate_addendum_points

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

generation_agent = LlmAgent(
    name="GenerationAgent",
    model=MODEL,
    description="Generates add-only bullets grouped by client/role.",
    instruction="""
You are the Generation Agent.

Call generate_addendum_points() which reads extracted_resume + gap_analysis.
Return a short acknowledgement.
""",
    tools=[generate_addendum_points],
    output_key="s3_generation_result",
)
