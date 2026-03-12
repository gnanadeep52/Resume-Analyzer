import os
from google.adk.agents import LlmAgent

from tools.extraction import extract_full_resume_structure, extract_resume_skills, extract_jd_skills

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

extraction_agent = LlmAgent(
    name="ExtractionAgent",
    model=MODEL,
    description="Extracts resume structure + proven resume skills + JD skills.",
    instruction="""
You are the Extraction Agent.

Call tools in this exact order — each tool reads its inputs automatically from session state:
1) extract_full_resume_structure()
2) extract_resume_skills()
3) extract_jd_skills()

Wait for each tool to complete before calling the next.
Then respond with: extraction complete.
""",
    tools=[extract_full_resume_structure, extract_resume_skills, extract_jd_skills],
    output_key="s1_extraction_result",
)


