import os
from google.adk.agents import LlmAgent

from tools.gap_analysis import analyze_gaps

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

gap_analysis_agent = LlmAgent(
    name="GapAnalysisAgent",
    model=MODEL,
    description="Computes ATS gaps and suggested keywords.",
    instruction="""
You are the Gap Analysis Agent.

Call analyze_gaps() (it reads resume_extracted and jd_extracted from state).
Then respond with the match_score and top keywords.
""",
    tools=[analyze_gaps],
    output_key="s2_gap_result",
)
