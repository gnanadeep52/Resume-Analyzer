# pipeline/sequential_pipeline.py

from google.adk.agents import LlmAgent, SequentialAgent

from tools.extraction import (
    extract_full_resume_structure,
    extract_resume_skills,
    extract_jd_skills,
)
from tools.gap_analysis import analyze_gaps
from tools.addendum_generator import generate_addendum_points
from tools.validation import validate_bullets
from tools.finalize_points import finalize_points

MODEL_NAME = "gemini-2.0-flash"

# Phase 1: extract everything + gap analysis
phase1_agent = SequentialAgent(
    name="Phase1Agent",
    description="Extract resume/JD and compute skill gaps.",
    sub_agents=[
        LlmAgent(
            name="ExtractionStructureAgent",
            model=MODEL_NAME,
            description="Parse full resume structure (roles & bullets).",
            instruction="Call extract_full_resume_structure() and wait for it to complete. Then respond with: done.",
            tools=[extract_full_resume_structure],
            output_key="s_extracted_resume",
        ),
        LlmAgent(
            name="ExtractionResumeSkillsAgent",
            model=MODEL_NAME,
            description="Extract resume skills/tools from experience sections.",
            instruction="Call extract_resume_skills() and wait for it to complete. Then respond with: done.",
            tools=[extract_resume_skills],
            output_key="s_resume_extracted",
        ),
        LlmAgent(
            name="ExtractionJDSkillsAgent",
            model=MODEL_NAME,
            description="Extract JD skills/tools.",
            instruction="Call extract_jd_skills() and wait for it to complete. Then respond with: done.",
            tools=[extract_jd_skills],
            output_key="s_jd_extracted",
        ),
        LlmAgent(
            name="GapAnalysisAgent",
            model=MODEL_NAME,
            description="Compute ATS score and missing skills/tools.",
            instruction="Call analyze_gaps() and wait for it to complete. Then respond with: done.",
            tools=[analyze_gaps],
            output_key="s_gap_analysis",
        ),
    ],
)

# Phase 2: addendum + validation + finalize
phase2_agent = SequentialAgent(
    name="Phase2Agent",
    description="Generate addendum bullets, validate, and finalize.",
    sub_agents=[
        LlmAgent(
            name="AddendumAgent",
            model=MODEL_NAME,
            description="Generate addendum bullets from placement map.",
            instruction="Call generate_addendum_points() and wait for it to complete. Then respond with: done.",
            tools=[generate_addendum_points],
            output_key="s_addendum",
        ),
        LlmAgent(
            name="ValidationAgent",
            model=MODEL_NAME,
            description="Validate bullets (ATS, timeframe, issues).",
            instruction="Call validate_bullets() and wait for it to complete. Then respond with: done.",
            tools=[validate_bullets],
            output_key="s_validation",
        ),
        LlmAgent(
            name="FinalizeAgent",
            model=MODEL_NAME,
            description="Finalize bullets and display the final output to the user.",
            instruction="""
You are the Finalize Agent.

Call finalize_points() and base your ENTIRE response ONLY on what that tool returns.

After the tool returns, follow these rules:

CASE 1 — points_to_add is non-empty:
  Print the bullets grouped by role in this exact format:

  --- BULLETS TO ADD TO YOUR RESUME ---

  Role: <title> @ <company> (<start_date> – <end_date>)
  • <bullet 1>
  • <bullet 2>
  (repeat for each role)

CASE 2 — points_to_add is empty:
  Print exactly:
  "No additional bullets are needed — your resume already covers the requirements well."
""",
            tools=[finalize_points],
            output_key="s_finalize",
        ),
    ],
)

