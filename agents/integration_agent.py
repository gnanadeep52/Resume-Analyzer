import os
from google.adk.agents import LlmAgent

from tools.finalize_points import finalize_points

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

integration_agent = LlmAgent(
    name="IntegrationAgent",
    model=MODEL,
    description="Final step: returns ONLY the points to add (no resume rewrite).",
    instruction="""
You are the Integration Agent.

Call finalize_points_to_add() and base your ENTIRE response ONLY on what that tool returns.
Ignore all previous agent outputs, session history, or any other context.

After the tool returns, follow these rules:

CASE 1 — tool returns passed=true AND points_to_add is non-empty:
  Print the bullets grouped by role in this format:

  ✅ Validation: PASSED (Grade: <ats_grade>, Keyword Coverage: <keyword_coverage_pct>%)

  --- BULLETS TO ADD TO YOUR RESUME ---

  Role: <title> @ <company> (<start_date> – <end_date>)
  • <bullet 1>
  • <bullet 2>
  ... (repeat for each role)

  Keywords Incorporated: <keywords_incorporated as comma-separated list>

  If issues[] is non-empty, add at the end:
  ⚠️ Advisory suggestions (optional improvements):
  • <issue 1>
  • <issue 2>

CASE 2 — tool returns passed=true AND points_to_add is empty:
  Print:
  "The gap analysis found no missing skills or tools for this JD —
   your resume already covers the requirements well.
   No additional bullets are needed."

CASE 3 — tool returns passed=false (timeframe violations exist):
  Print:
  ❌ Validation FAILED — Timeframe Violations:
  • <violation 1>
  • <violation 2>

  Issues:
  • <issue 1>
""",
    tools=[finalize_points],
    output_key="final_points_result",
)