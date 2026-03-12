import os
from google.adk.agents import LlmAgent

from tools.validation import validate_bullets

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

validation_agent = LlmAgent(
    name="ValidationAgent",
    model=MODEL,
    description="Validates the addendum bullets for ATS + timeframe compliance.",
    instruction="""
You are the Validation Agent.

Call validate_bullets() which reads addendum + extracted_resume + gap_analysis.

After the tool returns, respond ONLY in this exact format:
  passed: <true or false>
  grade: <A/B/C/D>
  issues_count: <number>
  timeframe_violations_count: <number>

RULES:
1) If tool returns passed=true → write passed: true
2) If tool returns passed=false with at least one issue or timeframe violation → write passed: false and list them.
3) If tool returns passed=false but issues=[] AND timeframe_violations=[] →
   write passed: true (no issues found — treat as approved).
   Do NOT write passed: false in this case.
""",
    tools=[validate_bullets],
    output_key="s4_validation_result",
)