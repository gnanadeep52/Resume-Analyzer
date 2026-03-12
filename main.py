import asyncio
import sys
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from pipeline.sequential_pipeline import phase1_agent, phase2_agent
from tools.file_parser import parse_resume_file
from state.session_state import (
    RESUME_RAW,
    JD_RAW,
    USER_INSTRUCTIONS,
    GAP_ANALYSIS,
    EXTRACTED_RESUME,
    PLACEMENT_MAP,
)

# ─────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("tools.gap_analysis").setLevel(logging.INFO)

APP_NAME = "resume_tailor_addendum"
USER_ID = "user"

# Domain keyword sets
SWE_FRONTEND_KEYWORDS = [
    "node.js", "node", "react", "typescript", "javascript",
    "express", "angular", "vue", "html", "css", "frontend",
    "webpack", "next.js", "redux", "rest api", "graphql",
    "tailwind", "sass", "jquery", "bootstrap", "svelte",
]

CLASSICAL_ML_KEYWORDS = [
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "arima", "sarima", "prophet", "random forest", "decision tree",
    "logistic regression", "linear regression", "svm", "kmeans",
    "clustering", "h2o", "statsmodels", "scipy", "numpy",
]

GENAI_KEYWORDS = [
    "langchain", "llamaindex", "openai", "azure openai", "azure ai",
    "bedrock", "vertex ai", "rag", "vector", "embedding", "llm",
    "llama", "gpt", "gemini", "claude", "autogen", "crewai",
    "langgraph", "mcp", "faiss", "pinecone", "weaviate", "milvus",
    "chromadb", "opensearch", "hugging face", "transformers",
    "azure ai foundry", "azure ai search", "azure document intelligence",
]

SWE_ROLE_KEYWORDS = [
    "swe", "software engineer", "software developer",
    "backend", "full stack", "fullstack", "full-stack", "engineer",
]

ML_ROLE_KEYWORDS = [
    "machine learning engineer", "ml engineer", "data engineer",
]

DS_ROLE_KEYWORDS = [
    "data scientist", "analyst", "research",
]

# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────


def suggest_role(skill: str, roles: list[dict]) -> tuple[int, str]:
    """Return (suggested_role_index_1_based, reason)."""
    skill_lower = skill.lower()

    is_swe = any(k in skill_lower for k in SWE_FRONTEND_KEYWORDS)
    is_classical_ml = any(k in skill_lower for k in CLASSICAL_ML_KEYWORDS)
    is_genai = any(k in skill_lower for k in GENAI_KEYWORDS)

    best_idx = None
    best_reason = ""

    for i, role in enumerate(roles, 1):
        title = (role.get("title") or "").lower()
        bullets = " ".join(role.get("bullets") or []).lower()
        end_year_str = role.get("end_date") or ""
        is_active = any(w in end_year_str.lower() for w in ["present", "current", "now"])

        if is_swe:
            if any(k in title for k in SWE_ROLE_KEYWORDS):
                best_idx = i
                best_reason = (
                    f"{skill} is a SWE/frontend tool. "
                    f"This role has backend/API responsibilities "
                    f"that make it the most credible placement."
                )
                break
            if "api" in bullets or "backend" in bullets or "service" in bullets:
                if best_idx is None:
                    best_idx = i
                    best_reason = (
                        f"{skill} is a SWE/frontend tool. "
                        f"This role has backend service work "
                        f"in its bullets that supports this placement."
                    )

        elif is_genai:
            m = re.search(r"\b(19|20)\d{2}\b", end_year_str)
            end_year = int(m.group()) if m else 0
            if is_active or end_year >= 2024:
                best_idx = i
                best_reason = (
                    f"{skill} is a modern GenAI/LLM tool (2024 era). "
                    f"Only currently active roles or roles ending 2024 "
                    f"are valid for this placement."
                )
                break

        elif is_classical_ml:
            if any(k in title for k in ML_ROLE_KEYWORDS + DS_ROLE_KEYWORDS):
                best_idx = i
                best_reason = (
                    f"{skill} is a classical ML tool. "
                    f"This role has model training and ML workflow "
                    f"responsibilities that best match this skill."
                )

    if best_idx is None:
        best_idx = 1
        best_reason = (
            f"No strong domain match found for skill {skill}. "
            f"Defaulting to your most recent role as a safe fallback. "
            f"You can override this if another role fits better."
        )

    return best_idx, best_reason


def display_gaps(gap_analysis: dict) -> list[str]:
    """Print ATS score + missing skills/tools and return combined list."""
    missing = gap_analysis.get("missing", {})
    match_score = gap_analysis.get("match_score", 0)
    missing_skills = missing.get("skills", []) or []
    missing_tools = missing.get("tools", []) or []
    all_missing = missing_skills + missing_tools

    print("-" * 65)
    print(f" ATS MATCH SCORE: {match_score}")
    print("-" * 65)

    if not all_missing:
        print("No gaps found — your resume already covers this JD!")
        return []

    print(f"\n MISSING SKILLS / TOOLS ({len(all_missing)} found):\n")
    for i, item in enumerate(all_missing, 1):
        tag = "skill" if item in missing_skills else "tool"
        print(f"  {i:2}. [{tag}] {item}")
    print()
    return all_missing


def ask_placement_with_suggestions(
    all_missing: list[str], extracted_resume: dict
) -> dict[str, str]:
    """
    Shows suggested role placements for each missing skill/tool and lets
    the user override them. Returns mapping skill_name -> "Title @ Company".
    """
    roles = extracted_resume.get("experience", []) or []

    if not roles:
        print("\n⚠️ No structured experience found in resume. Skipping placement step.")
        return {}

    suggestions: list[dict] = []
    for item in all_missing:
        idx, reason = suggest_role(item, roles)
        suggestions.append(
            {
                "skill": item,
                "suggested_idx": idx,
                "reason": reason,
            }
        )

    print("\n" + "-" * 70)
    print(" YOUR EXPERIENCE ROLES")
    print("-" * 70)
    for i, role in enumerate(roles, 1):
        print(
            f"  {i}. {role.get('title','?')} @ {role.get('company','?')} "
            f"({role.get('start_date','?')} – {role.get('end_date','?')})"
        )

    print("\n" + "-" * 70)
    print(" SUGGESTED PLACEMENTS")
    print("-" * 70)
    print(f"{'#':<4} {'Skill/Tool':<28} {'Suggested Role':<35}")
    print("-" * 70)
    for i, s in enumerate(suggestions, 1):
        role = roles[s["suggested_idx"] - 1]
        role_label = f"Role {s['suggested_idx']}: {role.get('title','?')} @ {role.get('company','?')}"
        print(f"{i:<4} {s['skill']:<28} {role_label:<35}")

    print("\n FULL REASONING")
    print("-" * 70)
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s['skill']}")
        print(f"   {s['reason']}\n")

    print("-" * 70)
    print(" HOW TO RESPOND")
    print("-" * 70)
    print("  • Press Enter alone → Accept ALL suggestions as shown")
    print("  • Type '2:1'       → Move item 2 to Role 1")
    print("  • Type '2:1, 4:0'  → Move item 2 to Role 1, skip item 4")
    print("  • Type '3:0'       → Skip item 3 only")
    print("  • Type '0' or 'all:0' → Skip ALL, generate nothing")
    print("-" * 70)

    while True:
        raw = input("\nYour input (Enter = accept all): ").strip()

        if raw == "":
            placement_map: dict[str, str] = {}
            print("\n ✅ ACCEPTED ALL SUGGESTIONS:\n")
            for s in suggestions:
                role = roles[s["suggested_idx"] - 1]
                role_label = (
                    f"{role.get('title','?')} @ {role.get('company','?')}"
                )
                placement_map[s["skill"]] = role_label
                print(f" ✅ {s['skill']:<28} → {role_label}")
            return placement_map

        if raw.strip() in ("0", "all:0"):
            print("\n ⏭ Skipping all. Nothing will be generated.")
            return {}

        override_map = {
            i: s["suggested_idx"] for i, s in enumerate(suggestions, 1)
        }
        errors: list[str] = []

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            if ":" not in part:
                errors.append(
                    f" ⚠️ Invalid format '{part}' — use #:role e.g. 2:1"
                )
                continue
            left, right = part.split(":", 1)
            try:
                item_num = int(left.strip())
                role_num = int(right.strip())
            except ValueError:
                errors.append(
                    f" ⚠️ '{part}' — both values must be numbers"
                )
                continue

            if not (1 <= item_num <= len(suggestions)):
                errors.append(
                    f" ⚠️ Item {item_num} doesn't exist "
                    f"(valid: 1–{len(suggestions)})"
                )
                continue
            if not (0 <= role_num <= len(roles)):
                errors.append(
                    f" ⚠️ Role {role_num} doesn't exist "
                    f"(valid: 0 = skip, 1–{len(roles)})"
                )
                continue

            override_map[item_num] = role_num

        if errors:
            for e in errors:
                print(e)
            print(" Please try again.\n")
            continue

        placement_map = {}
        print("\n 📋 FINAL PLACEMENT SUMMARY:\n")
        for i, s in enumerate(suggestions, 1):
            chosen = override_map[i]
            is_override = chosen != s["suggested_idx"]

            if chosen == 0:
                print(f" ⏭ {s['skill']:<28} → skipped")
                continue

            role = roles[chosen - 1]
            role_label = (
                f"{role.get('title','?')} @ {role.get('company','?')}"
            )
            placement_map[s["skill"]] = role_label
            tag = " (overridden)" if is_override else " (suggested) "
            print(f" ✅ {s['skill']:<28} → {role_label}{tag}")

        if not placement_map:
            print("\n ⏭ All skills skipped. Nothing will be generated.")
            return {}

        print()
        confirm = input(
            " Confirm these placements? [Enter = yes / n = redo]: "
        ).strip().lower()

        if confirm in ("", "y", "yes"):
            return placement_map

        print("\n 🔄 Redoing overrides...\n")


# ─────────────────────────────────────────────────────────────────────
# Phase runner and orchestrator
# ─────────────────────────────────────────────────────────────────────


async def run_phase(
    agent,
    initial_state: dict,
    message: str,
) -> tuple[str, dict]:
    """
    Run a phase agent with a brand-new InMemorySessionService pre-loaded
    with initial_state. Returns (final_text, state_after).

    InMemorySessionService stores sessions by value (deep copy), so the only
    reliable way to pass state between phases is to create a fresh session
    with the full desired state at the start of each phase.
    """
    svc = InMemorySessionService()
    session = await svc.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state=initial_state,
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=svc)

    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=Content(parts=[Part(text=message)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(
                p.text for p in event.content.parts
                if getattr(p, "text", None)
            )

    # Read back the state the agent wrote
    try:
        updated = await svc.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session.id
        )
        state_after = dict(updated.state) if updated else {}
    except Exception:
        state_after = {}

    return final_text, state_after


async def run(resume_path: str, jd_path: str) -> None:
    # Parse inputs
    parsed = parse_resume_file(resume_path)
    if not parsed["success"]:
        print(f"\n❌ Resume parse failed: {parsed['error']}")
        return

    jd_text = Path(jd_path).read_text(
        encoding="utf-8", errors="replace"
    ).strip()

    user_instructions = input(
        "\nAny extra instructions? (optional, press Enter to skip): "
    ).strip()

    # ── PHASE 1 ──────────────────────────────────────────────────────────
    phase1_state = {
        RESUME_RAW: parsed["content"],
        JD_RAW: jd_text,
        USER_INSTRUCTIONS: user_instructions,
    }

    print("\n⏳ Phase 1: Analyzing your resume and JD...\n")
    _, state_after_p1 = await run_phase(
        phase1_agent,
        phase1_state,
        "Extract resume and JD skills, then compute skill gaps.",
    )

    gap_analysis     = state_after_p1.get(GAP_ANALYSIS)
    extracted_resume = state_after_p1.get(EXTRACTED_RESUME) or {}

    if gap_analysis is None:
        print("\n⚠️ GAP_ANALYSIS missing; computing a simple fallback gap locally.")
        resume_extracted = state_after_p1.get("resume_extracted") or {}
        jd_extracted     = state_after_p1.get("jd_extracted") or {}

        resume_sk    = set(map(str.lower, resume_extracted.get("skills", []) or []))
        resume_tools = set(map(str.lower, resume_extracted.get("tools",  []) or []))
        jd_sk        = jd_extracted.get("skills", []) or []
        jd_tools     = jd_extracted.get("tools",  []) or []

        missing_sk = [s for s in jd_sk    if s.lower() not in resume_sk]
        missing_tl = [t for t in jd_tools if t.lower() not in resume_tools]

        total   = len(jd_sk) + len(jd_tools)
        covered = total - (len(missing_sk) + len(missing_tl))
        match_score = round(100 * covered / total, 1) if total > 0 else 0

        gap_analysis = {
            "match_score": match_score,
            "missing":  {"skills": missing_sk,  "tools": missing_tl},
            "matched":  {
                "skills": [s for s in jd_sk    if s.lower() in resume_sk],
                "tools":  [t for t in jd_tools if t.lower() in resume_tools],
            },
        }

    # Display gaps and get user placement decisions
    all_missing = display_gaps(gap_analysis)
    if not all_missing:
        return

    placement_map = ask_placement_with_suggestions(all_missing, extracted_resume)
    if not placement_map:
        print("\n⏭ No placements confirmed. Exiting.")
        return

    # ── PHASE 2 ──────────────────────────────────────────────────────────
    # Create a fresh session pre-loaded with everything phase 1 produced
    # PLUS the placement_map the user just confirmed.
    phase2_state = {
        **state_after_p1,
        GAP_ANALYSIS:  gap_analysis,   # ensure present even in fallback case
        PLACEMENT_MAP: placement_map,
    }

    print("\n⏳ Phase 2: Generating and validating resume bullets...\n")
    final_text, state_after_p2 = await run_phase(
        phase2_agent,
        phase2_state,
        "Generate validated resume bullets based on the placement map.",
    )

    # If agent output is useful, print it; otherwise fall back to reading state directly
    agent_output = (final_text or "").strip().lower()
    if agent_output in ("done.", "done", "") or len(final_text.strip()) < 20:
        # Fallback: render bullets directly from state
        points = state_after_p2.get("points_to_add") or []
        print("\n" + "=" * 70)
        if points:
            print("\n--- BULLETS TO ADD TO YOUR RESUME ---\n")
            for role in points:
                title    = role.get("title", "?")
                company  = role.get("company", "?")
                start    = role.get("start_date", "?")
                end      = role.get("end_date", "?")
                bullets  = role.get("bullets_to_add", [])
                print(f"Role: {title} @ {company} ({start} – {end})")
                for b in bullets:
                    print(f"  • {b}")
                print()
        else:
            print("No additional bullets were generated.")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print(final_text)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a resume against a job description"
    )
    parser.add_argument(
        "resume",
        nargs="?",
        help="path to resume file (.docx/.pdf/.txt)",
    )
    parser.add_argument(
        "jd",
        nargs="?",
        help="path to job description text file",
    )
    args = parser.parse_args()

    resume_path = args.resume or ""
    jd_path = args.jd or ""

    if resume_path and jd_path:
        resume_path = resume_path.strip('"').strip("'")
        jd_path = jd_path.strip('"').strip("'")
        if not Path(resume_path).exists():
            print(f"❌ Resume file not found: {resume_path}")
            sys.exit(1)
        if not Path(jd_path).exists():
            print(f"❌ JD file not found: {jd_path}")
            sys.exit(1)
    else:
        while True:
            resume_path = input(
                "\nEnter resume path (.docx / .pdf / .txt): "
            ).strip().strip('"').strip("'")
            if resume_path and Path(resume_path).exists():
                break
            print(" ⚠️ File not found. Please try again.")

        while True:
            jd_path = input(
                "Enter JD path (.txt): "
            ).strip().strip('"').strip("'")
            if jd_path and Path(jd_path).exists():
                break
            print(" ⚠️ File not found. Please try again.")

    asyncio.run(run(resume_path, jd_path))








# import asyncio
# import sys
# import logging
# import re
# from pathlib import Path

# from dotenv import load_dotenv
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService
# from google.genai.types import Content, Part

# from pipeline.sequential_pipeline import phase1_agent, phase2_agent
# from tools.file_parser import parse_resume_file
# from state.session_state import (
#     RESUME_RAW,
#     JD_RAW,
#     USER_INSTRUCTIONS,
#     GAP_ANALYSIS,
#     EXTRACTED_RESUME,
#     PLACEMENT_MAP,
# )

# # ─────────────────────────────────────────────────────────────────────
# # Setup
# # ─────────────────────────────────────────────────────────────────────

# load_dotenv(override=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )

# logging.getLogger("tools.gap_analysis").setLevel(logging.INFO)

# APP_NAME = "resume_tailor_addendum"
# USER_ID = "user"

# # Domain keyword sets
# SWE_FRONTEND_KEYWORDS = [
#     "node.js", "node", "react", "typescript", "javascript",
#     "express", "angular", "vue", "html", "css", "frontend",
#     "webpack", "next.js", "redux", "rest api", "graphql",
#     "tailwind", "sass", "jquery", "bootstrap", "svelte",
# ]

# CLASSICAL_ML_KEYWORDS = [
#     "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
#     "arima", "sarima", "prophet", "random forest", "decision tree",
#     "logistic regression", "linear regression", "svm", "kmeans",
#     "clustering", "h2o", "statsmodels", "scipy", "numpy",
# ]

# GENAI_KEYWORDS = [
#     "langchain", "llamaindex", "openai", "azure openai", "azure ai",
#     "bedrock", "vertex ai", "rag", "vector", "embedding", "llm",
#     "llama", "gpt", "gemini", "claude", "autogen", "crewai",
#     "langgraph", "mcp", "faiss", "pinecone", "weaviate", "milvus",
#     "chromadb", "opensearch", "hugging face", "transformers",
#     "azure ai foundry", "azure ai search", "azure document intelligence",
# ]

# SWE_ROLE_KEYWORDS = [
#     "swe", "software engineer", "software developer",
#     "backend", "full stack", "fullstack", "full-stack", "engineer",
# ]

# ML_ROLE_KEYWORDS = [
#     "machine learning engineer", "ml engineer", "data engineer",
# ]

# DS_ROLE_KEYWORDS = [
#     "data scientist", "analyst", "research",
# ]

# # ─────────────────────────────────────────────────────────────────────
# # Helper functions
# # ─────────────────────────────────────────────────────────────────────


# def suggest_role(skill: str, roles: list[dict]) -> tuple[int, str]:
#     """Return (suggested_role_index_1_based, reason)."""
#     skill_lower = skill.lower()

#     is_swe = any(k in skill_lower for k in SWE_FRONTEND_KEYWORDS)
#     is_classical_ml = any(k in skill_lower for k in CLASSICAL_ML_KEYWORDS)
#     is_genai = any(k in skill_lower for k in GENAI_KEYWORDS)

#     best_idx = None
#     best_reason = ""

#     for i, role in enumerate(roles, 1):
#         title = (role.get("title") or "").lower()
#         bullets = " ".join(role.get("bullets") or []).lower()
#         end_year_str = role.get("end_date") or ""
#         is_active = any(w in end_year_str.lower() for w in ["present", "current", "now"])

#         if is_swe:
#             if any(k in title for k in SWE_ROLE_KEYWORDS):
#                 best_idx = i
#                 best_reason = (
#                     f"{skill} is a SWE/frontend tool. "
#                     f"This role has backend/API responsibilities "
#                     f"that make it the most credible placement."
#                 )
#                 break
#             if "api" in bullets or "backend" in bullets or "service" in bullets:
#                 if best_idx is None:
#                     best_idx = i
#                     best_reason = (
#                         f"{skill} is a SWE/frontend tool. "
#                         f"This role has backend service work "
#                         f"in its bullets that supports this placement."
#                     )

#         elif is_genai:
#             m = re.search(r"\b(19|20)\d{2}\b", end_year_str)
#             end_year = int(m.group()) if m else 0
#             if is_active or end_year >= 2024:
#                 best_idx = i
#                 best_reason = (
#                     f"{skill} is a modern GenAI/LLM tool (2024 era). "
#                     f"Only currently active roles or roles ending 2024 "
#                     f"are valid for this placement."
#                 )
#                 break

#         elif is_classical_ml:
#             if any(k in title for k in ML_ROLE_KEYWORDS + DS_ROLE_KEYWORDS):
#                 best_idx = i
#                 best_reason = (
#                     f"{skill} is a classical ML tool. "
#                     f"This role has model training and ML workflow "
#                     f"responsibilities that best match this skill."
#                 )

#     if best_idx is None:
#         best_idx = 1
#         best_reason = (
#             f"No strong domain match found for skill {skill}. "
#             f"Defaulting to your most recent role as a safe fallback. "
#             f"You can override this if another role fits better."
#         )

#     return best_idx, best_reason


# def display_gaps(gap_analysis: dict) -> list[str]:
#     """Print ATS score + missing skills/tools and return combined list."""
#     missing = gap_analysis.get("missing", {})
#     match_score = gap_analysis.get("match_score", 0)
#     missing_skills = missing.get("skills", []) or []
#     missing_tools = missing.get("tools", []) or []
#     all_missing = missing_skills + missing_tools

#     print("-" * 65)
#     print(f" ATS MATCH SCORE: {match_score}")
#     print("-" * 65)

#     if not all_missing:
#         print("No gaps found — your resume already covers this JD!")
#         return []

#     print(f"\n MISSING SKILLS / TOOLS ({len(all_missing)} found):\n")
#     for i, item in enumerate(all_missing, 1):
#         tag = "skill" if item in missing_skills else "tool"
#         print(f"  {i:2}. [{tag}] {item}")
#     print()
#     return all_missing


# def ask_placement_with_suggestions(
#     all_missing: list[str], extracted_resume: dict
# ) -> dict[str, str]:
#     """
#     Shows suggested role placements for each missing skill/tool and lets
#     the user override them. Returns mapping skill_name -> "Title @ Company".
#     """
#     roles = extracted_resume.get("experience", []) or []

#     if not roles:
#         print("\n⚠️ No structured experience found in resume. Skipping placement step.")
#         return {}

#     suggestions: list[dict] = []
#     for item in all_missing:
#         idx, reason = suggest_role(item, roles)
#         suggestions.append(
#             {
#                 "skill": item,
#                 "suggested_idx": idx,
#                 "reason": reason,
#             }
#         )

#     print("\n" + "-" * 70)
#     print(" YOUR EXPERIENCE ROLES")
#     print("-" * 70)
#     for i, role in enumerate(roles, 1):
#         print(
#             f"  {i}. {role.get('title','?')} @ {role.get('company','?')} "
#             f"({role.get('start_date','?')} – {role.get('end_date','?')})"
#         )

#     print("\n" + "-" * 70)
#     print(" SUGGESTED PLACEMENTS")
#     print("-" * 70)
#     print(f"{'#':<4} {'Skill/Tool':<28} {'Suggested Role':<35}")
#     print("-" * 70)
#     for i, s in enumerate(suggestions, 1):
#         role = roles[s["suggested_idx"] - 1]
#         role_label = f"Role {s['suggested_idx']}: {role.get('title','?')} @ {role.get('company','?')}"
#         print(f"{i:<4} {s['skill']:<28} {role_label:<35}")

#     print("\n FULL REASONING")
#     print("-" * 70)
#     for i, s in enumerate(suggestions, 1):
#         print(f"{i}. {s['skill']}")
#         print(f"   {s['reason']}\n")

#     print("-" * 70)
#     print(" HOW TO RESPOND")
#     print("-" * 70)
#     print("  • Press Enter alone → Accept ALL suggestions as shown")
#     print("  • Type '2:1'       → Move item 2 to Role 1")
#     print("  • Type '2:1, 4:0'  → Move item 2 to Role 1, skip item 4")
#     print("  • Type '3:0'       → Skip item 3 only")
#     print("  • Type '0' or 'all:0' → Skip ALL, generate nothing")
#     print("-" * 70)

#     while True:
#         raw = input("\nYour input (Enter = accept all): ").strip()

#         if raw == "":
#             placement_map: dict[str, str] = {}
#             print("\n ✅ ACCEPTED ALL SUGGESTIONS:\n")
#             for s in suggestions:
#                 role = roles[s["suggested_idx"] - 1]
#                 role_label = (
#                     f"{role.get('title','?')} @ {role.get('company','?')}"
#                 )
#                 placement_map[s["skill"]] = role_label
#                 print(f" ✅ {s['skill']:<28} → {role_label}")
#             return placement_map

#         if raw.strip() in ("0", "all:0"):
#             print("\n ⏭ Skipping all. Nothing will be generated.")
#             return {}

#         override_map = {
#             i: s["suggested_idx"] for i, s in enumerate(suggestions, 1)
#         }
#         errors: list[str] = []

#         parts = [p.strip() for p in raw.split(",") if p.strip()]
#         for part in parts:
#             if ":" not in part:
#                 errors.append(
#                     f" ⚠️ Invalid format '{part}' — use #:role e.g. 2:1"
#                 )
#                 continue
#             left, right = part.split(":", 1)
#             try:
#                 item_num = int(left.strip())
#                 role_num = int(right.strip())
#             except ValueError:
#                 errors.append(
#                     f" ⚠️ '{part}' — both values must be numbers"
#                 )
#                 continue

#             if not (1 <= item_num <= len(suggestions)):
#                 errors.append(
#                     f" ⚠️ Item {item_num} doesn't exist "
#                     f"(valid: 1–{len(suggestions)})"
#                 )
#                 continue
#             if not (0 <= role_num <= len(roles)):
#                 errors.append(
#                     f" ⚠️ Role {role_num} doesn't exist "
#                     f"(valid: 0 = skip, 1–{len(roles)})"
#                 )
#                 continue

#             override_map[item_num] = role_num

#         if errors:
#             for e in errors:
#                 print(e)
#             print(" Please try again.\n")
#             continue

#         placement_map = {}
#         print("\n 📋 FINAL PLACEMENT SUMMARY:\n")
#         for i, s in enumerate(suggestions, 1):
#             chosen = override_map[i]
#             is_override = chosen != s["suggested_idx"]

#             if chosen == 0:
#                 print(f" ⏭ {s['skill']:<28} → skipped")
#                 continue

#             role = roles[chosen - 1]
#             role_label = (
#                 f"{role.get('title','?')} @ {role.get('company','?')}"
#             )
#             placement_map[s["skill"]] = role_label
#             tag = " (overridden)" if is_override else " (suggested) "
#             print(f" ✅ {s['skill']:<28} → {role_label}{tag}")

#         if not placement_map:
#             print("\n ⏭ All skills skipped. Nothing will be generated.")
#             return {}

#         print()
#         confirm = input(
#             " Confirm these placements? [Enter = yes / n = redo]: "
#         ).strip().lower()

#         if confirm in ("", "y", "yes"):
#             return placement_map

#         print("\n 🔄 Redoing overrides...\n")


# # ─────────────────────────────────────────────────────────────────────
# # Phase runner and orchestrator
# # ─────────────────────────────────────────────────────────────────────


# async def run_phase(
#     agent,
#     initial_state: dict,
#     message: str,
# ) -> tuple[str, dict]:
#     """
#     Run a phase agent with a brand-new InMemorySessionService pre-loaded
#     with initial_state. Returns (final_text, state_after).

#     InMemorySessionService stores sessions by value (deep copy), so the only
#     reliable way to pass state between phases is to create a fresh session
#     with the full desired state at the start of each phase.
#     """
#     svc = InMemorySessionService()
#     session = await svc.create_session(
#         app_name=APP_NAME,
#         user_id=USER_ID,
#         state=initial_state,
#     )
#     runner = Runner(agent=agent, app_name=APP_NAME, session_service=svc)

#     final_text = ""
#     async for event in runner.run_async(
#         user_id=USER_ID,
#         session_id=session.id,
#         new_message=Content(parts=[Part(text=message)]),
#     ):
#         if event.is_final_response() and event.content and event.content.parts:
#             final_text = "".join(
#                 p.text for p in event.content.parts
#                 if getattr(p, "text", None)
#             )

#     # Read back the state the agent wrote
#     try:
#         updated = await svc.get_session(
#             app_name=APP_NAME, user_id=USER_ID, session_id=session.id
#         )
#         state_after = dict(updated.state) if updated else {}
#     except Exception:
#         state_after = {}

#     return final_text, state_after


# async def run(resume_path: str, jd_path: str) -> None:
#     # Parse inputs
#     parsed = parse_resume_file(resume_path)
#     if not parsed["success"]:
#         print(f"\n❌ Resume parse failed: {parsed['error']}")
#         return

#     jd_text = Path(jd_path).read_text(
#         encoding="utf-8", errors="replace"
#     ).strip()

#     user_instructions = input(
#         "\nAny extra instructions? (optional, press Enter to skip): "
#     ).strip()

#     # ── PHASE 1 ──────────────────────────────────────────────────────────
#     phase1_state = {
#         RESUME_RAW: parsed["content"],
#         JD_RAW: jd_text,
#         USER_INSTRUCTIONS: user_instructions,
#     }

#     print("\n⏳ Phase 1: Analyzing your resume and JD...\n")
#     _, state_after_p1 = await run_phase(
#         phase1_agent,
#         phase1_state,
#         "Extract resume and JD skills, then compute skill gaps.",
#     )

#     gap_analysis     = state_after_p1.get(GAP_ANALYSIS)
#     extracted_resume = state_after_p1.get(EXTRACTED_RESUME) or {}

#     if gap_analysis is None:
#         print("\n⚠️ GAP_ANALYSIS missing; computing a simple fallback gap locally.")
#         resume_extracted = state_after_p1.get("resume_extracted") or {}
#         jd_extracted     = state_after_p1.get("jd_extracted") or {}

#         resume_sk    = set(map(str.lower, resume_extracted.get("skills", []) or []))
#         resume_tools = set(map(str.lower, resume_extracted.get("tools",  []) or []))
#         jd_sk        = jd_extracted.get("skills", []) or []
#         jd_tools     = jd_extracted.get("tools",  []) or []

#         missing_sk = [s for s in jd_sk    if s.lower() not in resume_sk]
#         missing_tl = [t for t in jd_tools if t.lower() not in resume_tools]

#         total   = len(jd_sk) + len(jd_tools)
#         covered = total - (len(missing_sk) + len(missing_tl))
#         match_score = round(100 * covered / total, 1) if total > 0 else 0

#         gap_analysis = {
#             "match_score": match_score,
#             "missing":  {"skills": missing_sk,  "tools": missing_tl},
#             "matched":  {
#                 "skills": [s for s in jd_sk    if s.lower() in resume_sk],
#                 "tools":  [t for t in jd_tools if t.lower() in resume_tools],
#             },
#         }

#     # Display gaps and get user placement decisions
#     all_missing = display_gaps(gap_analysis)
#     if not all_missing:
#         return

#     placement_map = ask_placement_with_suggestions(all_missing, extracted_resume)
#     if not placement_map:
#         print("\n⏭ No placements confirmed. Exiting.")
#         return

#     # ── PHASE 2 ──────────────────────────────────────────────────────────
#     # Create a fresh session pre-loaded with everything phase 1 produced
#     # PLUS the placement_map the user just confirmed.
#     phase2_state = {
#         **state_after_p1,
#         GAP_ANALYSIS:  gap_analysis,   # ensure present even in fallback case
#         PLACEMENT_MAP: placement_map,
#     }

#     print("\n⏳ Phase 2: Generating and validating resume bullets...\n")
#     final_text, state_after_p2 = await run_phase(
#         phase2_agent,
#         phase2_state,
#         "Generate validated resume bullets based on the placement map.",
#     )

#     # If agent output is useful, print it; otherwise fall back to reading state directly
#     agent_output = (final_text or "").strip().lower()
#     if agent_output in ("done.", "done", "") or len(final_text.strip()) < 20:
#         # Fallback: render bullets directly from state
#         points = state_after_p2.get("points_to_add") or []
#         print("\n" + "=" * 70)
#         if points:
#             print("\n--- BULLETS TO ADD TO YOUR RESUME ---\n")
#             for role in points:
#                 title    = role.get("title", "?")
#                 company  = role.get("company", "?")
#                 start    = role.get("start_date", "?")
#                 end      = role.get("end_date", "?")
#                 bullets  = role.get("bullets_to_add", [])
#                 print(f"Role: {title} @ {company} ({start} – {end})")
#                 for b in bullets:
#                     print(f"  • {b}")
#                 print()
#         else:
#             print("No additional bullets were generated.")
#         print("=" * 70 + "\n")
#     else:
#         print("\n" + "=" * 70)
#         print(final_text)
#         print("=" * 70 + "\n")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Analyze a resume against a job description"
#     )
#     parser.add_argument(
#         "resume",
#         nargs="?",
#         help="path to resume file (.docx/.pdf/.txt)",
#     )
#     parser.add_argument(
#         "jd",
#         nargs="?",
#         help="path to job description text file",
#     )
#     args = parser.parse_args()

#     resume_path = args.resume or ""
#     jd_path = args.jd or ""

#     if resume_path and jd_path:
#         resume_path = resume_path.strip('"').strip("'")
#         jd_path = jd_path.strip('"').strip("'")
#         if not Path(resume_path).exists():
#             print(f"❌ Resume file not found: {resume_path}")
#             sys.exit(1)
#         if not Path(jd_path).exists():
#             print(f"❌ JD file not found: {jd_path}")
#             sys.exit(1)
#     else:
#         while True:
#             resume_path = input(
#                 "\nEnter resume path (.docx / .pdf / .txt): "
#             ).strip().strip('"').strip("'")
#             if resume_path and Path(resume_path).exists():
#                 break
#             print(" ⚠️ File not found. Please try again.")

#         while True:
#             jd_path = input(
#                 "Enter JD path (.txt): "
#             ).strip().strip('"').strip("'")
#             if jd_path and Path(jd_path).exists():
#                 break
#             print(" ⚠️ File not found. Please try again.")

#     asyncio.run(run(resume_path, jd_path))