"""
streamlit_app.py — Resume Analyzer
Run: streamlit run streamlit_app.py
"""
import asyncio, re, sys, tempfile, logging
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from pipeline.sequential_pipeline import phase1_agent, phase2_agent
from tools.file_parser import parse_resume_file
from state.session_state import (
    RESUME_RAW, JD_RAW, USER_INSTRUCTIONS,
    GAP_ANALYSIS, EXTRACTED_RESUME, PLACEMENT_MAP,
)

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Analyzer", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 980px; }
h1,h2,h3,h4 { color: #e8e8f0 !important; }
.stButton > button {
    background: #7c6af7 !important; border: none !important; color: #fff !important;
    font-weight: 700 !important; border-radius: 99px !important; padding: 8px 24px !important;
}
.stButton > button:disabled { background: #2a2a3a !important; color: #6b6b88 !important; }
.stTextArea textarea, .stTextInput input {
    background: #111118 !important; border-color: #2a2a3a !important;
    color: #e8e8f0 !important; border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: #111118 !important; border-color: #2a2a3a !important; color: #e8e8f0 !important;
}
[data-testid="stFileUploader"] { background: #16161f !important; border-radius: 10px !important; }
[data-testid="stExpander"] { background: #16161f !important; border: 1px solid #2a2a3a !important; border-radius: 10px !important; }
div[data-testid="stCheckbox"] label { color: #e8e8f0 !important; }
hr { border-color: #2a2a3a !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Domain helpers ────────────────────────────────────────────────────
SWE_KW  = ["node.js","node","react","typescript","javascript","express","angular","vue","html","css","frontend","webpack","next.js","redux","graphql","tailwind","svelte"]
CML_KW  = ["scikit-learn","sklearn","xgboost","lightgbm","catboost","arima","sarima","prophet","random forest","decision tree","logistic regression","linear regression","svm","kmeans","clustering","statsmodels"]
GEN_KW  = ["langchain","llamaindex","openai","azure openai","bedrock","vertex ai","rag","vector","embedding","llm","llama","gpt","gemini","claude","autogen","crewai","langgraph","faiss","pinecone","hugging face","transformers"]
SWE_RL  = ["swe","software engineer","software developer","backend","full stack","fullstack","engineer"]
MLRL    = ["machine learning engineer","ml engineer","data engineer"]
DSRL    = ["data scientist","analyst","research"]

def suggest_role(skill, roles):
    sl = skill.lower()
    is_swe = any(k in sl for k in SWE_KW)
    is_cml = any(k in sl for k in CML_KW)
    is_gen = any(k in sl for k in GEN_KW)
    best_idx, best_reason = 1, f"Defaulting to most recent role for {skill}."
    for i, r in enumerate(roles, 1):
        title = (r.get("title") or "").lower()
        end   = r.get("end_date") or ""
        active = any(w in end.lower() for w in ["present","current","now"])
        if is_swe and any(k in title for k in SWE_RL):
            return i, f"{skill} is SWE/frontend — matched to role with engineering context."
        if is_gen:
            m = re.search(r"\b(20)\d{2}\b", end)
            if active or (m and int(m.group()) >= 2024):
                return i, f"{skill} is GenAI — matched to most recent/active role."
        if is_cml and any(k in title for k in MLRL + DSRL):
            best_idx, best_reason = i, f"{skill} is classical ML — matched to ML/DS role."
    return best_idx, best_reason


# ── Async runner ──────────────────────────────────────────────────────
async def _run(agent, state, msg):
    svc     = InMemorySessionService()
    session = await svc.create_session(app_name="ra", user_id="u", state=state)
    runner  = Runner(agent=agent, app_name="ra", session_service=svc)
    final   = ""
    async for ev in runner.run_async(user_id="u", session_id=session.id,
                                     new_message=Content(parts=[Part(text=msg)])):
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = "".join(p.text for p in ev.content.parts if getattr(p,"text",None))
    try:
        up = await svc.get_session(app_name="ra", user_id="u", session_id=session.id)
        return final, dict(up.state) if up else {}
    except Exception:
        return final, {}

def run(agent, state, msg):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _run(agent, state, msg)).result()
        return loop.run_until_complete(_run(agent, state, msg))
    except RuntimeError:
        return asyncio.run(_run(agent, state, msg))


# ── UI helpers (NO div HTML) ──────────────────────────────────────────
def pill(text, color="#7c6af7"):
    """Inline colored pill — only <span> which Streamlit allows."""
    st.markdown(
        f'<span style="padding:3px 12px;border-radius:99px;background:{color}22;'
        f'border:1px solid {color}55;color:{color};font-family:monospace;font-size:12px;">'
        f'{text}</span>',
        unsafe_allow_html=True,
    )

def pills_row(items, color):
    """Row of pill spans on one line — spans are allowed by Streamlit sanitizer."""
    if not items:
        return
    html = " ".join(
        f'<span style="padding:3px 12px;border-radius:99px;margin:2px;'
        f'background:{color}22;border:1px solid {color}55;color:{color};'
        f'font-family:monospace;font-size:12px;display:inline-block;">{x}</span>'
        for x in items
    )
    st.markdown(html, unsafe_allow_html=True)

def step_indicator(current):
    steps = ["1 · Upload", "2 · Analysis", "3 · Placement", "4 · Results"]
    cols  = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            if i < current:
                st.success(f"✓ {label}")
            elif i == current:
                st.info(f"▶ {label}")
            else:
                st.caption(label)

def section(title):
    st.markdown(f"**{title}**")
    st.markdown('<hr style="border-color:#2a2a3a;margin:4px 0 12px 0;">', unsafe_allow_html=True)

def role_card(num, title, company, start, end):
    """Render a role using only native Streamlit — no div HTML."""
    c1, c2 = st.columns([0.07, 0.93])
    with c1:
        # Use st.metric or just bold text for the number badge
        st.markdown(f"**{num}**")
    with c2:
        st.markdown(f"**{title}** :violet[@ {company}]")
        st.caption(f"{start} – {end}")
    st.markdown('<hr style="border-color:#1e1e2a;margin:6px 0;">', unsafe_allow_html=True)

def bullet_card(text):
    """Render a bullet using only native Streamlit."""
    st.markdown(f":violet[•] {text}")


# ── Session init ──────────────────────────────────────────────────────
for k, v in {"step":0,"p1_state":None,"gap":None,"ext_res":None,
              "suggestions":[],"roles":[],"results":None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Header ────────────────────────────────────────────────────────────
st.markdown("## ⚡ Resume Analyzer")
st.caption("AI-powered ATS gap analysis & bullet generation")
st.markdown("---")
step_indicator(st.session_state.step)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# STEP 0 — Upload
# ════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Resume")
        resume_file = st.file_uploader("Upload resume (.pdf / .docx / .txt)",
                                        type=["pdf","docx","txt"])
    with c2:
        section("Job Description")
        jd_text = st.text_area("Paste JD here", height=200, label_visibility="collapsed",
                                placeholder="Paste the full job description...")

    section("Extra Instructions (optional)")
    instructions = st.text_input("", placeholder="e.g. Focus on cloud, keep bullets under 20 words...")

    st.markdown("")
    _, btn = st.columns([3,1])
    with btn:
        go = st.button("Analyze →", use_container_width=True)

    if go:
        if not resume_file:
            st.error("Please upload a resume.")
        elif not jd_text or len(jd_text.strip()) < 20:
            st.error("Please paste the job description.")
        else:
            with st.spinner("Extracting skills & computing gaps…"):
                suffix   = Path(resume_file.name).suffix
                tmp      = Path(tempfile.mktemp(suffix=suffix))
                tmp.write_bytes(resume_file.read())
                parsed   = parse_resume_file(str(tmp))
                tmp.unlink(missing_ok=True)
                if not parsed["success"]:
                    st.error(f"Parse failed: {parsed['error']}"); st.stop()

                _, p1 = run(phase1_agent,
                    {RESUME_RAW: parsed["content"], JD_RAW: jd_text.strip(),
                     USER_INSTRUCTIONS: instructions.strip()},
                    "Extract resume and JD skills, then compute skill gaps.")

                gap     = p1.get(GAP_ANALYSIS)
                ext_res = p1.get(EXTRACTED_RESUME) or {}

                if gap is None:
                    rex = p1.get("resume_extracted") or {}
                    jex = p1.get("jd_extracted") or {}
                    rsk = set(map(str.lower, rex.get("skills",[]) or []))
                    rtl = set(map(str.lower, rex.get("tools", []) or []))
                    jsk = jex.get("skills",[]) or []; jtl = jex.get("tools",[]) or []
                    msk = [s for s in jsk if s.lower() not in rsk]
                    mtl = [t for t in jtl if t.lower() not in rtl]
                    tot = len(jsk)+len(jtl); cov = tot-len(msk)-len(mtl)
                    gap = {"match_score": round(100*cov/tot,1) if tot else 0,
                           "missing": {"skills":msk,"tools":mtl},
                           "matched": {"skills":[s for s in jsk if s.lower() in rsk],
                                       "tools": [t for t in jtl if t.lower() in rtl]}}

                roles   = ext_res.get("experience",[]) or []
                all_mis = (gap.get("missing",{}).get("skills",[]) or []) + \
                          (gap.get("missing",{}).get("tools", []) or [])
                sugg    = []
                for item in all_mis:
                    idx, reason = suggest_role(item, roles)
                    role = roles[idx-1] if roles else {}
                    sugg.append({"skill":item,"suggested_idx":idx,
                                 "suggested_role":f"{role.get('title','?')} @ {role.get('company','?')}",
                                 "reason":reason})

                st.session_state.update(step=1, p1_state=p1, gap=gap,
                                        ext_res=ext_res, suggestions=sugg, roles=roles)
                st.rerun()


# ════════════════════════════════════════════════════════════════════
# STEP 1 — Analysis
# ════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    gap     = st.session_state.gap
    roles   = st.session_state.roles
    missing = (gap.get("missing",{}).get("skills",[]) or []) + \
              (gap.get("missing",{}).get("tools", []) or [])
    matched = (gap.get("matched",{}).get("skills",[]) or []) + \
              (gap.get("matched",{}).get("tools", []) or [])
    score   = gap.get("match_score", 0)

    # Score row
    s_col, m_col, mm_col = st.columns(3)
    with s_col:
        color = "normal" if score >= 70 else "off"
        st.metric("ATS Match Score", f"{int(score)}%",
                  delta="Strong" if score>=70 else "Partial" if score>=40 else "Low")
    with m_col:
        st.metric("Missing", len(missing), delta=f"-{len(missing)}", delta_color="inverse")
    with mm_col:
        st.metric("Covered", len(matched), delta=f"+{len(matched)}", delta_color="normal")

    st.markdown("---")

    # Missing skills
    section(f"❌ Missing Skills & Tools ({len(missing)})")
    if missing:
        pills_row(missing, "#f76a8c")
    else:
        st.success("✓ All requirements are already covered!")

    st.markdown("")

    # Matched skills
    section(f"✓ Already Covered ({len(matched)})")
    if matched:
        pills_row(matched, "#4af7a0")
    else:
        st.caption("No matches found.")

    st.markdown("---")

    # Roles
    section(f"Your Experience ({len(roles)} roles)")
    for i, r in enumerate(roles, 1):
        role_card(i, r.get("title","?"), r.get("company","?"),
                  r.get("start_date","?"), r.get("end_date","?"))

    st.markdown("")
    c1, _, c3 = st.columns([1,2,1])
    with c1:
        if st.button("← Start Over"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    with c3:
        if missing:
            if st.button("Set Placements →", use_container_width=True):
                st.session_state.step = 2; st.rerun()
        else:
            st.success("No gaps to fill!")


# ════════════════════════════════════════════════════════════════════
# STEP 2 — Placement
# ════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    suggestions = st.session_state.suggestions
    roles       = st.session_state.roles

    role_opts = {
        f"Role {i+1}: {r.get('title','?')} @ {r.get('company','?')} "
        f"({r.get('start_date','?')} – {r.get('end_date','?')})":
        f"{r.get('title','?')} @ {r.get('company','?')}"
        for i, r in enumerate(roles)
    }
    role_labels = list(role_opts.keys())

    st.info("For each missing skill, choose which role the bullet should go under. Uncheck to skip.")
    st.markdown("")

    placements = {}
    for s in suggestions:
        default_lbl = next((l for l,v in role_opts.items() if v == s["suggested_role"]), role_labels[0] if role_labels else "")
        default_idx = role_labels.index(default_lbl) if default_lbl in role_labels else 0

        with st.container():
            chk_col, content_col = st.columns([0.06, 0.94])
            with chk_col:
                include = st.checkbox("", value=True, key=f"chk_{s['skill']}")
            with content_col:
                # Header line: skill name + badge — only spans, no divs
                st.markdown(
                    f'**{s["skill"]}** '
                    f'<span style="padding:2px 10px;border-radius:99px;'
                    f'background:#7c6af722;border:1px solid #7c6af755;'
                    f'color:#7c6af7;font-family:monospace;font-size:10px;">missing</span>',
                    unsafe_allow_html=True,
                )
                selected = st.selectbox("Role", options=role_labels, index=default_idx,
                                        key=f"sel_{s['skill']}", disabled=not include,
                                        label_visibility="collapsed")
                st.caption(f"💡 {s['reason']}")
            st.markdown("")

        if include:
            placements[s["skill"]] = role_opts[selected]

    st.markdown("---")
    cnt = len(placements)
    c1, c_mid, c3 = st.columns([1,2,1])
    with c1:
        if st.button("← Back"):
            st.session_state.step = 1; st.rerun()
    with c_mid:
        st.caption(f"{cnt} of {len(suggestions)} skills selected")
    with c3:
        gen = st.button(f"Generate ({cnt}) →", disabled=cnt==0, use_container_width=True)

    if gen and cnt > 0:
        with st.spinner("Generating & validating bullets… (~30s)"):
            _, p2 = run(phase2_agent,
                {**st.session_state.p1_state, GAP_ANALYSIS: st.session_state.gap,
                 PLACEMENT_MAP: placements},
                "Generate validated resume bullets based on the placement map.")
            st.session_state.results = {
                "points_to_add": p2.get("points_to_add") or [],
                "validation":    p2.get("validation"),
            }
            st.session_state.step = 3
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# STEP 3 — Results
# ════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    results    = st.session_state.results or {}
    points     = results.get("points_to_add", [])
    validation = results.get("validation")

    st.markdown("### ⚡ Resume Bullets Generated")

    # Validation metrics
    if validation:
        v1, v2, v3, v4 = st.columns(4)
        passed = validation.get("passed", False)
        with v1: st.metric("Validation", "✓ Passed" if passed else "✗ Failed")
        with v2: st.metric("ATS Grade", validation.get("ats_grade","?"))
        with v3: st.metric("Keyword Coverage", f"{int(validation.get('keyword_coverage_pct',0))}%")
        with v4: st.metric("Roles Updated", len(points))

    st.markdown("---")

    if not points:
        st.success("✓ Your resume already covers all requirements. No additional bullets needed.")
    else:
        # Download
        all_text = "\n\n".join(
            f"{r.get('title','?')} @ {r.get('company','?')} ({r.get('start_date','?')} – {r.get('end_date','?')})\n" +
            "\n".join(f"• {b}" for b in (r.get("bullets_to_add") or []))
            for r in points
        )
        st.download_button("⬇ Download All Bullets (.txt)", data=all_text,
                           file_name="resume_bullets.txt", mime="text/plain")
        st.markdown("")

        for ri, role in enumerate(points):
            title   = role.get("title","?")
            company = role.get("company","?")
            start   = role.get("start_date","?")
            end     = role.get("end_date","?")
            bullets = role.get("bullets_to_add") or []
            notes   = role.get("notes","")

            with st.expander(f"{ri+1}. {title} @ {company}  ·  {len(bullets)} bullets", expanded=True):
                st.caption(f"{start} – {end}")
                st.markdown("---")
                for b in bullets:
                    bullet_card(b)
                if notes:
                    st.info(f"💡 {notes}")

        if validation and validation.get("issues"):
            st.markdown("---")
            section("⚠️ Advisory Notes")
            for issue in (validation.get("issues") or []):
                st.caption(f"• {issue}")

    st.markdown("---")
    if st.button("← Start Over"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()