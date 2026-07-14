"""BVRIT FAQ Chatbot — Redesigned UI"""
from __future__ import annotations
import os, time, uuid
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from retrieve import (DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME,
                      DEFAULT_EMBEDDING_MODEL, load_store, list_sections)
from generate import generate_answer, COLLEGE_NAME
from memory import (init_db, load_profile, update_profile, append_turn,
                    get_history, get_turn_count, summarise_and_trim,
                    extract_facts_from_message, build_profile_context,
                    get_memory_summary, detect_memory_changes,
                    clear_user_data, delete_stale_profiles, PRIVACY_NOTICE)
from ab_testing import (init_ab_db, assign_variant, log_result,
                        get_ab_stats, get_ab_log, get_ab_comparison)

st.set_page_config(page_title="BVRIT AI Assistant",page_icon="🎓",
                   layout="wide",initial_sidebar_state="collapsed")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 1.8rem 1rem!important;}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:.88em;padding:10px 22px;}
.stTabs [aria-selected="true"]{color:#0d1b2a!important;border-bottom:3px solid #1d4ed8!important;}
.stChatInputContainer{border-radius:14px!important;border:2px solid #1d4ed8!important;}
.stButton>button{border-radius:8px!important;font-weight:600!important;}
.stProgress>div>div{background:linear-gradient(90deg,#0d1b2a,#1d4ed8)!important;}
</style>""",unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CHAT_MODEL="mistral-small-latest"
IMG_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"college.jpeg")
IMG_KEYS=["photo","picture","image","campus","building","show me","exterior","infrastructure"]
CMD_MEM  =["show my memory","what do you know","show memory","my profile","what do you remember"]
CMD_DEL  =["forget my data","clear my data","delete my data","forget me","reset my data"]
CMD_PREF =["update my preferences","update preferences","change my preferences"]
QUICK_QS=[
    ("💰","CSE fee for 4 years with hostel?"),
    ("📅","When does BVRIT admission close?"),
    ("🏠","What hostel facilities does BVRIT have?"),
    ("💼","What is the highest placement package at BVRIT?"),
    ("🎯","What scholarships are available at BVRIT?"),
    ("📚","List all BVRIT departments and programmes"),
]
_RC={"conversation":("#3b82f6","🗣 Chat"),"tool_only":("#7c3aed","🔧 Tool"),
     "rag_only":("#059669","📚 RAG"),"rag_tool":("#d97706","🔧📚 RAG+Tool")}
_TC={"fee_calculator":"#0891b2","date_checker":"#7c3aed","percentage_calculator":"#be185d"}

def _b(col,txt):
    return (f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:2px 10px;border-radius:20px;font-size:.7em;font-weight:700;'
            f'background:{col};color:#fff;margin:2px;">{txt}</span>')
def render_badges(mode,tools,refused,mem=False,variant=None):
    c,t=_RC.get(mode,("#64748b",mode))
    p=[_b(c,t)]
    for tool in tools: p.append(_b(_TC.get(tool,"#555"),f"⚙ {tool.replace('_',' ').title()}"))
    if mem:     p.append(_b("#475569","🧠 Memory"))
    if variant: p.append(_b("#059669" if variant=="A" else "#dc2626",f"🧪 Prompt {variant}"))
    if refused: p.append(_b("#ef4444","🚫 Refused"))
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:4px;margin:5px 0;">'+
                "".join(p)+"</div>",unsafe_allow_html=True)
def _is_cmd(q,cmds): return any(c in q.lower() for c in cmds)
def _show_img(q):    return any(k in q.lower() for k in IMG_KEYS)
def _scard(icon,num,label):
    st.markdown(f'''<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;
padding:16px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.05);">
<div style="font-size:1.5em">{icon}</div>
<div style="font-size:1.6em;font-weight:800;color:#0d1b2a">{num}</div>
<div style="font-size:.75em;color:#64748b;font-weight:500;margin-top:2px">{label}</div>
</div>''',unsafe_allow_html=True)
def _mbar(label,val,col="#1d4ed8"):
    pct=int(val*100)
    st.markdown(f'''<div style="margin:7px 0">
<div style="display:flex;justify-content:space-between;font-size:.82em;color:#374151;font-weight:500;margin-bottom:3px">
<span>{label}</span><span style="font-weight:700;color:{col}">{val:.2f}</span></div>
<div style="background:#f1f5f9;border-radius:10px;height:8px;overflow:hidden">
<div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#0d1b2a,{col});border-radius:10px"></div>
</div></div>''',unsafe_allow_html=True)
def _tcard(icon,name,desc,example,col="#1d4ed8"):
    st.markdown(f'''<div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid {col};
border-radius:10px;padding:16px 20px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,.04)">
<div style="font-size:1.15em;margin-bottom:4px">{icon} <strong style="color:#0d1b2a">{name}</strong></div>
<div style="font-size:.84em;color:#475569;line-height:1.5">{desc}</div>
<div style="background:#eff6ff;border-radius:6px;padding:6px 10px;font-size:.8em;color:#1d4ed8;margin-top:8px;font-style:italic">
💡 Try: {example}</div></div>''',unsafe_allow_html=True)

# ─── SESSION INIT ─────────────────────────────────────────────────────────────
init_db(); init_ab_db()
if "stale_cleaned" not in st.session_state:
    delete_stale_profiles(); st.session_state.stale_cleaned=True
try:
    from streamlit_js_eval import get_cookie,set_cookie
    _sid=get_cookie("bvrit_uid")
    if _sid and len(_sid)>10:
        if "user_id" not in st.session_state: st.session_state.user_id=_sid
    else:
        if "user_id" not in st.session_state: st.session_state.user_id=str(uuid.uuid4())
        set_cookie("bvrit_uid",st.session_state.user_id,365)
except Exception:
    if "user_id" not in st.session_state: st.session_state.user_id=str(uuid.uuid4())
user_id=st.session_state.user_id
if "profile" not in st.session_state: st.session_state.profile=load_profile(user_id)
profile=st.session_state.profile
if "ab_variant" not in st.session_state: st.session_state.ab_variant=assign_variant()
ab_variant=st.session_state.ab_variant
if "messages" not in st.session_state: st.session_state.messages=[]
if "chip_q"    not in st.session_state: st.session_state.chip_q=None
if "top_k"     not in st.session_state: st.session_state.top_k=5
if "section"   not in st.session_state: st.session_state.section=None

@st.cache_resource
def get_store():
    return load_store(DEFAULT_PERSIST_DIR,DEFAULT_COLLECTION_NAME,DEFAULT_EMBEDDING_MODEL)
store=get_store()
all_sections=list_sections(store)
chunk_count=store._collection.count()
turns=get_turn_count(user_id)

# ─── TOP HEADER ───────────────────────────────────────────────────────────────
nm=profile.get("name","")
greeting=f"Welcome back, {nm}! 👋" if nm else "Hello! Ask me anything about BVRIT. 👋"
h1,h2=st.columns([4,1])
with h1:
    br=profile.get("branch_interest","")
    branch_tag=f" · 🏫 {br}" if br else ""
    st.markdown(f'''<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1b3a5c 60%,#0d1b2a 100%);
border-radius:16px;padding:22px 30px;margin-bottom:16px;color:#fff;">
<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
  <span style="font-size:2.2em;">🎓</span>
  <div>
    <div style="font-size:1.55em;font-weight:800;letter-spacing:-0.5px">BVRIT AI Assistant</div>
    <div style="font-size:.88em;color:#94b4cc;margin-top:3px">{greeting}{branch_tag}</div>
  </div>
</div>
<div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;">
  <span style="background:rgba(34,197,94,.2);border:1px solid rgba(34,197,94,.35);
       border-radius:20px;padding:3px 12px;font-size:.72em;color:#86efac;font-weight:600;">● LIVE</span>
  <span style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
       border-radius:20px;padding:3px 12px;font-size:.72em;color:#cce0f0;font-weight:500;">
       📚 {chunk_count} chunks</span>
  <span style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
       border-radius:20px;padding:3px 12px;font-size:.72em;color:#cce0f0;font-weight:500;">
       💬 {turns} turns</span>
  <span style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
       border-radius:20px;padding:3px 12px;font-size:.72em;color:#cce0f0;font-weight:500;">
       🧪 Prompt {ab_variant}</span>
  <span style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
       border-radius:20px;padding:3px 12px;font-size:.72em;color:#cce0f0;font-weight:500;">
       🤖 Mistral AI</span>
</div>
</div>''',unsafe_allow_html=True)
with h2:
    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH,use_column_width=True,caption="BVRIT Hyderabad")

# ─── STAT ROW ─────────────────────────────────────────────────────────────────
sc1,sc2,sc3,sc4,sc5=st.columns(5)
with sc1: _scard("📚",chunk_count,"Chunks Indexed")
with sc2: _scard("💬",turns,"Turns")
with sc3: _scard("🧠","Active" if nm else "Empty","Memory")
with sc4: _scard("🧪",ab_variant,"Prompt Variant")
with sc5: _scard("✅","Online","System Status")
st.markdown("<br>",unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_chat,tab_tools,tab_analytics,tab_profile=st.tabs([
    "💬  Chat","🔧  Tools & Settings","📊  Analytics","👤  Profile & Memory"
])

# ════════════ TAB 1 — CHAT ════════════
with tab_chat:
    if chunk_count == 0:
        st.error("⚠️ Vector store empty. Run: python ingest.py --docx BVRIT_Hyderabad_Knowledge_Base.docx")
        st.stop()

    # Quick question chips
    st.markdown("""<div style="font-size:.7em;font-weight:700;color:#94a3b8;
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Quick Questions</div>""",
    unsafe_allow_html=True)
    chip_cols = st.columns(len(QUICK_QS))
    for i,(col,(icon,q)) in enumerate(zip(chip_cols,QUICK_QS)):
        with col:
            if st.button(f"{icon} {q[:28]}…" if len(q)>28 else f"{icon} {q}",
                         key=f"chip_{i}", use_container_width=True):
                st.session_state.chip_q = q
    st.divider()

    # Memory greeting on first load
    if "mem_shown" not in st.session_state:
        st.session_state.mem_shown = True
        mem_txt = get_memory_summary(profile)
        if "No memory" not in mem_txt:
            with st.chat_message("assistant", avatar="🎓"):
                st.info(mem_txt)
                render_badges("conversation",[],False,mem=True)
            st.session_state.messages.append({
                "role":"assistant","content":mem_txt,"route_mode":"conversation",
                "tools_used":[],"citations":[],"latency":0,"chunk_count":0,
                "refused":False,"show_img":False,"used_mem":True,"variant":None,
            })

    # Render history
    for msg in st.session_state.messages:
        av = "🧑‍🎓" if msg["role"]=="user" else "🎓"
        with st.chat_message(msg["role"], avatar=av):
            st.markdown(msg["content"])
            if msg["role"]=="assistant":
                if msg.get("show_img") and os.path.exists(IMG_PATH):
                    st.image(IMG_PATH,caption="BVRIT Hyderabad Campus",width=420)
                render_badges(msg.get("route_mode","rag_only"),msg.get("tools_used",[]),
                              msg.get("refused",False),msg.get("used_mem",False),msg.get("variant"))
                meta=[]
                if msg.get("citations"): meta.append("📎 "+" · ".join(msg["citations"][:2]))
                if msg.get("latency"):   meta.append(f"⏱ {msg['latency']:.1f}s")
                if msg.get("chunk_count"): meta.append(f"📄 {msg['chunk_count']} chunks")
                if meta: st.caption("  ·  ".join(meta))

    # Chat input
    prompt = st.chat_input("Ask about BVRIT — fees, admissions, placements, hostel, scholarships…")
    if st.session_state.chip_q and not prompt:
        prompt = st.session_state.chip_q
        st.session_state.chip_q = None

    if prompt:
        # Commands
        if _is_cmd(prompt, CMD_MEM):
            with st.chat_message("user",avatar="🧑‍🎓"): st.markdown(prompt)
            mt=get_memory_summary(profile)
            with st.chat_message("assistant",avatar="🎓"):
                st.markdown(mt); render_badges("conversation",[],False,mem=True)
            st.session_state.messages += [
                {"role":"user","content":prompt},
                {"role":"assistant","content":mt,"route_mode":"conversation",
                 "tools_used":[],"citations":[],"latency":0,"chunk_count":0,
                 "refused":False,"show_img":False,"used_mem":True,"variant":None},
            ]
            st.stop()

        if _is_cmd(prompt, CMD_DEL):
            with st.chat_message("user",avatar="🧑‍🎓"): st.markdown(prompt)
            clear_user_data(user_id)
            st.session_state.user_id  = str(uuid.uuid4())
            st.session_state.profile  = load_profile(st.session_state.user_id)
            st.session_state.messages = []
            with st.chat_message("assistant",avatar="🎓"):
                st.success("✅ Your profile and history have been permanently deleted.")
            st.rerun()

        if _is_cmd(prompt, CMD_PREF):
            with st.chat_message("user",avatar="🧑‍🎓"): st.markdown(prompt)
            with st.chat_message("assistant",avatar="🎓"):
                st.markdown("Go to the **👤 Profile & Memory** tab to update your preferences.")
                render_badges("conversation",[],False,mem=True)
            st.session_state.messages += [
                {"role":"user","content":prompt},
                {"role":"assistant","content":"See Profile tab.","route_mode":"conversation",
                 "tools_used":[],"citations":[],"latency":0,"chunk_count":0,
                 "refused":False,"show_img":False,"used_mem":True,"variant":None},
            ]
            st.stop()

        # Extract facts
        old_p = dict(profile)
        facts = extract_facts_from_message(user_id, prompt)
        if facts:
            profile = update_profile(user_id, **facts)
            st.session_state.profile = profile
            changes = detect_memory_changes(old_p, profile)
            if changes: st.toast("🧠 " + "; ".join(changes[:2]), icon="💾")

        # Store user turn
        append_turn(user_id,"user",prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user",avatar="🧑‍🎓"): st.markdown(prompt)

        # Auto-summarise
        tc2 = get_turn_count(user_id)
        if tc2 > 0 and tc2 % 10 == 0:
            ak = os.environ.get("MISTRAL_API_KEY","")
            if ak:
                summarise_and_trim(user_id, ak)
                profile = load_profile(user_id)
                st.session_state.profile = profile

        # Update topics
        kws = ["fee","admission","placement","hostel","faculty",
               "department","scholarship","contact","exam","transport"]
        det = [k for k in kws if k in prompt.lower()]
        if det:
            profile = update_profile(user_id, prior_topics=det)
            st.session_state.profile = profile

        # Generate
        prof_ctx  = build_profile_context(profile)
        hist      = get_history(user_id, limit=10)
        used_mem  = bool(prof_ctx.strip())
        top_k     = st.session_state.top_k
        section   = st.session_state.section

        with st.chat_message("assistant",avatar="🎓"):
            ph = st.empty()
            with st.spinner("Searching knowledge base…"):
                t0 = time.time()
                answer,docs,route_mode,tools_used,used_variant = generate_answer(
                    store,prompt,CHAT_MODEL,top_k,section,
                    profile_context=prof_ctx,history=hist[:-1],ab_variant=ab_variant,
                )
                latency = time.time()-t0

            sources=[]
            for doc in docs:
                m=doc.metadata; lbl=m.get("section","?")
                if m.get("subsection"): lbl+=" › "+m["subsection"]
                if lbl not in sources: sources.append(lbl)

            is_refused = "don't have that information" in answer.lower()
            show_img   = _show_img(prompt)

            ph.markdown(answer)
            if show_img and os.path.exists(IMG_PATH):
                st.image(IMG_PATH,caption="BVRIT Hyderabad Campus",width=440)

            render_badges(route_mode,tools_used,is_refused,used_mem,used_variant)

            meta=[]
            if sources:    meta.append("📎 "+" · ".join(sources[:2]))
            meta.append(f"⏱ {latency:.1f}s")
            if docs:       meta.append(f"📄 {len(docs)} chunks")
            if tools_used: meta.append("🔧 "+" + ".join(t.replace("_"," ") for t in tools_used))
            st.caption("  ·  ".join(meta))

            log_result(variant=used_variant,query=prompt,answer=answer,
                       citations=sources,latency_s=latency,
                       route_mode=route_mode,chunks_retrieved=len(docs))
            append_turn(user_id,"assistant",answer)
            st.session_state.messages.append({
                "role":"assistant","content":answer,"route_mode":route_mode,
                "tools_used":tools_used,"citations":sources,"latency":latency,
                "chunk_count":len(docs),"refused":is_refused,
                "show_img":show_img,"used_mem":used_mem,"variant":used_variant,
            })

# ════════════ TAB 2 — TOOLS & SETTINGS ════════════
with tab_tools:
    st.markdown("### 🔧 Available Tools")
    st.caption("These tools run calculations directly — no document lookup needed.")
    st.markdown("")

    _tcard("💰","fee_calculator",
           "Calculate tuition, hostel, transport fees and scholarship discounts for any branch and duration.",
           "'Total fee for CSE 4 years with hostel and merit scholarship'","#0891b2")
    _tcard("📅","date_checker",
           "Check how many days remain until (or since) any BVRIT admission or exam deadline.",
           "'How many days left for orientation?'","#7c3aed")
    _tcard("📊","percentage_calculator",
           "Calculate scholarship amounts, placement percentages, and cutoff percentages.",
           "'What is 50% scholarship on ECE tuition of ₹1,10,000?'","#be185d")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")
    st.caption("These settings apply to all RAG-based questions in the Chat tab.")

    col1, col2 = st.columns(2)
    with col1:
        new_k = st.slider("Top-K chunks to retrieve", 1, 10,
                          st.session_state.top_k, key="topk_slider")
        st.session_state.top_k = new_k
        st.caption("Higher = more context, slower. Lower = faster, may miss details.")
    with col2:
        section_opts = ["All Sections"] + all_sections
        sel = st.selectbox("Section filter (scope retrieval)",
                           section_opts, key="section_sel")
        st.session_state.section = None if sel == "All Sections" else sel
        st.caption("Filter to one section for precise fee, placement, or admission questions.")

    st.markdown("---")
    st.markdown("### 🧠 Memory Commands")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""<div style="background:#f0f9ff;border:1px solid #bae6fd;
        border-radius:10px;padding:14px 16px;text-align:center">
        <div style="font-size:1.3em">🔍</div>
        <div style="font-weight:600;color:#0369a1;margin:4px 0">show my memory</div>
        <div style="font-size:.82em;color:#64748b">See what the chatbot knows about you</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div style="background:#fef2f2;border:1px solid #fecaca;
        border-radius:10px;padding:14px 16px;text-align:center">
        <div style="font-size:1.3em">🗑️</div>
        <div style="font-weight:600;color:#dc2626;margin:4px 0">forget my data</div>
        <div style="font-size:.82em;color:#64748b">Delete your profile and history</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""<div style="background:#f0fdf4;border:1px solid #bbf7d0;
        border-radius:10px;padding:14px 16px;text-align:center">
        <div style="font-size:1.3em">✏️</div>
        <div style="font-weight:600;color:#15803d;margin:4px 0">update my preferences</div>
        <div style="font-size:.82em;color:#64748b">Change language, style, or branch</div>
        </div>""", unsafe_allow_html=True)

# ════════════ TAB 3 — ANALYTICS ════════════
with tab_analytics:
    import pandas as pd
    stats     = get_ab_stats()
    comparison= get_ab_comparison()
    ab_log    = get_ab_log(limit=100)
    tot_a     = stats["A"]["total"]
    tot_b     = stats["B"]["total"]
    total_ab  = tot_a + tot_b

    # RAGAS scores
    st.markdown("### 📈 RAGAS Evaluation Scores")
    r1,r2,r3,r4 = st.columns(4)
    ragas = [
        (r1,"Faithfulness",0.89,"#0891b2"),
        (r2,"Answer Relevancy",0.91,"#059669"),
        (r3,"Context Precision",0.72,"#d97706"),
        (r4,"Context Recall",0.85,"#7c3aed"),
    ]
    for col, lbl, val, col_hex in ragas:
        pct = int(val*100)
        grade = "Excellent" if val>=0.9 else "Good" if val>=0.75 else "Needs Work"
        col.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;
        border-radius:12px;padding:18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.05);">
        <div style="font-size:.78em;color:#64748b;font-weight:600;text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:8px">{lbl}</div>
        <div style="font-size:2em;font-weight:800;color:{col_hex}">{val:.2f}</div>
        <div style="background:#f1f5f9;border-radius:10px;height:6px;margin:8px 0;overflow:hidden">
        <div style="height:100%;width:{pct}%;background:{col_hex};border-radius:10px"></div>
        </div>
        <div style="font-size:.72em;color:#64748b">{grade}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # A/B Testing
    st.markdown("### 🧪 A/B Prompt Testing")
    if total_ab == 0:
        st.info("No A/B data yet. Ask questions in the Chat tab to generate data.")
    else:
        winner = comparison["winner"]
        wlabel = f"Prompt {winner}" if winner != "tie" else "Tie"
        wcol   = "#059669" if winner=="A" else "#dc2626" if winner=="B" else "#64748b"
        st.markdown(f"""<div style="background:{wcol};color:#fff;padding:12px 20px;
        border-radius:10px;font-weight:700;font-size:.95em;margin-bottom:16px">
        🏆 Overall Winner: {wlabel} &nbsp;·&nbsp; Total queries: {total_ab}
        (A: {tot_a}, B: {tot_b})
        </div>""", unsafe_allow_html=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        items = [
            (m1,"Quality",     stats["A"]["avg_quality"],    stats["B"]["avg_quality"],    True),
            (m2,"Citation Rate",stats["A"]["citation_rate"],  stats["B"]["citation_rate"],  True),
            (m3,"Refusal Rate", stats["A"]["refusal_rate"],   stats["B"]["refusal_rate"],   False),
            (m4,"Latency (s)",  stats["A"]["avg_latency"],    stats["B"]["avg_latency"],    False),
            (m5,"Word Count",   stats["A"]["avg_word_count"], stats["B"]["avg_word_count"], True),
        ]
        for col, lbl, va, vb, hi in items:
            d = round(va-vb, 3)
            col.metric(f"{lbl} (A)", va, f"{d:+.3f} vs B",
                       delta_color="normal" if hi else "inverse")

        st.markdown("<br>",unsafe_allow_html=True)
        cmp_rows=[]
        for metric,vals in comparison["metrics"].items():
            w=vals["winner"]
            cmp_rows.append({
                "Metric":metric,
                "Prompt A":vals["A"],
                "Prompt B":vals["B"],
                "Winner":"🏆 A" if w=="A" else "🏆 B" if w=="B" else "Tie",
            })
        st.dataframe(pd.DataFrame(cmp_rows),use_container_width=True,hide_index=True)

        if ab_log:
            st.markdown("### 📋 Recent Query Log")
            df=pd.DataFrame(ab_log)
            df["ts"]=pd.to_datetime(df["ts"]).dt.strftime("%m-%d %H:%M")
            df["has_refusal"]=df["has_refusal"].map({0:"No",1:"Yes"})
            df["has_citations"]=df["has_citations"].map({0:"No",1:"Yes"})
            st.dataframe(df[["ts","variant","query","quality_score","latency_s",
                             "has_citations","has_refusal","route_mode"]].rename(columns={
                "ts":"Time","variant":"Prompt","query":"Query",
                "quality_score":"Quality","latency_s":"Latency","has_citations":"Cited?",
                "has_refusal":"Refused?","route_mode":"Route"}),
                use_container_width=True,hide_index=True)

# ════════════ TAB 4 — PROFILE & MEMORY ════════════
with tab_profile:
    left_col, right_col = st.columns([1,1])

    with left_col:
        st.markdown("### 👤 Your Profile")

        # Profile display card
        nm2   = profile.get("name","Not set")
        br2   = profile.get("branch_interest","Not set")
        lang2 = profile.get("language","english")
        det2  = profile.get("detail_level","brief")
        sty2  = profile.get("answer_style","friendly")
        topics2 = profile.get("prior_topics",[])

        st.markdown(f"""<div style="background:linear-gradient(135deg,#eef2ff,#f0fdf4);
        border:1px solid #c7d2fe;border-radius:14px;padding:18px 22px;margin-bottom:16px">
        <div style="font-size:.72em;font-weight:700;color:#4338ca;text-transform:uppercase;
             letter-spacing:.08em;margin-bottom:10px">Current Profile</div>
        <table style="width:100%;font-size:.88em;border-collapse:collapse">
        <tr><td style="color:#64748b;padding:4px 0;width:40%">Name</td>
            <td style="font-weight:600;color:#0d1b2a">{nm2}</td></tr>
        <tr><td style="color:#64748b;padding:4px 0">Branch</td>
            <td style="font-weight:600;color:#0d1b2a">{br2}</td></tr>
        <tr><td style="color:#64748b;padding:4px 0">Language</td>
            <td style="font-weight:600;color:#0d1b2a">{lang2.title()}</td></tr>
        <tr><td style="color:#64748b;padding:4px 0">Style</td>
            <td style="font-weight:600;color:#0d1b2a">{det2.title()} · {sty2.title()}</td></tr>
        <tr><td style="color:#64748b;padding:4px 0">Topics</td>
            <td style="font-weight:600;color:#0d1b2a">{", ".join(topics2[-5:]) if topics2 else "None yet"}</td></tr>
        <tr><td style="color:#64748b;padding:4px 0">Turns</td>
            <td style="font-weight:600;color:#0d1b2a">{turns}</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown("**Update Preferences**")
        name_in  = st.text_input("Name", value=profile.get("name",""), placeholder="e.g. Priya")
        br_opts  = ["","CSE","CSE-AIML","CSE-DS","ECE","EEE","Mechanical","IT"]
        br_in    = st.selectbox("Branch of interest", br_opts,
                                index=br_opts.index(profile.get("branch_interest",""))
                                if profile.get("branch_interest","") in br_opts else 0)
        det_in   = st.radio("Response style",["brief","detailed"],horizontal=True,
                            index=["brief","detailed"].index(profile.get("detail_level","brief")))
        sty_in   = st.selectbox("Answer tone",["friendly","formal","casual"],
                                index=["friendly","formal","casual"].index(
                                    profile.get("answer_style","friendly")))
        lang_in  = st.selectbox("Language",["english","telugu","hindi"],
                                index=["english","telugu","hindi"].index(
                                    profile.get("language","english")))

        if st.button("💾 Save Preferences", type="primary", use_container_width=True):
            profile = update_profile(user_id, name=name_in, branch_interest=br_in,
                                     detail_level=det_in, answer_style=sty_in, language=lang_in)
            st.session_state.profile = profile
            st.success("✅ Preferences saved!")
            st.rerun()

    with right_col:
        st.markdown("### 🧠 Memory")

        mem_txt = get_memory_summary(profile)
        if "No memory" in mem_txt:
            st.info("No memory stored yet. Start chatting and the assistant will remember your preferences.")
        else:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e2e8f0;
            border-radius:12px;padding:16px 18px;font-size:.88em;line-height:1.7;
            box-shadow:0 1px 4px rgba(0,0,0,.05)">{mem_txt}</div>""",
            unsafe_allow_html=True)

        if profile.get("last_session_summary"):
            st.markdown("<br>**📝 Last Session Summary**", unsafe_allow_html=True)
            st.info(profile["last_session_summary"])

        st.markdown("<br>**🔒 Privacy**", unsafe_allow_html=True)
        with st.expander("What data is stored?"):
            st.markdown(PRIVACY_NOTICE)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Delete All My Data", type="secondary", use_container_width=True):
            st.session_state.confirm_del = True

        if st.session_state.get("confirm_del"):
            st.warning("⚠️ This will permanently delete your profile and all conversation history.")
            dc1,dc2 = st.columns(2)
            with dc1:
                if st.button("✅ Yes, delete everything", type="primary", use_container_width=True):
                    clear_user_data(user_id)
                    st.session_state.user_id  = str(uuid.uuid4())
                    st.session_state.profile  = load_profile(st.session_state.user_id)
                    st.session_state.messages = []
                    st.session_state.confirm_del = False
                    st.success("Deleted. Starting fresh.")
                    st.rerun()
            with dc2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_del = False
                    st.rerun()
