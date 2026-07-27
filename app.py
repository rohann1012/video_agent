import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import shutil
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question
load_dotenv()

if os.path.exists("downloads"):
    shutil.rmtree("downloads")

os.makedirs("downloads", exist_ok=True)

st.set_page_config(page_title="Video Agent  AI Meeting Assistant", page_icon="🎙️", layout="wide")

ACCEPTED_TYPES = ["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a"]

# ---------------------------------------------------------------------------
# Theme: deep studio-console navy, amber "tally light" accent, teal waveform
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0E1320;
    --surface: #161D2E;
    --surface-2: #1D2538;
    --border: #2A3348;
    --amber: #E8A33D;
    --teal: #5EC9C0;
    --text: #F1F0EA;
    --muted: #8A93AA;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp { background: var(--bg); color: var(--text); }

#MainMenu, footer, header { visibility: hidden; }

/* ---- Hero ---- */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #171F33 0%, #101526 100%);
    border: 1px solid var(--border);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.hero-eyebrow .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 10px 2px var(--amber);
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    color: var(--text);
    letter-spacing: -0.01em;
}
.hero p {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 640px;
    margin: 0;
}

/* Waveform divider — signature element */
.waveform {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 22px;
    margin: 0.4rem 0 1.6rem 0;
    opacity: 0.9;
}
.waveform span {
    width: 3px;
    background: linear-gradient(180deg, var(--teal), var(--amber));
    border-radius: 2px;
    display: inline-block;
}

/* ---- Section labels ---- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--teal);
    margin: 0 0 0.5rem 0;
}

/* ---- Cards (native containers with border=True get styled via this) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px 10px 0 0;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ---- Inputs ---- */
.stTextInput input, .stFileUploader, textarea {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2);
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px;
}

/* ---- Buttons ---- */
.stButton button {
    background: var(--amber);
    color: #17130A;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton button:hover {
    box-shadow: 0 0 0 3px rgba(232, 163, 61, 0.25);
    transform: translateY(-1px);
}
.stButton button:disabled {
    background: var(--surface-2);
    color: var(--muted);
}

/* ---- Status / expander ---- */
div[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* ---- Chat ---- */
.stChatMessage {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* Headings inside content */
h2, h3 { font-family: 'Space Grotesk', sans-serif; }

code, pre { font-family: 'IBM Plex Mono', monospace !important; }

/* Hide the hover anchor-link icon Streamlit adds next to headers */
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link {
    display: none !important;
}
</style>
"""


def render_hero():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow"><span class="dot"></span> ON AIR — TRANSCRIPTION READY</div>
            <h1>Video Agent</h1>
            <p>Drop in a YouTube link or upload a recording. Get a clean transcript,
            a summary, action items, decisions, open questions then chat with the
            whole meeting like it's still in the room.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    bars = " ".join(
        f'<span style="height:{h}px"></span>'
        for h in [6, 14, 9, 20, 12, 22, 8, 16, 10, 18, 7, 14, 20, 9, 15, 6, 12, 22, 8, 16]
    )
    st.markdown(f'<div class="waveform">{bars}</div>', unsafe_allow_html=True)


def run_pipeline(source: str) -> dict:
    """Same pipeline as the original main.py, with Streamlit UI feedback
    instead of print statements."""

    status = st.status("🎛️ Rolling tape...", expanded=True)

    status.write("📥 Reading source...")
    chunks = process_input(source)

    status.write("🎙️ Transcribing audio...")
    transcript = transcribe_all(chunks)

    with st.expander("📝 Transcript preview", expanded=False):
        st.code(transcript[:300] + "...", language=None)

    status.write("📌 Generating meeting title...")
    title = generate_title(transcript)

    status.write("📋 Generating summary...")
    summary = summarize(transcript)

    status.write("✅ Extracting action items...")
    action_items = extract_action_items(transcript)

    status.write("🔑 Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    status.write("❓ Extracting open questions...")
    questions = extract_questions(transcript)

    status.write("🧠 Ready for chat when needed")
    status.update(label="✅ Ready", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
    }


def save_upload_to_disk(uploaded_file) -> str:
    """Persist an uploaded file to a temp path and return that path, so it can
    be passed to process_input exactly like a local file path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def main():
    render_hero()

    if "result" not in st.session_state:
        st.session_state.result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "temp_path" not in st.session_state:
        st.session_state.temp_path = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    st.markdown('<p class="section-label">01 · SOURCE</p>', unsafe_allow_html=True)

    with st.container(border=True):
        tab_url, tab_upload = st.tabs(["🔗  YouTube URL", "📁  Upload a video/audio file"])

        source = None

        with tab_url:
            url = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
            if url.strip():
                source = url.strip()

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload a video or audio file",
                type=ACCEPTED_TYPES,
                label_visibility="collapsed",
                help=f"Supported: {', '.join(ACCEPTED_TYPES)}",
            )
            if uploaded_file is not None:
                st.audio(uploaded_file) if uploaded_file.type.startswith("audio") else st.video(uploaded_file)
                source = ("__upload__", uploaded_file)

        col_run, col_status = st.columns([1, 3])
        with col_run:
            run_clicked = st.button("▶  Run", type="primary", disabled=source is None, use_container_width=True)
        with col_status:
            if source is None:
                st.caption("Paste a link or upload a file to get started.")

    if run_clicked and source is not None:
        try:
            if isinstance(source, tuple) and source[0] == "__upload__":
                resolved_source = save_upload_to_disk(source[1])
                st.session_state.temp_path = resolved_source
            else:
                resolved_source = source

            st.session_state.result = run_pipeline(resolved_source)
            st.session_state.chat_history = []  # reset chat for new source
            st.session_state.rag_chain = None
        except Exception as e:
            st.session_state.result = None
            st.error(f"❌ Error: {e}")

    result = st.session_state.result

    if result:
        st.markdown('<p class="section-label">02 · BRIEFING</p>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"### 📌 {result['title']}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📋 Summary**")
                st.write(result["summary"])
                st.markdown("**🔑 Key Decisions**")
                st.write(result["key_decisions"])
            with c2:
                st.markdown("**✅ Action Items**")
                st.write(result["action_items"])
                st.markdown("**❓ Open Questions**")
                st.write(result["open_questions"])

            with st.expander("📝 Full transcript"):
                st.write(result["transcript"])

        st.markdown('<p class="section-label">03 · CHAT WITH THE MEETING</p>', unsafe_allow_html=True)

        with st.container(border=True):
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.write(msg)

            question = st.chat_input("Ask a question about this meeting...")

            if question:
                question = question.strip()
                if question.lower() in ["exit", "quit", "q"]:
                    st.info("👋 Goodbye! (Just close the tab or start a new source above.)")
                elif question:
                    st.session_state.chat_history.append(("user", question))
                    with st.chat_message("user"):
                        st.write(question)

                    try:
                        if st.session_state.rag_chain is None:
                            with st.spinner("🧠 Building knowledge base for chat..."):
                                st.session_state.rag_chain = build_rag_chain(result["transcript"])

                        answer = ask_question(st.session_state.rag_chain, question)
                    except Exception as e:
                        answer = f"❌ Error: {e}"

                    st.session_state.chat_history.append(("assistant", answer))
                    with st.chat_message("assistant"):
                        st.write(answer)
    else:
        st.markdown(
            '<p style="color:var(--muted); text-align:center; margin-top:2rem;">'
            "Your briefing, action items, and chat will appear here once you run a source above."
            "</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()


    