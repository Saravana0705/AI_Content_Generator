import os
import csv
import time
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from typing import List
from src.main_agent.supervisor import Supervisor
from src.sub_agents.text_generator.modules.exporter.exporter import Exporter


# ---------------------------
# Setup
# ---------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Content Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

supervisor = Supervisor()
exporter = Exporter()

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

CSV_FILE = os.path.join(EXPORT_DIR, "sus_scores.csv")


# ---------------------------
# Helpers
# ---------------------------
def save_to_csv(score, grade, adjective, responses):
    file_exists = os.path.isfile(CSV_FILE)
    header = [
        "timestamp", "sus_score", "grade", "adjective",
        "q1_response", "q2_response", "q3_response", "q4_response", "q5_response",
        "q6_response", "q7_response", "q8_response", "q9_response", "q10_response"
    ]
    data_row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), score, grade, adjective, *responses]
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)


def _try_get_image_payload(result: dict):
    """
    Supports multiple return formats:
    - result["image_bytes"] as bytes
    - result["image_b64"] as base64 string
    - result["export_paths"]["files"]["image"] as a path to an image file
    """
    if not isinstance(result, dict):
        return None

    img_bytes = result.get("image_bytes")
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        return bytes(img_bytes)

    img_b64 = result.get("image_b64") or result.get("b64") or result.get("image_base64")
    if isinstance(img_b64, str) and img_b64.strip():
        try:
            return base64.b64decode(img_b64)
        except Exception:
            pass

    export_paths = result.get("export_paths") or result.get("export_result") or {}
    files = export_paths.get("files") or {}
    img_path = files.get("image") or files.get("png") or files.get("jpg")
    if isinstance(img_path, str) and img_path and os.path.exists(img_path):
        try:
            return Path(img_path).read_bytes()
        except Exception:
            return None

    return None


def _img_to_data_uri(path: str):
    if not path or not os.path.exists(path):
        return None

    suffix = Path(path).suffix.lower()
    mime = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


def _render_meta_caption(meta: dict):
    if not meta:
        return

    meta_parts = []
    if meta.get("content_type"):
        meta_parts.append(f"Type: {meta['content_type']}")
    if meta.get("tone"):
        meta_parts.append(f"Tone: {meta['tone']}")
    if meta.get("style"):
        meta_parts.append(f"Style: {meta['style']}")
    if meta.get("size"):
        meta_parts.append(f"Size: {meta['size']}")
    if meta.get("language"):
        meta_parts.append(f"Language: {meta['language']}")
    if meta.get("total_time_sec") is not None:
        meta_parts.append(f"Generated in {meta['total_time_sec']:.2f}s")

    if meta_parts:
        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
        st.caption(" | ".join(meta_parts))


# ---------------------------
# Global CSS
# ---------------------------
st.markdown("""
<style>
/* =========================
   MAIN LAYOUT
========================= */
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

.stApp {
    background: #F8FAFC;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.2rem !important;
}

[data-testid="stToolbar"] {
    right: 0.5rem !important;
    top: 0.2rem !important;
}

.main .block-container > div:first-child {
    margin-top: 0 !important;
}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {
    background: #F1F4F8 !important;
    border-right: 1px solid #E2E8F0 !important;
    overflow: visible !important;
    z-index: 50 !important;
}

section[data-testid="stSidebar"] {
    overflow: visible !important;
}

section[data-testid="stSidebar"] > div {
    overflow: visible !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
    overflow: visible !important;
}

/* Sidebar inner panel feel */
[data-testid="stSidebar"] .block-container {
    padding-top: 0rem !important;
    padding-bottom: 0.5rem !important;
    background: #F1F4F8;
    border-radius: 18px;
}

/* Sticky Controls heading */
.sidebar-controls-title {
    position: sticky;
    top: 0;
    z-index: 5;
    background: #F1F4F8;
    padding-top: 0.05rem;
    padding-bottom: 0.25rem;
    margin-bottom: 0.2rem;
}

/* Sidebar field spacing */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stTextArea {
    margin-bottom: 0.6rem !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    margin-bottom: 0.15rem !important;
    font-size: 0.93rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Sidebar control sizing */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stTextInput input {
    min-height: 2.7rem !important;
    border-radius: 12px !important;
}

/* Sidebar text sizing */
[data-testid="stSidebar"] .stSelectbox span,
[data-testid="stSidebar"] input {
    font-size: 0.95rem !important;
}

/* Premium hover for sidebar controls */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    transition: all 0.18s ease !important;
    border: 1px solid #D8E0EA !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] .stTextInput input:hover,
[data-testid="stSidebar"] .stTextArea textarea:hover {
    border-color: #C5D4E6 !important;
    box-shadow: 0 3px 10px rgba(15,23,42,0.06) !important;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:focus-within,
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {
    border-color: #9DBCE3 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.10) !important;
}

/* =========================
   TYPOGRAPHY
========================= */
h1, h2, h3 {
    margin-top: 0 !important;
    margin-bottom: 0.3rem !important;
    color: #1E293B !important;
}

label,
.stSelectbox label,
.stTextInput label,
.stTextArea label {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* =========================
   INPUTS & BUTTONS
========================= */
.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px !important;
    border: 1px solid #D8E0EA !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: all 0.16s ease !important;
}
            
/* Secondary buttons */
.stButton > button {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #D8E0EA !important;
}

/* Primary form submit button */
.stFormSubmitButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.22) !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}

.stFormSubmitButton > button:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.28) !important;
}

/* =========================
   HEADER
========================= */
.header-divider {
    height: 1px;
    background: linear-gradient(to right, #E5EAF0, transparent);
    margin: 0.55rem 0 1rem 0;
}

/* =========================
   EMPTY STATE
========================= */
.empty-state {
    background: #FFFFFF;
    border: 1px solid #E8EDF3;
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 4px 214px rgba(15, 23, 42, 0.04);
    margin: 0.7rem 0 1rem 0;
}

.empty-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.2rem;
}

.empty-subtitle {
    font-size: 0.9rem;
    color: #64748B;
}

/* =========================
   COMPOSER
========================= */
.sticky-composer {
    position: sticky;
    bottom: 0;
    z-index: 20;
    background: linear-gradient(to top, #F8FAFC 70%, rgba(248,250,252,0));
    padding-top: 1rem;
    padding-bottom: 0.35rem;
}

.composer-wrap {
    background: #FFFFFF;
    border: 1px solid #E5EAF0;
    border-radius: 18px;
    padding: 0.35rem 0.35rem 0.15rem 0.35rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    margin-top: 0.35rem;
}

/* =========================
   CHAT UI
========================= */
[data-testid="stChatMessageAvatar"] img {
    width: 34px !important;
    height: 34px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    border: 1px solid #E2E8F0 !important;
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.9rem !important;
    background: #F8FAFC;
    border: 1px solid #EDF2F7;
    box-shadow: none;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}

[data-testid="stChatMessage"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.chat-user-wrap [data-testid="stChatMessage"] {
    background: #EEF4FF !important;
    border: 1px solid #D8E7FF !important;
}

.chat-assistant-wrap [data-testid="stChatMessage"] {
    background: #F7F8FA !important;
    border: 1px solid #ECEFF3 !important;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    margin: 0 0 0.05rem 0 !important;
    line-height: 1.24 !important;
    font-size: 0.97rem !important;
    color: #1F2937 !important;
    word-break: break-word !important;
}

[data-testid="stChatMessageContent"] h1 {
    font-size: 1.2rem !important;
    line-height: 1.18 !important;
    margin-top: 0.15rem !important;
    margin-bottom: 0.2rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}

[data-testid="stChatMessageContent"] h2 {
    font-size: 1.05rem !important;
    line-height: 1.18 !important;
    margin-top: 0.22rem !important;
    margin-bottom: 0.12rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}

[data-testid="stChatMessageContent"] h3 {
    font-size: 1rem !important;
    line-height: 1.18 !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.1rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
}

[data-testid="stChatMessageContent"] code {
    font-family: inherit !important;
    font-size: inherit !important;
    background: transparent !important;
    color: inherit !important;
    padding: 0 !important;
}

[data-testid="stChatMessageContent"] pre {
    background: #F8FAFC !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    border: 1px solid #E5EAF0 !important;
}

[data-testid="stMarkdownContainer"] > * {
    margin-bottom: 0.08rem !important;
}

/* =========================
   CAPTIONS / MEDIA
========================= */
[data-testid="stCaptionContainer"] {
    color: #64748B !important;
}

[data-testid="stImage"] img {
    border-radius: 16px;
}

/* =========================
   UTILITIES
========================= */
h1 a, h2 a, h3 a {
    display: none !important;
}

hr {
    margin: 0.8rem 0 !important;
    border-color: #E5EAF0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "show_sus_form" not in st.session_state:
    st.session_state.show_sus_form = False


# ---------------------------
# Model Options (UI)
# ---------------------------
TEXT_MODEL_OPTIONS = {
    "OpenAI GPT-4o": ("openai", "gpt-4o"),
    "Google Gemini Pro (Coming soon)": ("google", "gemini-pro"),
}

IMAGE_MODEL_OPTIONS = {
    "OpenAI GPT Image 1 Mini": ("openai", "gpt-image-1-mini"),
    "Gemini Flash (Coming soon)": ("google", "gemini-flash"),
}

DISPLAY_MODEL_NAMES = {
    "gpt-4o": "OpenAI GPT-4o",
    "gemini-pro": "Google Gemini Pro",
    "gpt-image-1-mini": "OpenAI GPT Image 1 Mini",
    "gemini-flash": "Google Gemini Flash",
}

_default_provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
_default_model_openai = os.getenv("OPENAI_MODEL", "gpt-4o")
_default_model_groq = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_default_image_model = os.getenv("IMAGE_MODEL", "gpt-image-1-mini")


def _default_choice_key(model_options, provider=None, model=None):
    if provider and model:
        for k, (p, m) in model_options.items():
            if p == provider and m == model:
                return k

    if provider:
        for k, (p, _) in model_options.items():
            if p == provider:
                return k

    if model:
        for k, (_, m) in model_options.items():
            if m == model:
                return k

    return list(model_options.keys())[0]


# ---------------------------
# Paths
# ---------------------------
APP_ICON_PATH = "assets/ai-content-generator-icon.png"

USER_AVATAR_PATH = "assets/user-icon.png"
BOT_AVATAR_PATH = "assets/bot-icon.png"

user_avatar = USER_AVATAR_PATH if os.path.exists(USER_AVATAR_PATH) else "🧑"
bot_avatar = BOT_AVATAR_PATH if os.path.exists(BOT_AVATAR_PATH) else "🤖"
app_icon_data_uri = _img_to_data_uri(APP_ICON_PATH)


# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.markdown(
        "<div class='sidebar-controls-title' style='font-size:1.32rem; font-weight:700; color:#1E293B;'>⚙️ Controls</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='font-size:0.76rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.04em; margin:0.2rem 0 0.3rem 0;'>Generation</div>", unsafe_allow_html=True)

    mode = st.selectbox(
        "Generator Mode",
        options=["Text Generator", "Image Generator"],
        index=0,
    )
    subagent_type = "text_generator" if mode == "Text Generator" else "image_generator"

    model_group_gap = "<div style='height:0.1rem;'></div>"

    st.markdown(model_group_gap, unsafe_allow_html=True)
    model_choice = None

    if subagent_type == "text_generator":
        current_model_options = TEXT_MODEL_OPTIONS

        default_text_model = (
            _default_model_openai if _default_provider == "openai"
            else _default_model_groq if _default_provider == "groq"
            else _default_model_openai
        )

        default_key = _default_choice_key(
            current_model_options,
            provider=_default_provider,
            model=default_text_model,
        )
    else:
        current_model_options = IMAGE_MODEL_OPTIONS
        default_key = _default_choice_key(
            current_model_options,
            provider=_default_provider,
            model=_default_image_model,
        )

    default_index = list(current_model_options.keys()).index(default_key)

    model_choice = st.selectbox(
        "Model",
        options=list(current_model_options.keys()),
        index=default_index,
    )

    selected_provider, selected_model = current_model_options[model_choice]
    display_model_name = DISPLAY_MODEL_NAMES.get(selected_model, model_choice)

    user_friendly_content_type = None
    keywords_input = ""

    if subagent_type == "text_generator":
        st.markdown("<div style='font-size:0.76rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.04em; margin:0.35rem 0 0.3rem 0;'>Content settings</div>", unsafe_allow_html=True)

        user_friendly_content_type = st.selectbox(
            "Content Type",
            options=[
                "Blog Article",
                "News Article",
                "Email Copy",
                "LinkedIn & Facebook Post",
                "TikTok Caption",
                "YouTube Video Description",
                "Twitter Tweet",
                "Webinar Script",
                "Podcast Transcript",
                "FAQ Section",
            ],
            index=0,
        )

        tone = st.selectbox(
            "Tone",
            options=["Neutral", "Friendly", "Professional", "Persuasive", "Casual"],
            index=0,
        )

        st.markdown("<div style='height:0.05rem;'></div>", unsafe_allow_html=True)
        text_language = st.selectbox(
            "Language",
            options=["English", "German"],
            index=0,
        )

        keywords_input = st.text_input("Keywords (comma-separated)", value="")

        text_language_code = "de" if text_language == "German" else "en"

        image_style = None
        image_size = None
        image_language = None
        image_language_code = "en"

    else:
        st.markdown("<div style='font-size:0.76rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.04em; margin:0.35rem 0 0.3rem 0;'>Visual settings</div>", unsafe_allow_html=True)

        image_style = st.selectbox(
            "Style",
            options=["Photorealistic", "Anime", "3D", "Illustration"],
            index=0,
        )

        image_size = st.selectbox(
            "Size",
            options=["1024x1024", "1024x1536", "1536x1024"],
            index=0,
        )

        image_language = st.selectbox(
            "Language",
            options=["English", "German"],
            index=0,
        )

        image_language_code = "de" if image_language == "German" else "en"

        tone = image_style
        user_friendly_content_type = "Image"
        keywords_input = ""


# ---------------------------
# Main Header
# ---------------------------
friendly_mode = "Text Generator" if mode == "Text Generator" else "Image Generator"
subtitle_text = "Create polished text and visuals with AI assistance"

icon_html = ""
if app_icon_data_uri:
    icon_html = f'<img src="{app_icon_data_uri}" style="width:48px; height:48px; object-fit:contain; margin-top:2px;" />'

header_html = f"""
<div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:0.45rem;">
{icon_html}
<div style="display:flex; flex-direction:column; justify-content:flex-start;">
<div style="font-size:2.08rem; font-weight:800; color:#1E293B; line-height:1.08; margin:0;">
AI Content Generator
</div>
<div style="font-size:0.94rem; color:#64748B; margin-top:0.10rem;">
{subtitle_text}
</div>
<div style="font-size:0.88rem; color:#94A3B8; margin-top:0.24rem;">
Mode: {friendly_mode} | Model: {display_model_name}
</div>
</div>
</div>
<div class="header-divider"></div>
"""

st.markdown(header_html, unsafe_allow_html=True)

# ---------------------------
# Conversation History
# ---------------------------
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">Start creating</div>
        <div class="empty-subtitle">
            Generate articles or images from a simple prompt.
        </div>
    </div>
    """, unsafe_allow_html=True)

for i, chat in enumerate(st.session_state.history):
    st.markdown('<div class="chat-user-wrap">', unsafe_allow_html=True)
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(
            "<div style='font-size:0.78rem; font-weight:700; color:#2563EB; margin-bottom:0.18rem;'>You</div>",
            unsafe_allow_html=True
        )
        st.markdown(chat.get("prompt", ""))
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if chat.get("type") == "image":
        st.markdown('<div class="chat-assistant-wrap">', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(
                "<div style='font-size:0.78rem; font-weight:700; color:#475569; margin-bottom:0.18rem;'>Assistant</div>",
                unsafe_allow_html=True
            )
            st.markdown(chat.get("response_text", "Image generated."))
            img_payload = chat.get("image_bytes")
            if isinstance(img_payload, (bytes, bytearray)) and len(img_payload) > 0:
                st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
                st.image(img_payload, use_container_width=True)
            _render_meta_caption(chat.get("meta", {}))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-assistant-wrap">', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(
                "<div style='font-size:0.78rem; font-weight:700; color:#475569; margin-bottom:0.18rem;'>Assistant</div>",
                unsafe_allow_html=True
            )
            st.markdown(chat.get("response", ""))
            _render_meta_caption(chat.get("meta", {}))
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Composer / Input Form
# ---------------------------
st.markdown('<div class="sticky-composer">', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:1.15rem; font-weight:700; color:#1E293B; margin:0.45rem 0 0.55rem 0;'>What would you like to create?</div>",
    unsafe_allow_html=True
)
st.markdown('<div class="composer-wrap">', unsafe_allow_html=True)

with st.form("prompt_form", clear_on_submit=True):
    user_input = st.text_area(
        "Enter your prompt",
        height=110,
        placeholder="Describe the content or image you want to generate..."
    )
    submit_pressed = st.form_submit_button("Generate", type = "primary")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Generation Handling
# ---------------------------
if submit_pressed and user_input:
    if subagent_type == "text_generator":
        os.environ["TEXT_PROVIDER"] = selected_provider

        if selected_provider == "openai":
            os.environ["OPENAI_TEXT_MODEL"] = selected_model
        elif selected_provider == "groq":
            os.environ["GROQ_TEXT_MODEL"] = selected_model
    else:
        os.environ["IMAGE_PROVIDER"] = selected_provider

        if selected_provider == "openai":
            os.environ["OPENAI_IMAGE_MODEL"] = selected_model
        elif selected_provider == "stability":
            os.environ["STABILITY_IMAGE_MODEL"] = selected_model
        elif selected_provider == "freepik":
            os.environ["FREEPIK_IMAGE_MODEL"] = selected_model

    if selected_provider == "openai":
        os.environ["OPENAI_MODEL"] = selected_model
    elif selected_provider == "groq":
        os.environ["GROQ_MODEL"] = selected_model

    with st.spinner("Generating... Please wait"):
        if subagent_type == "text_generator":
            CONTENT_TYPE_MAPPING = {
                "Blog Article": "blog_article",
                "News Article": "news_article",
                "Email Copy": "email_copy",
                "LinkedIn & Facebook Post": "social_post",
                "TikTok Caption": "short_form_social",
                "YouTube Video Description": "video_description",
                "Twitter Tweet": "tweet",
                "Webinar Script": "script",
                "Podcast Transcript": "script",
                "FAQ Section": "faq_section",
            }

            content_type_key = CONTENT_TYPE_MAPPING[user_friendly_content_type]
            keywords_list = [k.strip() for k in keywords_input.split(",")] if keywords_input else []

            t0 = time.perf_counter()
            result = supervisor.coordinate_workflow(
                user_input,
                subagent_type="text_generator",
                content_type=content_type_key,
                tone=tone,
                keywords=keywords_list,
                model_name=selected_model,
                language=text_language_code,
            )
            total_time_sec = time.perf_counter() - t0

            st.session_state.last_result = {**(result or {}), "subagent_type": "text_generator"}

            final_text = result.get("optimized_text") or result.get("final_text") or "Not available."

            st.session_state.history.append({
                "prompt": user_input,
                "response": final_text,
                "type": "text",
                "meta": {
                    "content_type": user_friendly_content_type,
                    "tone": tone,
                    "language": text_language,
                    "total_time_sec": total_time_sec,
                }
            })
            st.rerun()

        else:
            t0 = time.perf_counter()
            result = supervisor.coordinate_workflow(
                user_input,
                subagent_type="image_generator",
                content_type="image",
                tone=image_style,
                keywords=[],
                model_name=selected_model,
                image_size=image_size,
                language=image_language_code,
            )
            total_time_sec = time.perf_counter() - t0

            st.session_state.last_result = {**(result or {}), "subagent_type": "image_generator"}

            img_bytes = _try_get_image_payload(result or {})
            if not img_bytes:
                st.session_state.history.append({
                    "prompt": user_input,
                    "response": "Image generation failed (no image payload returned).",
                    "type": "text",
                    "meta": {
                        "content_type": "Image",
                        "tone": image_style,
                        "language": image_language,
                        "total_time_sec": total_time_sec,
                    }
                })
            else:
                st.session_state.history.append({
                    "prompt": user_input,
                    "response_text": "Image generated.",
                    "type": "image",
                    "image_bytes": img_bytes,
                    "meta": {
                        "style": image_style,
                        "size": image_size,
                        "language": image_language,
                        "total_time_sec": total_time_sec,
                    }
                })
            st.rerun()

# ---------------------------
# Publish UI (Text only)
# ---------------------------
last = st.session_state.last_result or {}
last_type = (last.get("subagent_type") or "text_generator").strip().lower()

review = (last.get("review_result") or last.get("review") or {})
approved = bool(review.get("approved", False))

if last_type == "text_generator" and approved:
    with st.expander("📤 Publish options", expanded=False):
        platform = st.selectbox(
            "Platform to publish",
            options=[
                "DEV.to",
                "Medium (Coming soon)",
                "LinkedIn (Coming soon)",
            ],
            index=0,
            key="publish_platform_select",
        )

        colA, colB = st.columns([1, 4])

        with colA:
            publish_clicked = st.button("Publish now", type="primary")

        if publish_clicked:
            export_result = last.get("export_result", {}) or last.get("export_paths", {}) or {}
            files = export_result.get("files", {}) or {}
            md_path = files.get("md")

            if not md_path or not os.path.exists(md_path):
                st.error("Could not find exported Markdown file to publish. (MD path missing)")
            else:
                with open(md_path, "r", encoding="utf-8") as f:
                    body_md = f.read()

                title = "Generated Article"
                for line in body_md.splitlines():
                    if line.strip().startswith("# "):
                        title = line.strip().lstrip("#").strip()
                        break
                if title == "Generated Article":
                    title = (last.get("original_topic") or "").strip() or "Generated Article"

                default_pub = os.getenv("DEVTO_DEFAULT_PUBLISHED", "false").strip().lower() in ("1", "true", "yes")
                published_flag = default_pub

                tags = []
                if "keywords_list" in locals():
                    tags = keywords_list[:5]

                pub_res = exporter.publish_to_devto(
                    title=title,
                    body_markdown=body_md,
                    published=published_flag,
                    tags=tags,
                )

                if pub_res.get("ok"):
                    st.success("Published successfully to DEV.to.")
                else:
                    st.error(f"DEV.to publish failed: {pub_res.get('message')}")
                    details = pub_res.get("details")
                    if details:
                        st.code(details)

# ---------------------------
# SUS Survey
# ---------------------------
with st.expander("📝 Usability survey", expanded=False):
    st.caption("Evaluate usability of the system after testing the interface.")

    if st.button("Start survey"):
        st.session_state.show_sus_form = True

    if st.session_state.show_sus_form:
        st.markdown(
            "<div style='font-size:1.2rem; font-weight:700; color:#1E293B; margin:0.4rem 0 0.25rem 0;'>System Usability Scale (SUS) Survey</div>",
            unsafe_allow_html=True
        )
        st.caption("Rate each statement from 1 to 5, where 1 = Strongly Disagree and 5 = Strongly Agree.")

        questions = [
            "I think that I would like to use this system frequently.",
            "I found the system unnecessarily complex.",
            "I thought the system was easy to use.",
            "I think that I would need the support of a technical person to be able to use this system.",
            "I found the various functions in this system were well integrated.",
            "I thought there was too much inconsistency in this system.",
            "I would imagine that most people would learn to use this system very quickly.",
            "I found the system very cumbersome to use.",
            "I felt very confident using the system.",
            "I needed to learn a lot of things before I could get going with this system.",
        ]

        responses = []
        for i, q in enumerate(questions, start=1):
            resp = st.radio(
                f"Q{i}. {q}",
                options=[1, 2, 3, 4, 5],
                index=2,
                horizontal=True,
                key=f"sus_q{i}",
            )
            responses.append(resp)

        if st.button("Submit survey", type="primary"):
            odd = [responses[i] - 1 for i in range(0, 10, 2)]
            even = [5 - responses[i] for i in range(1, 10, 2)]
            sus_score = (sum(odd) + sum(even)) * 2.5

            if sus_score >= 80:
                grade = "A"
                adjective = "Excellent"
            elif sus_score >= 70:
                grade = "B"
                adjective = "Good"
            elif sus_score >= 60:
                grade = "C"
                adjective = "OK"
            elif sus_score >= 50:
                grade = "D"
                adjective = "Poor"
            else:
                grade = "F"
                adjective = "Awful"

            save_to_csv(sus_score, grade, adjective, responses)
            st.success(f"Thanks! SUS Score = {sus_score:.1f} ({grade} / {adjective})")
            st.session_state.show_sus_form = False