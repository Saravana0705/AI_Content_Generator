import streamlit as st
from dotenv import load_dotenv
import os
import csv  
from datetime import datetime  
from openai import OpenAI
from src.sub_agents.text_generator.modules.generator.content_generator import Generator
from src.sub_agents.text_generator.modules.content_retrieval.llamaindex_retriever import Retriever
from src.sub_agents.text_generator.modules.optimizer.optimizer import Optimizer
from src.sub_agents.text_generator.modules.human_review.reviewer import Reviewer
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from streamlit_chat import message

# --- Page Configuration & CSS ---
st.set_page_config(page_title="AI Content Generator", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* General App Styling */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to right top, #d1e4f6, #e1eaf9, #eef1fb, #f8f8fc, #ffffff);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* Main Title */
    h1 {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        color: #1A2E44;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        padding-top: 0.5rem;
    }

    /* Landing Page Container - Glassmorphism Effect */
    .landing-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        max-width: 900px;
        margin: 0.5rem auto;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Tagline */
    .tagline {
        font-family: 'Poppins', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 1.5rem;
    }

    /* Feature Section & Cards */
    .feature-section {
        display: flex;
        justify-content: space-around;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.25rem;
        border-radius: 15px;
        width: 250px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        display: inline-block;
        line-height: 55px;
        width: 55px;
        height: 55px;
        border-radius: 50%;
        background-color: #eaf2ff;
    }
    .feature-card h3 {
        font-family: 'Poppins', sans-serif;
        color: #1A2E44;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .feature-card p {
        font-family: 'Poppins', sans-serif;
        color: #5a6a7b;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* CTA Button */
    .cta-button {
        background-color: #000000;
        color: #ffffff !important;
        text-decoration: none;
        padding: 0.9rem 2.5rem;
        border-radius: 50px;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        display: inline-block;
        margin-top: 1rem;
    }
    .cta-button:hover {
        color: #ffffff !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        background-color: #333333;
    }
    
    /* General Button Styling */
    .stButton>button {
        border: none;
        background-color: transparent;
    }
    
    /* Chat Input Styling */
    .chat-input-container {
        display: flex;
        align-items: center;
        width: 100%;
    }

    .chat-input-container textarea {
        flex: 1;
        border-radius: 10px;
        padding: 12px;
        min-height: 60px;
        width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stFormSubmitButton"] {
        margin: 0 !important;
    }

    [data-testid="stFormSubmitButton"] button {
        position: absolute;
        right: 10px;  
        top: 10%;  
        transform: translateY(-150%);  /* centers vertically */
        border: none;
        border-radius: 50%;
        width: 42px;
        height: 42px;
        font-size: 18px;
        font-weight: bold;
        color: #111111;
        cursor: pointer;
        transition: all 0.3s ease;
        background-color: #d1d5db;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #d3d3d3;
        color: #111111;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .feature-section {
            flex-direction: column;
            align-items: center;
        }
        .feature-card {
            width: 80%;
            max-width: 300px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Backend Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

# --- NEW: CSV Saving Functionality ---
CSV_FILE = 'sus_scores.csv'

def save_to_csv(score, grade, adjective, responses):
    """Saves the survey results to a local CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    header = [
        'timestamp', 'sus_score', 'grade', 'adjective',
        'q1_response', 'q2_response', 'q3_response', 'q4_response', 'q5_response',
        'q6_response', 'q7_response', 'q8_response', 'q9_response', 'q10_response'
    ]
    data_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        score, grade, adjective, *responses
    ]
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)
# --- END NEW SECTION ---

@dataclass
class AgentState:
    input_text: str = ""
    original_topic: str = ""
    content_type: str = "blog_article"
    retrieved_data: str = ""
    generated_text: str = ""
    optimized_text: str = ""
    notes: str = ""
    score: float = 0.0
    tone: str = "neutral"
    keywords: list = field(default_factory=list)
    review_result: dict = field(default_factory=dict)

def retrieve_content(state: AgentState) -> AgentState:
    retriever = Retriever()
    state.retrieved_data = retriever.retrieve(state.input_text) or "No data retrieved"
    return state

def generate_content(state: AgentState) -> AgentState:
    generator = Generator()
    state.generated_text = generator.generate(state.input_text) or "No content generated"
    return state

def optimize_content(state: AgentState) -> AgentState:
    optimizer = Optimizer()
    state.optimized_text, state.score, state.notes = optimizer.optimize(
        text=state.generated_text,
        original_topic=state.original_topic,
        content_type=state.content_type,
        tone=state.tone,
        keywords=state.keywords
    )
    return state

def review_content(state: AgentState) -> AgentState:
    reviewer = Reviewer(threshold=60.0)
    state.review_result = reviewer.review(
        content=state.optimized_text,
        score=state.score,
        notes=state.notes,
    )
    return state


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_content)
workflow.add_node("generate", generate_content)
workflow.add_node("optimize", optimize_content)
workflow.add_node("review", review_content) 

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "optimize")
workflow.add_edge("optimize", "review")  
workflow.add_edge("review", END)  

app_graph = workflow.compile()

# --- Sidebar --
with st.sidebar:
    st.title("💡 Content Controls")

    CONTENT_CATEGORY_MAPPING = {"Text": "text", "Images": "images", "Videos": "videos"}
    category_options = list(CONTENT_CATEGORY_MAPPING.keys())
    category_index = st.session_state.get('category_index', None)
    user_friendly_content_category = st.selectbox(
        "Content Category",
        options=category_options,
        placeholder="Select a category...",
        index=category_index
    )

    # Initialize dependent variables
    user_friendly_content_type = None
    tone = None
    keywords_input = ""
    content_type_options = []
    tone_options = []

    if user_friendly_content_category == "Text":
        CONTENT_TYPE_MAPPING = {
            "Blog Article": "blog_article", "News Article": "news_article", "Email Copy": "email_copy",
            "LinkedIn & Facebook Post": "social_post", "TikTok Caption": "short_form_social",
            "YouTube Video Description": "video_description", "Twitter Tweet": "tweet",
            "Webinar Script": "script", "Podcast Transcript": "script", "FAQ Section": "faq_section"
        }
        content_type_options = list(CONTENT_TYPE_MAPPING.keys())
        content_type_index = st.session_state.get('content_type_index', None)
        user_friendly_content_type = st.selectbox(
            "Select Content Type",
            options=content_type_options,
            placeholder="Select a type...",
            index=content_type_index
        )

        tone_options = ["neutral", "formal", "informal", "positive", "persuasive"]
        tone_index = st.session_state.get('tone_index', None)
        tone = st.selectbox(
            "Select Tone",
            options=tone_options,
            placeholder="Select a tone...",
            index=tone_index
        )

        keywords_input = st.text_input("SEO Keywords (comma-separated)", value=st.session_state.get('keywords_str', ''))

    elif user_friendly_content_category in ["Images", "Videos"]:
        st.info(f"{user_friendly_content_category} generation is coming soon!")

# --- Main Page ---
st.title("AI Content Generator 🧠")

# Initialize page state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'chat'

# --- Page Navigation Logic ---
if st.session_state.page == 'survey':
    # Survey Page
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
            background-image: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("🔍 System Usability Scale (SUS) Survey")
    st.info("Rate your experience with this AI Content Generator. Answer all 10 questions (1=Strongly Disagree, 5=Strongly Agree). Takes ~2 minutes!")

    # Collect responses
    responses = []
    for i, q in enumerate([
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need the support of a technical person to be able to use this system.",
        "I found the various functions in this system were well integrated.",
        "I thought there was too much inconsistency in this system.",
        "I would imagine that most people would learn to use this system very quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot of things before I could get going with this system."
    ], 1):
        st.markdown(f"{i}. {q}")
        response = st.radio(
            label=f"Response for question {i}",
            options=[1, 2, 3, 4, 5],
            index=None,
            key=f"sus_{i}",
            horizontal=True,
            label_visibility="collapsed",
            help="1=Strongly Disagree | 5=Strongly Agree"
        )
        responses.append(response)

    # Create two columns for buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        # Back to Chat button
        if st.button("⬅ Back to Chat", key="back_to_chat"):
            st.session_state.page = 'chat'
            st.rerun()

    with col2:
        # Submit Survey button
        submitted = st.button("Submit Survey", key="submit_sus")

    if submitted:
        if None in responses:
            st.error("Please answer all 10 questions before submitting.")
        else:
            # Compute SUS Score
            contributions = []
            for i, resp in enumerate(responses, 1):
                if i % 2 == 1:  # Odd questions
                    contributions.append(resp - 1)
                else:  # Even questions
                    contributions.append(5 - resp)
            total = sum(contributions) * 2.5
            sus_score = round(total, 1)

            # Grade + Adjective
            if sus_score >= 80.3: grade, adjective = "A", "Best Imaginable"
            elif sus_score >= 68: grade, adjective = "B", "Excellent"
            elif sus_score >= 60.7: grade, adjective = "C", "Good"
            elif sus_score >= 52.1: grade, adjective = "D", "OK"
            else: grade, adjective = "F", "Poor"

            # Display Results
            st.success(f"*Your SUS Score: {sus_score}/100* (Grade: {grade} | Adjective: {adjective})")

            # --- MODIFIED: Call the save function ---
            save_to_csv(sus_score, grade, adjective, responses)
            # --- END MODIFICATION ---

            # Store in session
            if 'sus_scores' not in st.session_state:
                st.session_state.sus_scores = []
            st.session_state.sus_scores.append(sus_score)
            if len(st.session_state.sus_scores) > 0:
                avg_score = sum(st.session_state.sus_scores) / len(st.session_state.sus_scores)
                #st.metric("Your Average SUS Score (this session)", f"{avg_score:.1f}/100")

    # ---FIXED CSS FOR BUTTONS ---
    st.markdown("""
        <style>
        /* Target the "Submit Survey" button (in the second column) */
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
            background-color: #000000 !important;
            color: #ffffff !important;
            text-decoration: none;
            padding: 1rem 2.8rem;
            border-radius: 50px;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.2rem;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            width: 100%;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
            background-color: #333333 !important;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        /* Target the "Back to Chat" button (in the first column) */
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
            background-color: #d1d5db !important;
            color: #111111 !important;
            text-decoration: none;
            padding: 1rem 2.8rem;
            border-radius: 50px;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.2rem;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            width: 100%;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
            background-color: #b0b7c3 !important;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

else:
    # Landing or Chat Page
    if not user_friendly_content_category or user_friendly_content_category != "Text":
        # Landing Page
        st.markdown(
            """
            <div class="landing-container">
                <h2 class="tagline">Empower Your Content with AI Precision</h2>
                <div class="feature-section">
                    <div class="feature-card">
                        <h3>📥 Smart Retrieval</h3>
                        <p>Fetch relevant data to enrich your content effortlessly.</p>
                    </div>
                    <div class="feature-card">
                        <h3>✍ AI Generation</h3>
                        <p>Create high-quality text tailored to your needs.</p>
                    </div>
                    <div class="feature-card">
                        <h3>⚡ Optimization</h3>
                        <p>Enhance readability, SEO, and engagement.</p>
                    </div>
                </div>
                <a href="#" class="cta-button" onclick="document.querySelector('[data-testid=sidebar]').style.display='block'; return false;">Get Started</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Chat Page
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background: #ffffff !important;
                background-image: none !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Session State Init
        if 'history' not in st.session_state:
            st.session_state.history = [{
                "prompt": "Hi!",
                "response": "Hello! What content can I create for you today?"
            }]
        if 'sus_scores' not in st.session_state:
            st.session_state.sus_scores = []

        # Display Chat History
        for chat in st.session_state.history:
            message(chat['prompt'], is_user=True, key=f"user_{st.session_state.history.index(chat)}")
            message(chat['response'], key=f"ai_{st.session_state.history.index(chat)}", allow_html=True)

        # User Input Form & Processing Logic
        with st.form(key="prompt_form", clear_on_submit=True):
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            user_input = st.text_area(
                label="prompt_input",
                label_visibility="collapsed",
                placeholder="Type your topic or prompt here...",
                height=150,
                key="prompt_input"
            )
            submit_pressed = st.form_submit_button("▶", help="Send")
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_pressed and user_input:
                if not user_friendly_content_type or not tone:
                    st.warning("⚠ Please make a selection for Content Type and Tone in the sidebar.")
                else:
                    with st.spinner("🤖 Agent is thinking..."):
                        content_type_key = CONTENT_TYPE_MAPPING[user_friendly_content_type]
                        detailed_prompt = (
                            f"You are an expert content writer. Create content about: '{user_input}'.\n"
                            f"The content type must be a '{user_friendly_content_type}'.\n"
                            "Follow all structural and stylistic rules appropriate for this content type."
                        )
                        keywords_list = [k.strip() for k in keywords_input.split(',')] if keywords_input else []
                        initial_state = AgentState(
                            input_text=detailed_prompt, original_topic=user_input, content_type=content_type_key,
                            tone=tone, keywords=keywords_list
                        )
                        
                        result = app_graph.invoke(initial_state)

                        # --- Build rich response with review info ---
                        review = result.get("review_result", {}) or {}
                        approved = review.get("approved")
                        decision = review.get("decision", "N/A")
                        quality_band = review.get("quality_band", "N/A")
                        comments = review.get("comments", [])

                        comments_md = ""
                        if comments:
                            comments_md = "\n".join(f"- {c}" for c in comments)

                        score_value = result.get("score", 0)

                        ai_response = (
                            f"### ✨ Optimized & Final Content\n"
                            f"{result.get('optimized_text', 'Not available.')}\n\n"
                            f"---\n\n"
                            f"### 🧪 Optimization Score\n"
                            f"**{score_value:.1f} / 100** (Quality: `{quality_band}`)\n\n"
                            f"### 📝 Review Decision\n"
                            f"**{'✅ Approved' if approved else '⚠ Needs Revision'}** "
                            f"(decision: `{decision}`)\n\n"
                            f"{comments_md}\n\n"
                            f"---\n\n"
                            f"### 📌 Analysis Notes\n"
                            f"{result.get('notes', 'N/A')}"
                        )

                        st.session_state.history.append({
                            "prompt": user_input,
                            "response": ai_response
                        })
                        st.rerun()


        # Example Prompt
        if len(st.session_state.history) == 1:
            st.caption(
                "*Example Prompt:* Create a detailed blog post about the benefits of AI in healthcare, written in a positive tone, targeting a general audience. "
                "Include SEO keywords: AI healthcare, machine learning diagnostics, telemedicine. "
                "Add a brief introduction, 3 key benefits with examples, and a conclusion encouraging further exploration."
            )

        st.markdown("---")
        if st.button("📊 Take Usability Survey (SUS)", help="Rate the app's usability"):
            st.session_state.page = 'survey'
            st.rerun()