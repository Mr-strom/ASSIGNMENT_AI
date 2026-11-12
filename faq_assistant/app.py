"""
Streamlit web interface for the Course & College FAQ Assistant.
Provides student chat, prompt strategy selection, side-by-side comparison,
an FAQ browser, and evaluation metrics.
"""

import html
import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from faq_assistant.config import (
    SYSTEM_SPECS,
    OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    DATA_PATH,
    RESULTS_DIR
)
from faq_assistant.engine import FAQAssistantEngine
from faq_assistant.compare import compare_prompting_strategies
from faq_assistant.intent import classify_intent_rule_based
from faq_assistant.memory import ConversationMemory

# Page Configuration
st.set_page_config(
    page_title="Course & College FAQ Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Card, badge, and header styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* App Header Styling */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 6px;
}

/* Intent Pills */
.intent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.intent-Attendance { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
.intent-Examination { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
.intent-Assignment { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
.intent-Course-Content { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.intent-General { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }

/* Comparison Cards */
.strategy-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px;
    height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.strategy-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
}

.stat-pill {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.75rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "engine" not in st.session_state:
    st.session_state.engine = FAQAssistantEngine()


# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">🎓 Course & College FAQ Assistant</h1>
        <div class="main-subtitle">
            Local Ollama academic FAQ assistant with Zero-Shot, Few-Shot, and Role prompting
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Assistant Settings")

    selected_strategy = st.selectbox(
        "Prompting Strategy",
        ["zero-shot", "few-shot", "role"],
        index=0,
        help="Select between direct zero-shot, few-shot with 3 exemplars, or Dr. Vance advising role prompting.",
        key="strategy_selector"
    )

    strategy_desc = {
        "zero-shot": "📌 **Zero-Shot:** Direct policy instruction without Q&A exemplars.",
        "few-shot": "💡 **Few-Shot:** Demonstrates 3 reference Q&A pairs (including out-of-scope refusal).",
        "role": "🧑‍🏫 **Role (Dr. Vance):** Seasoned CS Department Academic Advisor persona."
    }
    st.info(strategy_desc[selected_strategy])

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        key="temperature_slider"
    )

    use_memory = st.checkbox("Enable Conversation Memory (N-turns)", value=True, key="memory_toggle")

    if st.button("🧹 Clear Chat History", use_container_width=True, key="clear_chat_btn"):
        st.session_state.chat_history.clear()
        st.session_state.engine.memory.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("🖥️ Hardware & Model")
    st.markdown(f"**Model:** `{st.session_state.engine.client.model}`")
    st.markdown(f"**Available RAM:** `{SYSTEM_SPECS['available_ram_gb']} GB`")
    st.markdown(f"**Total RAM:** `{SYSTEM_SPECS['total_ram_gb']} GB`")
    st.markdown(f"**Compute:** `{SYSTEM_SPECS['gpu']}`")


# Main Tabs
tab_chat, tab_compare, tab_faqs, tab_eval = st.tabs([
    "💬 Student Advising Chat",
    "⚖️ Side-by-Side Prompt Comparison",
    "📚 Official FAQ Knowledge Base",
    "📊 Evaluation & Metrics"
])

# ---------------------------------------------------------------------------
# TAB 1: Chat Interface
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown("### Ask any academic or college policy question:")

    # Quick question prompt chips
    chips = [
        "What is the minimum attendance required for final exams?",
        "What is the penalty for late assignment submission?",
        "Can I bring calculators to the midterm examination?",
        "What are the prerequisites for CS102 Data Structures?",
        "Can I bring my pet python to live in the dorms?"
    ]
    st.caption("Quick sample inquiries:")
    chip_cols = st.columns(len(chips))
    chip_clicked = None
    for idx, chip in enumerate(chips):
        with chip_cols[idx]:
            if st.button(chip, key=f"chip_{idx}", use_container_width=True):
                chip_clicked = chip

    # Render Existing History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            if msg["role"] == "assistant":
                intent_safe = msg.get("intent", "General").replace(" ", "-")
                st.markdown(
                    f'<span class="intent-badge intent-{intent_safe}">Intent: {msg.get("intent", "General")}</span> '
                    f'<span class="stat-pill">Strategy: {msg.get("strategy", "zero-shot")}</span> '
                    f'<span class="stat-pill">Latency: {msg.get("latency", 0.0)}s</span>',
                    unsafe_allow_html=True
                )
            st.markdown(msg["content"])

    # Chat Input
    query_input = st.chat_input("Type your question here (e.g., 'What is the attendance condonation rule?')...", key="chat_input_field")
    active_query = chip_clicked or query_input

    if active_query:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(active_query)

        # Process Assistant Response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Consulting official university handbook..."):
                t0 = time.time()
                result = st.session_state.engine.ask(
                    query=active_query,
                    strategy=selected_strategy,
                    temperature=temperature,
                    use_memory=use_memory
                )
                latency = round(time.time() - t0, 2)

                intent_safe = result["intent"].replace(" ", "-")
                st.markdown(
                    f'<span class="intent-badge intent-{intent_safe}">Intent: {result["intent"]}</span> '
                    f'<span class="stat-pill">Strategy: {result["strategy"]}</span> '
                    f'<span class="stat-pill">Latency: {latency}s</span>',
                    unsafe_allow_html=True
                )
                st.markdown(result["response"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["response"],
                    "intent": result["intent"],
                    "strategy": result["strategy"],
                    "latency": latency
                })


# ---------------------------------------------------------------------------
# TAB 2: Side-by-Side Comparison Studio
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Multi-Strategy Prompting Comparison Studio")
    st.write(
        "Evaluate how the **exact same question** is handled simultaneously across "
        "**Zero-Shot**, **Few-Shot**, and **Role Prompting (Dr. Vance)**."
    )

    compare_query = st.text_input(
        "Enter question to compare side-by-side:",
        value="What is the penalty if I turn in an assignment late?",
        key="compare_input_field"
    )

    if st.button("🚀 Run Side-by-Side Comparison", type="primary", key="run_compare_btn"):
        with st.spinner("Executing inferences across all 3 prompt strategies..."):
            data = compare_prompting_strategies(compare_query)
            res = data["results"]

            col_z, col_f, col_r = st.columns(3)

            with col_z:
                st.markdown(
                    f"""
                    <div class="strategy-card">
                        <h4 style="color: #38bdf8; margin-top: 0;">1. Zero-Shot Prompting</h4>
                        <div style="margin-bottom: 12px;">
                            <span class="stat-pill">⏱️ {res['zero-shot']['latency_seconds']}s</span>
                            <span class="stat-pill">📝 {res['zero-shot']['word_count']} words</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                            {html.escape(res['zero-shot']['response'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_f:
                st.markdown(
                    f"""
                    <div class="strategy-card">
                        <h4 style="color: #facc15; margin-top: 0;">2. Few-Shot Prompting</h4>
                        <div style="margin-bottom: 12px;">
                            <span class="stat-pill">⏱️ {res['few-shot']['latency_seconds']}s</span>
                            <span class="stat-pill">📝 {res['few-shot']['word_count']} words</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                            {html.escape(res['few-shot']['response'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_r:
                st.markdown(
                    f"""
                    <div class="strategy-card">
                        <h4 style="color: #c084fc; margin-top: 0;">3. Role Prompting (Dr. Vance)</h4>
                        <div style="margin-bottom: 12px;">
                            <span class="stat-pill">⏱️ {res['role']['latency_seconds']}s</span>
                            <span class="stat-pill">📝 {res['role']['word_count']} words</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                            {html.escape(res['role']['response'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Summary table
            st.markdown("#### Strategy Comparison Summary")
            summary_df = pd.DataFrame([
                {
                    "Strategy": "Zero-Shot",
                    "Latency (s)": res["zero-shot"]["latency_seconds"],
                    "Word Count": res["zero-shot"]["word_count"],
                    "Detected Intent": res["zero-shot"]["intent"],
                    "Reliable Context": res["zero-shot"]["is_reliable"]
                },
                {
                    "Strategy": "Few-Shot",
                    "Latency (s)": res["few-shot"]["latency_seconds"],
                    "Word Count": res["few-shot"]["word_count"],
                    "Detected Intent": res["few-shot"]["intent"],
                    "Reliable Context": res["few-shot"]["is_reliable"]
                },
                {
                    "Strategy": "Role Prompting",
                    "Latency (s)": res["role"]["latency_seconds"],
                    "Word Count": res["role"]["word_count"],
                    "Detected Intent": res["role"]["intent"],
                    "Reliable Context": res["role"]["is_reliable"]
                }
            ])
            st.dataframe(summary_df, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: FAQ Knowledge Base Catalog
# ---------------------------------------------------------------------------
with tab_faqs:
    st.markdown("### FAQ Knowledge Base (`faq_data.json`)")
    st.caption("Browse or filter the FAQ entries loaded from faq_data.json.")

    search_kw = st.text_input("Filter FAQs by keyword or question:", "", key="faq_filter_input")

    faqs = st.session_state.engine.retriever.faqs
    if search_kw:
        faqs = [f for f in faqs if search_kw.lower() in f.full_text().lower()]

    for faq in faqs:
        with st.expander(f"[{faq.category}] {faq.question} ({faq.id})"):
            st.markdown(f"**Policy Answer:**\n{faq.answer}")
            st.markdown(f"**Indexed Keywords:** `{'`, `'.join(faq.keywords)}`")


# ---------------------------------------------------------------------------
# TAB 4: Evaluation & Diagnostics
# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown("### Evaluation Results")
    summary_md_path = RESULTS_DIR / "evaluation_summary.md"
    csv_path = RESULTS_DIR / "evaluation_results.csv"

    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run `python run.py --eval` to generate the evaluation report.")

    if csv_path.exists():
        st.markdown("#### Raw Evaluation Results")
        raw_df = pd.read_csv(csv_path)
        st.dataframe(raw_df, use_container_width=True)
