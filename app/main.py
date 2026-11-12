import html
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.pipeline import RAGPipeline


HISTORY_FILE = PROJECT_ROOT / "data" / "chat_history.json"


def load_history():
    """
    Restore the previous conversation from disk so
    chat history survives restarts and refreshes.
    """

    if HISTORY_FILE.exists():

        try:

            saved_messages = json.loads(
                HISTORY_FILE.read_text(encoding="utf-8")
            )

            # Errors from a previous session are stale —
            # they describe problems that were likely
            # already fixed by the restart. Replay only
            # the real conversation.

            return [
                message
                for message in saved_messages
                if not message.get("error")
            ]

        except (json.JSONDecodeError, OSError):

            return []

    return []


def save_history(messages):
    """
    Persist the conversation to disk after every turn.
    """

    try:

        HISTORY_FILE.write_text(
            json.dumps(
                messages,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    except OSError:

        pass  # history is best-effort, never break the app


st.set_page_config(
    page_title="Stitch — Document Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. SESSION STATE
# ==================================================

defaults = {
    "pipeline": None,
    "document_names": [],
    "messages": load_history(),
    "retrieval_mode": "normal"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 3. CUSTOM CSS
# ==================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,600;0,700;0,800;0,900;1,600;1,700;1,800;1,900&display=swap');

/* GLOBAL */

.stApp {
    color: #111111;
    background-color: #ffffff;
    background-image:
        radial-gradient(circle at 85% 12%, rgba(17, 17, 17, 0.05) 0%, transparent 42%),
        radial-gradient(circle at 8% 88%, rgba(17, 17, 17, 0.04) 0%, transparent 38%),
        repeating-linear-gradient(0deg, transparent, transparent 79px, #f2f2f2 79px, #f2f2f2 80px),
        repeating-linear-gradient(90deg, transparent, transparent 79px, #f2f2f2 79px, #f2f2f2 80px);
    animation: bgDrift 28s linear infinite alternate;
}

@keyframes bgDrift {
    from {
        background-position:
            0 0,
            0 0,
            0 0,
            0 0;
    }
    to {
        background-position:
            70px 45px,
            -60px -40px,
            0 0,
            0 0;
    }
}

.block-container {
    max-width: 1320px;
    padding-top: 2.5rem !important;
    padding-bottom: 4rem;
}

html, body, [class*="css"] {
    font-family: 'Montserrat', Arial, Helvetica, sans-serif;
}

/* TYPOGRAPHY */

h1, h2, h3 {
    color: #111111 !important;
    font-weight: 800 !important;
    letter-spacing: -1.2px;
}

p, label {
    color: #333333;
}

/* BRAND — STITCH MARK */

.brand {
    width: fit-content;
    border-bottom: 4px dashed #111111;
    padding-bottom: 2px;
}

/* STITCH MOTIF — DASHED DIVIDERS */

hr {
    border: none !important;
    border-top: 2px dashed #dddddd !important;
}

/* STREAMLIT HEADER */

header[data-testid="stHeader"] {
    background: #ffffff;
    border-bottom: 2px dashed rgba(236, 28, 36, 0.35);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #f7f7f7;
    border-right: 1px solid #e5e5e5;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #111111 !important;
    letter-spacing: -0.5px;
}

/* BUTTONS */

.stButton > button {
    background: #111111;
    color: #ffffff;
    border: 1px solid #111111;
    border-radius: 30px;
    padding: 0.65rem 1.5rem;
    font-weight: 700;
    font-style: italic;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #333333;
    border-color: #333333;
    color: #ffffff;
}

/* INPUTS */

.stTextInput input,
.stChatInput textarea {
    background: #f5f5f5 !important;
    color: #111111 !important;
    border: 1px solid #dddddd !important;
    border-radius: 12px !important;
}

.stTextInput input:focus,
.stChatInput textarea:focus {
    border-color: #111111 !important;
    box-shadow: none !important;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #dddddd !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Inner upload drop zone */

[data-testid="stFileUploaderDropzone"] {
    background: #f7f7f7 !important;
    border: 1px dashed #bbbbbb !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #333333 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #111111 !important;
    border-radius: 24px !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color: #666666 !important;
}

/* Uploaded files */

[data-testid="stFileUploaderFile"] {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #222222 !important;
}

/* CHAT */

[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    padding: 18px;
    margin-bottom: 14px;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
}

/* METRICS */

div[data-testid="stMetric"] {
    background: #f7f7f7;
    border: 1px solid #eeeeee;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetric"] label {
    color: #777777 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111111 !important;
    font-weight: 800;
}

/* DIVIDERS */

hr {
    border-color: #e5e5e5;
}

/* BRANDING AND TOP NAVIGATION */

.top-nav {
    width: 100%;
    min-height: 72px;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    column-gap: 24px;
    padding: 8px 0;
    box-sizing: border-box;
}

/* Brand + tagline stacked, sharing a baseline */

.brand-block {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.brand {
    display: block;
    margin: 0;
    padding: 0 0 2px 0;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.1 !important;
    color: #111111;
    white-space: nowrap;
}

/* The period — the single red accent of the wordmark */

.brand-dot {
    color: #ec1c24;
}

.brand-small {
    margin: 0;
    padding: 0;
    font-size: 9px;
    letter-spacing: 2.4px;
    font-weight: 700;
    line-height: 1.4;
    color: #777777;
    text-align: left;
    white-space: nowrap;
}

.top-nav-status {
    margin: 0;
    padding: 0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #888888;
    text-align: right;
    white-space: nowrap;
}

.top-nav-status.active {
    color: #111111;
}

/* HERO */

.hero {
    padding: 65px 0 55px 0;
    position: relative;
    overflow: hidden;
}

/* Rotating stitch hoop — ambient motion */

.hero-ring {
    position: absolute;
    right: 20px;
    top: 50%;
    margin-top: -170px;
    width: 340px;
    height: 340px;
    border: 3px dashed #e3e3e3;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: hoopSpin 70s linear infinite;
}

@keyframes hoopSpin {
    to { transform: rotate(360deg); }
}

/* Inner hoop for depth — Nike red accent */

.hero-ring::before {
    content: "";
    position: absolute;
    inset: 26px;
    border: 2px dashed rgba(236, 28, 36, 0.35);
    border-radius: 50%;
}

/* FLOATING FEATURE CARDS — hero right side */

.hero-cards {
    position: absolute;
    right: 48px;
    top: 50%;
    margin-top: -170px;
    width: 320px;
    height: 340px;
    z-index: 2;
    pointer-events: none;
}

.fcard {
    position: absolute;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.14);
    animation:
        fadeUpCard 0.6s ease both,
        floatRot 8s ease-in-out 1.4s infinite;
}

@keyframes fadeUpCard {
    from {
        opacity: 0;
        transform: translateY(16px) rotate(var(--rot, 0deg));
    }
    to {
        opacity: 1;
        transform: translateY(0) rotate(var(--rot, 0deg));
    }
}

@keyframes floatRot {
    0%, 100% {
        transform: translateY(0) rotate(var(--rot, 0deg));
    }
    50% {
        transform: translateY(-9px) rotate(var(--rot, 0deg));
    }
}

.fcard-index {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin: 0;
}

.fcard-title {
    font-size: 16px;
    font-weight: 900;
    font-style: italic;
    letter-spacing: -0.5px;
    text-transform: uppercase;
    margin: 8px 0 0 0;
}

.fcard-text {
    font-size: 11.5px;
    line-height: 1.6;
    margin: 8px 0 0 0;
}

.fcard-black {
    background: #111111;
    --rot: 3deg;
    top: 0;
    right: 70px;
    transform: rotate(3deg);
    animation-delay: 0.45s, 1.4s;
}

.fcard-black .fcard-index { color: #999999; }
.fcard-black .fcard-title { color: #ffffff; }
.fcard-black .fcard-text { color: #cccccc; }

.fcard-white {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-left: 5px solid #ec1c24;
    --rot: -2.5deg;
    top: 128px;
    right: 0;
    transform: rotate(-2.5deg);
    animation-delay: 0.6s, 1.75s;
}

.fcard-white .fcard-index { color: #ec1c24; }
.fcard-white .fcard-title { color: #111111; }
.fcard-white .fcard-text { color: #555555; }

.fcard-red {
    background: #ec1c24;
    --rot: 2deg;
    top: 252px;
    right: 108px;
    transform: rotate(2deg);
    animation-delay: 0.75s, 2.1s;
}

.fcard-red .fcard-index { color: rgba(255, 255, 255, 0.75); }
.fcard-red .fcard-title { color: #ffffff; }
.fcard-red .fcard-text { color: rgba(255, 255, 255, 0.9); }

.hero > div:not(.hero-ring):not(.hero-cards) {
    position: relative;
    z-index: 1;
}

.hero-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #777777;
    margin-bottom: 22px;
}

.hero-title {
    font-size: clamp(44px, 7vw, 82px);
    font-weight: 900;
    font-style: italic;
    line-height: 0.98;
    letter-spacing: -5px;
    color: #111111;
    max-width: 900px;
}

.hero-description {
    color: #666666;
    font-size: 16px;
    margin-top: 25px;
    max-width: 550px;
    line-height: 1.7;
    animation: fadeUp 0.7s ease 0.35s both;
}

/* TAGLINE — the stitch thread */

.hero-tagline {
    display: inline-block;
    margin-top: 34px;
    font-size: 19px;
    font-style: italic;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #111111;
    border-bottom: 3px dashed #ec1c24;
    padding-bottom: 8px;
    animation: fadeUp 0.7s ease 0.5s both;
}

/* SECTION LABELS — with a red stitch dash */

.section-label {
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 800;
    color: #888888;
    margin: 4px 0 14px 0;
}

.section-label::before {
    content: "";
    display: inline-block;
    width: 20px;
    border-top: 3px dashed #ec1c24;
    vertical-align: middle;
    margin-right: 9px;
}

/* SIDEBAR GEOMETRY */

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.7rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

.side-brand {
    font-size: 16px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: #111111;
    margin-bottom: 20px;
}

.source-meta {
    color: #777777;
    font-size: 12px;
}

/* FOOTER */

.footer {
    display: flex;
    justify-content: space-between;
    color: #888888;
    font-size: 11px;
    padding: 15px 0;
    letter-spacing: 1px;
}

/* TRANSITIONS AND MOTION */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes softPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* Hero entrance */

.hero {
    animation: fadeUp 0.7s ease both;
}

.hero-label {
    animation: fadeUp 0.7s ease 0.1s both;
}

.hero-title {
    animation: fadeUp 0.7s ease 0.2s both;
}

.hero-description {
    animation: fadeUp 0.7s ease 0.35s both;
}

/* Section reveal */

.section-label {
    animation: fadeUp 0.5s ease both;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    animation: fadeUp 0.4s ease both;
    transition:
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: #dddddd;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.06);
}

/* FLOATING CARDS */

@keyframes floatIdle {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

/* Metrics: enter, then float gently */

div[data-testid="stMetric"] {
    animation:
        fadeUp 0.55s ease both,
        floatIdle 7s ease-in-out 1.4s infinite;
    transition:
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}

/* Stagger metric cards left to right */

[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {
    animation-delay: 0.05s, 1.4s;
}

[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
    animation-delay: 0.18s, 1.7s;
}

[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
    animation-delay: 0.31s, 2s;
}

[data-testid="stExpander"] {
    animation: none;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #dddddd;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07);
}

/* Expanders ease open */

[data-testid="stExpander"] {
    transition: border-color 0.25s ease;
}

[data-testid="stExpander"]:hover {
    border-color: #cccccc;
}

/* Buttons lift */

.stButton > button {
    transition: all 0.25s cubic-bezier(0.2, 0.7, 0.3, 1);
}

/* Button labels live inside <p> tags, which inherit
   the global paragraph color. Force white on dark. */

.stButton > button p,
[data-testid="stFileUploaderDropzone"] button p {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: #ec1c24;
    border-color: #ec1c24;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(236, 28, 36, 0.3);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: none;
}

/* Inputs focus glow */

.stTextInput input,
.stChatInput textarea {
    transition:
        border-color 0.25s ease,
        background 0.25s ease,
        box-shadow 0.25s ease;
}

.stTextInput input:focus,
.stChatInput textarea:focus {
    border-color: #111111 !important;
    box-shadow: 0 0 0 3px rgba(17, 17, 17, 0.06) !important;
}

/* File uploader hover */

[data-testid="stFileUploaderDropzone"] {
    transition:
        border-color 0.25s ease,
        background 0.25s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #111111 !important;
    background: #f2f2f2 !important;
}

/* Live workspace status pulse */

.top-nav-status.active {
    animation: softPulse 2.6s ease-in-out infinite;
}

/* ERROR STATE CARD */

.error-card {
    background: #111111;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
    animation: fadeUp 0.4s ease both;
}

.error-card-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    font-weight: 800;
    color: #ff7b72;
    margin: 0;
}

.error-card-text {
    color: #f0f0f0;
    font-size: 14px;
    line-height: 1.6;
    margin: 10px 0 0 0;
}

.error-card-hint {
    color: #8a8a8a;
    font-size: 12px;
    line-height: 1.6;
    margin: 12px 0 0 0;
}

/* COMMAND CAPSULE CHAT INPUT */

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: #ffffff !important;
}

[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 24px !important;
    padding: 4px 6px 4px 4px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.14);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stChatInput"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 38px rgba(0, 0, 0, 0.18);
}

@keyframes inputHalo {
    0%, 100% {
        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.14),
            0 0 0 5px rgba(17, 17, 17, 0.08);
    }
    50% {
        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.14),
            0 0 0 11px rgba(17, 17, 17, 0.04);
    }
}

[data-testid="stChatInput"]:focus-within {
    animation: inputHalo 2.6s ease-in-out infinite;
}

[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #111111 !important;
    border: none !important;
    font-size: 15px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #999999 !important;
    font-weight: 600;
}

[data-testid="stChatInput"] button {
    background: #111111 !important;
    border-radius: 50% !important;
    color: #ffffff !important;
    transition: all 0.25s ease;
}

[data-testid="stChatInput"] button:hover {
    background: #ec1c24 !important;
    transform: scale(1.08);
}

/* Uploaded file chips — keep light, readable */

[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileList"] {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
}

[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileList"] * {
    color: #222222 !important;
    background: transparent !important;
}

/* ALERTS MATCH THE PALETTE */

[data-testid="stAlert"],
[data-testid="stAlertContainer"],
[data-testid="stAlertBody"] {
    background: #f7f7f7 !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] *,
[data-testid="stAlertContainer"] *,
[data-testid="stAlertBody"] * {
    color: #111111 !important;
}

/* EMPTY STATE CARD */

.empty-state {
    background: #f7f7f7;
    border: 1px dashed #cccccc;
    border-radius: 14px;
    padding: 28px 26px;
    animation: fadeUp 0.5s ease both;
}

.empty-state-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    font-weight: 800;
    color: #888888;
    margin: 0;
}

.empty-state-text {
    color: #333333;
    font-size: 15px;
    line-height: 1.6;
    margin: 10px 0 0 0;
}

/* EXPANDERS — unified card language */

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stExpander"]:hover {
    border-color: #dddddd;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
}

/* ============================================
   POLISH LAYER — scrollbars, selection, shine
   ============================================ */

/* Thin custom scrollbars everywhere */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #d8d8d8;
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: #ec1c24;
}

/* Nike-red text selection */

::selection {
    background: #ec1c24;
    color: #ffffff;
}

/* Sidebar geometry — tighter, focused */

section[data-testid="stSidebar"] {
    width: 300px !important;
    min-width: 300px !important;
}

/* Button shine sweep on hover */

.stButton > button {
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: "";
    position: absolute;
    top: 0;
    left: -80%;
    width: 50%;
    height: 100%;
    background: linear-gradient(
        120deg,
        transparent,
        rgba(255, 255, 255, 0.35),
        transparent
    );
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}

.stButton > button:hover::after {
    left: 130%;
}

/* STITCH TICKER — scrolling capability band */

.ticker-wrap {
    overflow: hidden;
    background: #111111;
    border-radius: 14px;
    padding: 14px 0;
    margin-top: 40px;
}

.ticker {
    display: inline-flex;
    white-space: nowrap;
    animation: tickerScroll 24s linear infinite;
}

.ticker span {
    color: #ffffff;
    font-style: italic;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 0 26px;
}

.ticker span::after {
    content: "✦";
    color: #ec1c24;
    margin-left: 52px;
}

@keyframes tickerScroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* Chat section title — brand italic weight */

.ask-title {
    font-size: 30px;
    font-weight: 900;
    font-style: italic;
    letter-spacing: -1.5px;
    color: #111111;
    margin: 0 0 4px 0;
    animation: fadeUp 0.5s ease both;
}

/* MOBILE */

@media (max-width: 1100px) {

    .hero-cards {
        display: none;
    }

}

@media (max-width: 768px) {

    .hero {
        padding: 40px 0;
    }

    .hero-ring {
        display: none;
    }

    .hero-title {
        font-size: 46px;
        letter-spacing: -2px;
    }

    .top-nav {
        min-height: 64px;
        grid-template-columns: 1fr auto;
        column-gap: 12px;
    }

    .brand {
        font-size: 24px;
        letter-spacing: -1.5px;
    }

    .brand-small {
        display: none;
    }

    .top-nav-status {
        font-size: 9px;
        letter-spacing: 1px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ==================================================
# 4. TOP NAVIGATION
# ==================================================

if st.session_state.pipeline is not None:
    workspace_status = (
        '<div class="top-nav-status active">'
        '● WORKSPACE ACTIVE'
        '</div>'
    )
else:
    workspace_status = (
        '<div class="top-nav-status">'
        'YOUR PRIVATE WORKSPACE'
        '</div>'
    )

st.markdown(
    '<div class="top-nav">'
    '<div class="brand-block">'
    '<div class="brand">STITCH<span class="brand-dot">.</span></div>'
    '<div class="brand-small">ENTERPRISE DOCUMENT INTELLIGENCE</div>'
    '</div>'
    f'{workspace_status}'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 5. SIDEBAR - DOCUMENT MANAGEMENT
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="side-brand">STITCH<span class="brand-dot">.</span></div>',
        unsafe_allow_html=True
    )

    # ==============================================
    # SECTION 01 / LIBRARY
    # ==============================================

    st.markdown(
        '<div class="section-label">01 / LIBRARY</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "PROCESS DOCUMENTS →",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing your documents..."
                ):

                    file_paths = []
                    source_names = []

                    with tempfile.TemporaryDirectory() as temp_dir:

                        for uploaded_file in uploaded_files:

                            filename = Path(
                                uploaded_file.name
                            ).name

                            file_path = (
                                Path(temp_dir) / filename
                            )

                            file_path.write_bytes(
                                uploaded_file.getbuffer()
                            )

                            file_paths.append(str(file_path))
                            source_names.append(uploaded_file.name)

                        pipeline = RAGPipeline(
                            file_paths,
                            source_name=source_names
                        )

                    st.session_state.pipeline = pipeline
                    st.session_state.document_names = source_names

                st.success(
                    f"Workspace ready · "
                    f"{len(source_names)} document(s) · "
                    f"{len(pipeline.chunks)} passages indexed"
                )

            except Exception as error:

                st.error(
                    f"Document processing failed: {error}"
                )

    # Compact knowledge-base status — one line, no repetition

    if st.session_state.pipeline is not None:

        st.caption(
            f"🧠 {len(st.session_state.pipeline.chunks)} passages · "
            f"{len(st.session_state.document_names)} documents indexed"
        )

    st.divider()

    # ==============================================
    # SECTION 02 / ASSISTANT
    # ==============================================

    st.markdown(
        '<div class="section-label">02 / ASSISTANT</div>',
        unsafe_allow_html=True
    )

    retrieval_mode = st.radio(
        "Retrieval strategy",
        ["normal", "diverse (MMR)"],
        index=0 if st.session_state.retrieval_mode == "normal" else 1,
        help=(
            "Normal picks the most similar passages. "
            "Diverse (MMR) trades a little relevance for "
            "variety when passages repeat."
        )
    )

    st.session_state.retrieval_mode = (
        "normal"
        if retrieval_mode == "normal"
        else "mmr"
    )

    if st.button(
        "NEW CHAT",
        use_container_width=True
    ):

        st.session_state.messages = []

        save_history(st.session_state.messages)

        st.rerun()

    st.divider()

    st.caption("STITCH · LOCAL-FIRST · V0.1")


# ==================================================
# 6. HERO SECTION — shown until the workspace
#    is active, then the chat takes over
# ==================================================

hero_html = (
    '<div class="hero">'
    '<div class="hero-ring"></div>'
    '<div class="hero-label">CONNECTING INFORMATION</div>'
    '<div class="hero-title">'
    'MAKING KNOWLEDGE<br>ACCESSIBLE.'
    '</div>'
    '<div class="hero-description">'
    'Enterprise document intelligence and knowledge '
    'retrieval. Search, understand and explore your '
    'documents with an assistant that answers from '
    'your actual sources.'
    '</div>'
    '<div class="hero-tagline">'
    'Connecting the threads of knowledge.'
    '</div>'
    '<div class="hero-cards">'
    '<div class="fcard fcard-black">'
    '<div class="fcard-index">01 / CITED BY DEFAULT</div>'
    '<div class="fcard-title">Traced answers</div>'
    '<div class="fcard-text">Every response points back to the exact '
    'source file and page it came from.</div>'
    '</div>'
    '<div class="fcard fcard-white">'
    '<div class="fcard-index">02 / TRUST SCORES</div>'
    '<div class="fcard-title">Know the confidence</div>'
    '<div class="fcard-text">Confidence, groundedness and completeness '
    'scored on every single answer.</div>'
    '</div>'
    '<div class="fcard fcard-red">'
    '<div class="fcard-index">03 / LOCAL FIRST</div>'
    '<div class="fcard-title">Yours only</div>'
    '<div class="fcard-text">Your documents never leave your machine. '
    'Fully local retrieval and generation.</div>'
    '</div>'
    '</div>'
    '<div class="ticker-wrap"><div class="ticker">'
    '<span>Cited answers</span>'
    '<span>Trust scores</span>'
    '<span>Local-first</span>'
    '<span>MMR retrieval</span>'
    '<span>Source pages</span>'
    '<span>Grounded AI</span>'
    '<span>Cited answers</span>'
    '<span>Trust scores</span>'
    '<span>Local-first</span>'
    '<span>MMR retrieval</span>'
    '<span>Source pages</span>'
    '<span>Grounded AI</span>'
    '</div></div>'
    '</div>'
)

if st.session_state.pipeline is None:

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )


# ==================================================
# 7. DOCUMENT ASSISTANT
# ==================================================

st.markdown(
    '<div class="section-label">'
    'DOCUMENT ASSISTANT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ask-title">Ask anything.</div>',
    unsafe_allow_html=True
)

st.caption(
    "Get answers directly from your uploaded documents."
)


if st.session_state.pipeline is None:

    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-state-label">WORKSPACE EMPTY</div>'
        '<div class="empty-state-text">Your workspace is waiting. '
        'Upload documents from the sidebar to begin.</div>'
        '</div>',
        unsafe_allow_html=True
    )

else:

    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        # ERROR MESSAGES RENDER AS STYLED CARDS

        if message.get("error"):

            error_text = html.escape(
                message["content"]
            )

            st.markdown(
                '<div class="error-card">'
                '<div class="error-card-label">'
                '⚠ UNABLE TO GENERATE ANSWER'
                '</div>'
                f'<div class="error-card-text">{error_text}</div>'
                '<div class="error-card-hint">'
                'Check the error above, resolve it, then ask '
                'your question again. Your documents stay '
                'indexed while you fix it.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            continue

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                results = message.get("sources", [])

                if results:

                    with st.expander(
                        f"VIEW SOURCES · {len(results)} PASSAGES"
                    ):

                        for index, result in enumerate(
                            results,
                            start=1
                        ):

                            source = result.get(
                                "source",
                                "Unknown source"
                            )

                            page = result.get("page")
                            score = result.get("score", 0)

                            if page is not None:
                                page_label = f"Page {page}"
                            else:
                                page_label = "Page unavailable"

                            st.markdown(
                                f"**{index:02d} / {source}**"
                            )

                            st.caption(
                                f"{page_label} · "
                                f"Similarity {score:.4f}"
                            )

                            st.write(result["text"])

                            if index < len(results):
                                st.divider()

                # ANSWER QUALITY PANEL

                evaluation = message.get("evaluation")

                if evaluation:

                    verdict = evaluation["verdict"]

                    verdict_badges = {
                        "confident": "🟢 CONFIDENT",
                        "partial": "🟡 PARTIAL",
                        "unconfident": "🔴 UNCONFIDENT"
                    }

                    st.markdown(
                        "**ANSWER QUALITY · "
                        + verdict_badges.get(
                            verdict,
                            verdict.upper()
                        )
                        + "**"
                    )

                    metric_col1, metric_col2, metric_col3 = (
                        st.columns(3)
                    )

                    with metric_col1:
                        st.metric(
                            "Confidence",
                            f"{evaluation['confidence']:.0%}"
                        )

                    with metric_col2:
                        st.metric(
                            "Groundedness",
                            f"{evaluation['groundedness']:.0%}"
                        )

                    with metric_col3:
                        st.metric(
                            "Completeness",
                            f"{evaluation['completeness']:.0%}"
                        )

                    if evaluation["issues"]:

                        with st.expander(
                            f"DETECTED ISSUES · {len(evaluation['issues'])}"
                        ):

                            for issue in evaluation["issues"]:
                                st.markdown(f"- {issue}")

    # CHAT INPUT

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        pipeline = st.session_state.pipeline

        # Prior turns (excluding the question just asked,
        # which is passed separately) for follow-up context

        history = [
            m for m in st.session_state.messages[:-1]
            if not m.get("error")
        ]

        try:

            with st.spinner(
                "Searching your documents..."
            ):

                results, context = pipeline.prepare(
                    question
                )

            # STREAM THE ANSWER LIVE

            with st.chat_message("assistant"):

                full_answer = st.write_stream(
                    pipeline.generator.generate_stream(
                        question,
                        context,
                        history=history
                    )
                )

            # Score the completed answer, then hand
            # over to the persisted conversation view

            evaluation = pipeline.evaluator.evaluate_answer(
                question,
                context,
                full_answer,
                results
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": results,
                "evaluation": evaluation,
                "error": False
            })

            save_history(st.session_state.messages)

        except Exception as error:

            st.session_state.messages.append({
                "role": "assistant",
                "content": str(error),
                "sources": [],
                "evaluation": None,
                "error": True
            })

            save_history(st.session_state.messages)

        st.rerun()


# ==================================================
# 9. FOOTER
# ==================================================

footer_html = (
    '<div class="footer">'
    '<span>STITCH<span class="brand-dot">.</span></span>'
    '<span>ENTERPRISE DOCUMENT INTELLIGENCE</span>'
    '</div>'
)

st.divider()

st.markdown(
    footer_html,
    unsafe_allow_html=True
)