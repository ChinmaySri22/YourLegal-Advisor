import streamlit as st

st.set_page_config(page_title="AI-LegalAdvisor", layout="wide")

import os
import pickle
import faiss
import numpy as np
import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# — Gemini config —

genai.configure(api_key=("AIzaSyDaqqAuBHx3MBHXd9jXWKzrALO6xTkvdvM"))
GEMINI_MODEL = "gemini-1.5-flash"

# — Load all models & resources once —
@st.cache_resource
def load_resources():
    # 1) Emotion detector: load the exact same TF-IDF you tested
    with open("models/emotion/emotion_vectorizer.pkl", "rb") as f:
        emo_vect = pickle.load(f)
    with open("models/emotion/emotion_classifier.pkl", "rb") as f:
        emo_model = pickle.load(f)
    try:
        with open("models/emotion/emotion_metrics.json", "r", encoding="utf-8") as f:
            emo_metrics = json.load(f)
    except:
        emo_metrics = {"accuracy":0,"precision":0,"recall":0,"f1_score":0}

    # 2) IPC classifier
    with open("models/Predict_Section/Law_vectorizer.pkl", "rb") as f:
        ipc_vect = pickle.load(f)
    with open("models/Predict_Section/Law_classifier.pkl", "rb") as f:
        ipc_clf = pickle.load(f)

    # 3) Section map
    with open("data/section_map.json", "r", encoding="utf-8") as f:
        section_map = json.load(f)

    # 4) FAISS retrieval
    with open("models/Retrival/index.pkl", "rb") as f:
        law_chunks = pickle.load(f)   # list of strings
    faiss_idx = faiss.read_index("models/Retrival/index.faiss")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return emo_vect, emo_model, emo_metrics, ipc_vect, ipc_clf, section_map, law_chunks, faiss_idx, embedder

# Unpack
(
    emo_vect, emo_model, emo_metrics,
    ipc_vect, ipc_clf, section_map,
    law_chunks, faiss_idx, embedder
) = load_resources()

# Emotion label map
EMO_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def detect_emotion(text: str) -> str:
    """Predict emotion using the trained model."""
    if not text or not text.strip():
        return "unknown"

    X = emo_vect.transform([text])
    raw_label = emo_model.predict(X)[0]

    if isinstance(raw_label, str):
        return raw_label

    try:
        idx = int(raw_label)
    except (ValueError, TypeError):
        raise RuntimeError(f"Model returned non-int, non-str label: {raw_label!r}")

    return EMO_MAP.get(idx, "unknown")


def predict_section(text: str) -> str:
    X = ipc_vect.transform([text])
    return ipc_clf.predict(X)[0]

def retrieve_laws(query: str, k: int = 3) -> list[str]:
    emb = embedder.encode([query])
    _, idx = faiss_idx.search(np.array(emb), k)
    return [law_chunks[i] for i in idx[0]]

def summarize(texts: list[str]) -> str:
    prompt = (
        "You are a legal assistant. Read these IPC excerpts and:\n"
        "1. List the section numbers.\n"
        "2. Summarize each simply.\n"
        "3. Highlight key takeaways.\n\n"
        + "\n\n---\n\n".join(texts)
    )
    gm = genai.GenerativeModel(GEMINI_MODEL)
    resp = gm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2, max_output_tokens=500
        )
    )
    return resp.text.strip()

# Session state init
if "submitted" not in st.session_state:
    st.session_state.submitted = False
for key in ("emotion","ipc_section","docs","summary"):
    if key not in st.session_state:
        st.session_state[key] = None

# UI
st.title("📜 AI-LegalAdvisor — Know Your Rights, Emotionally")
st.markdown(
    "Enter your legal query and click **Get Legal Help** to:\n"
    "1. Detect your **emotion**\n"
    "2. Fetch relevant **IPC snippets**\n"
    "3. Predict the **IPC section** & description\n"
    "4. Optionally **summarize** those snippets"
)

# Sidebar: emotion metrics only
st.sidebar.title("🧠 Emotion Model Metrics")
st.sidebar.metric("Accuracy",  f"{emo_metrics['accuracy']:.4f}")
st.sidebar.metric("Precision", f"{emo_metrics['precision']:.4f}")
st.sidebar.metric("Recall",    f"{emo_metrics['recall']:.4f}")
st.sidebar.metric("F1 Score",  f"{emo_metrics['f1_score']:.4f}")

query = st.text_area("Your legal question:", height=150)

if st.button("Get Legal Help"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.submitted   = True
        st.session_state.emotion     = detect_emotion(query)
        st.session_state.docs        = retrieve_laws(query)
        st.session_state.ipc_section = predict_section(query)
        st.session_state.summary     = None

if st.session_state.submitted:
    # Detected Emotion block with support message
    st.subheader("🧠 Detected Emotion")
    emotion_label = st.session_state.emotion.lower()
    st.success(f"**{emotion_label.upper()}**")
    if emotion_label in ("sadness", "fear"):
        st.warning(
            "💛 If you're feeling overwhelmed or hopeless, you're not alone. "
            "Please reach out for help. Trained mental health professionals in India are here to support you.\n\n"
            "**📞 Call or visit:** [Mental Health Support Numbers (PDF)](https://nhm.hp.gov.in/storage/app/media/uploaded-files/Mental%20Health%20Support%20Numbers.pdf)"
        )

    # IPC prediction
    sec = st.session_state.ipc_section
    desc = section_map.get(str(sec), "No description available.")
    st.subheader("📌 Predicted IPC Section")
    st.info(f"Section {sec}: {desc}")

    # Law snippets
    st.subheader("📘 Relevant Law Snippets")
    for i, txt in enumerate(st.session_state.docs, start=1):
        st.markdown(f"**{i}.** {txt}")

    if st.button("Summarize Laws"):
        with st.spinner("Summarizing…"):
            st.session_state.summary = summarize(st.session_state.docs)

    if st.session_state.summary:
        st.subheader("✏️ Summary of Sections")
        st.markdown(st.session_state.summary)
