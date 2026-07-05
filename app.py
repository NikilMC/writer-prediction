import os
import streamlit as st
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

from transformers import BertTokenizer, BertModel
import torch

# =========================
# LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_mean = joblib.load(os.path.join(BASE_DIR, "model_mean.pkl"))
model_cls = joblib.load(os.path.join(BASE_DIR, "model_cls.pkl"))
model_max = joblib.load(os.path.join(BASE_DIR, "model_max.pkl"))
meta_model = joblib.load(os.path.join(BASE_DIR, "meta_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# =========================
# LOAD BERT
# =========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Author Predictor",
    layout="centered"
)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# =========================
# URL TEXT EXTRACTION
# =========================
def extract_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
    paragraphs = [p for p in paragraphs if len(p) > 60]
    return "\n\n".join(paragraphs)

# =========================
# BERT ENCODING
# =========================
def encode_all(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        out = bert_model(**inputs)
    hidden = out.last_hidden_state
    mean_vec = hidden.mean(dim=1).squeeze().numpy()
    cls_vec = hidden[:, 0, :].squeeze().numpy()
    max_vec = hidden.max(dim=1).values.squeeze().numpy()
    return mean_vec, cls_vec, max_vec

# =========================
# PREDICTION FUNCTION
# =========================
def predict_author(text):
    text = clean_text(text)
    mean_vec, cls_vec, max_vec = encode_all(text)
    x_mean = mean_vec.reshape(1, -1)
    x_cls = cls_vec.reshape(1, -1)
    x_max = max_vec.reshape(1, -1)
    p1 = model_mean.predict_proba(x_mean)
    p2 = model_cls.predict_proba(x_cls)
    p3 = model_max.predict_proba(x_max)
    meta_X = np.hstack([p1, p2, p3])
    final_probs = meta_model.predict_proba(meta_X)[0]
    pred_idx = np.argmax(final_probs)
    predicted_author = le.inverse_transform([pred_idx])[0]
    return predicted_author, final_probs

# =========================
# UI
# =========================
st.title("🧠 Author Prediction")
st.write(
    "Enter a blog paragraph manually, or fetch content from a URL, "
    "then click **Predict** to identify the author using BERT embeddings and a stacking ensemble model."
)

st.divider()

# Initialize session state
if "extracted_url_text" not in st.session_state:
    st.session_state["extracted_url_text"] = ""
if "fetched_url" not in st.session_state:
    st.session_state["fetched_url"] = ""

# Tabs
tab_manual, tab_url = st.tabs(["📝 Enter Text Manually", "🌐 Fetch from URL"])

user_input = ""

with tab_manual:
    manual_text = st.text_area(
        "Enter Blog Text:",
        height=250,
        placeholder="Paste a paragraph or blog post content here...",
        key="manual_input"
    )
    if manual_text:
        user_input = manual_text

with tab_url:
    url_input = st.text_input(
        "Enter Blog Post URL:",
        placeholder="e.g., https://the-shooting-star.com/kigali-rwanda-meaningful-things-to-do/",
        key="url_input"
    )

    if st.button("Extract Text", key="extract_btn"):
        if url_input:
            try:
                with st.spinner("Fetching content from URL..."):
                    extracted = extract_text_from_url(url_input)
                if extracted:
                    st.session_state["extracted_url_text"] = extracted
                    st.session_state["extracted_url_input"] = extracted
                    st.session_state["fetched_url"] = url_input
                    st.success("✅ Content successfully fetched!")
                else:
                    st.error("No substantial paragraph text could be extracted from this URL.")
            except Exception as e:
                st.error(f"Error fetching URL: {str(e)}")
        else:
            st.warning("Please enter a URL first.")

    if st.session_state["extracted_url_text"]:
        if url_input != st.session_state["fetched_url"]:
            st.info("⚠️ URL has changed. Click 'Extract Text' to fetch content from the new URL.")

        url_text = st.text_area(
            "Extracted Text (You can edit this before predicting):",
            height=250,
            key="extracted_url_input"
        )
        if url_text:
            user_input = url_text

st.divider()

# =========================
# PREDICT BUTTON
# =========================
if st.button("🔍 Predict Author", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text or extract content from a URL first.")
    else:
        with st.spinner("Predicting author..."):
            predicted_author, probs = predict_author(user_input)

        top_indices = np.argsort(probs)[::-1][:3]
        top_authors = le.inverse_transform(top_indices)
        top_probs = probs[top_indices]
        confidence = top_probs[0] * 100

        st.success(f"🎯 Predicted Author: **{predicted_author}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        st.subheader("📊 Top 3 Probabilities")
        for i in range(3):
            author = top_authors[i]
            prob_val = float(top_probs[i])
            prob_pct = prob_val * 100
            st.write(f"**{i + 1}. {author}** — {prob_pct:.2f}%")
            st.progress(prob_val)