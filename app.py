import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="TruthCheck", page_icon="🔍", layout="centered")

st.title("🔍 TruthCheck")
st.subheader("Amazon Review Authenticity Detector")
st.markdown("Paste an Amazon review below to check if it's **genuine or fake**.")

# Load models
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained('./truthcheck_model')
    model = DistilBertForSequenceClassification.from_pretrained('./truthcheck_model')
    model.eval()
    tfidf = joblib.load('./baseline_model/lr_model.pkl')
    lr = joblib.load('./baseline_model/tfidf.pkl')
    return tokenizer, model, tfidf, lr

tokenizer, model, tfidf, lr = load_models()

# Input
review = st.text_area("Paste review here", height=150, placeholder="e.g. This product is amazing! Best purchase I've ever made...")

if st.button("Analyze Review"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):

            # DistilBERT prediction
            inputs = tokenizer(review, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).squeeze()
                bert_pred = torch.argmax(probs).item()
                bert_conf = probs[bert_pred].item()

        # Display result
        st.divider()
        if bert_pred == 1:
            st.success(f"✅ **GENUINE** review — {bert_conf*100:.1f}% confidence")
        else:
            st.error(f"🚨 **FAKE** review — {bert_conf*100:.1f}% confidence")

        # Confidence bar
        st.markdown("**Confidence Score**")
        st.progress(bert_conf)

        # Model comparison
        with st.expander("See model comparison"):
            st.markdown("| Model | Prediction | F1 Score |")
            st.markdown("|---|---|---|")

            bert_label = "Genuine ✅" if bert_pred == 1 else "Fake 🚨"
            st.markdown(f"| DistilBERT (fine-tuned) | {bert_label} | 0.93 |")
            st.markdown(f"| TF-IDF + Logistic Regression | baseline | 0.90 |")

st.divider()
st.caption("Built by Sambhavi Gupta | DistilBERT fine-tuned on 40K Amazon reviews")