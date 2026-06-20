# TruthCheck 🔍
### Amazon Review Authenticity Detector
<img width="541" height="619" alt="TruthCheck Demo" src="https://github.com/user-attachments/assets/5f5dbd53-0d83-41b6-9759-c8b40f255910" />

A machine learning pipeline that detects suspicious Amazon reviews using fine-tuned DistilBERT, benchmarked against a TF-IDF + Logistic Regression baseline.

## Results

| Model | F1 Score |
|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.90 |
| DistilBERT (fine-tuned) | 0.93 |

## How It Works

1. **Preprocessing** — Combines review title + body, cleans and tokenizes text
2. **Baseline** — TF-IDF vectorization (10K features, bigrams) + Logistic Regression
3. **Fine-tuning** — DistilBERT fine-tuned for 2 epochs on 40K labeled Amazon reviews
4. **Inference** — End-to-end pipeline with confidence scoring via Streamlit

## Dataset

Amazon Customer Reviews Dataset — 40,000 reviews (balanced, 20K per class).  
Source: Kaggle (`kritanjalijain/amazon-reviews`)

> **Note:** Labels use sentiment as a proxy for authenticity (positive reviews as genuine, negative as suspicious). True fake review detection requires proprietary human-labeled data. This is a proof-of-concept implementation.

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- scikit-learn, pandas, NumPy
- Streamlit (demo UI)
- Google Colab (T4 GPU for training)

## Run Locally

```bash
git clone https://github.com/SambhaviGupta/TruthCheck.git
cd TruthCheck
pip install -r requirements.txt
streamlit run app.py
```

> Model weights (~256MB) are not included in this repo.  
> Download from [Google Drive](https://drive.google.com/drive/folders/15Qp1BJNYFbG4NdtjUrohQ0_Cmh47z4T0?usp=sharing) and place in `truthcheck_model/` folder.

## Author

**Sambhavi Gupta** — B.Tech CSE (AI-ML), PSIT Kanpur
