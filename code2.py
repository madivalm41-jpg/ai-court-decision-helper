# app_court_final_improved.py
"""
Court Judgment Predictor — BERT-enhanced with 50:50 downsampling (deletes extra rows)
- Loads CSV dataset (NyayaAnumana or similar)
- Cleans text, extracts statutes
- Balances dataset 50:50 by DOWN-sampling majority class (rows deleted)
- Optionally fine-tunes a BERT classifier (HuggingFace transformers)
- Keeps TF-IDF + Stacking fallback and sentence-transformer retrieval
- Streamlit UI for input, prediction, similar-case retrieval, PDF report
"""

import os
import re
import base64
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Optional / heavy dependencies
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    S2V_AVAILABLE = True
except Exception:
    S2V_AVAILABLE = False

# Transformers / PyTorch
try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------- CONFIG ----------------
DATASET_PATH = r"C:\Users\Hp\OneDrive\Desktop\Nayaanuman\CJPE_ext_SCI_HCs_Tribunals_daily_orders_single.csv"
CLEANED_DATASET = "court_cleaned_dataset.csv"
VECT_FILE = "court_tfidf_vect.joblib"
EMB_FILE = "s2v_embeddings.npy"
EMB_INDEX_FILE = "emb_index.joblib"
MODEL_FILE = "court_stack_model.joblib"
BERT_MODEL_DIR = "bert_finetuned_model"
BERT_MODEL_NAME_DEFAULT = "bert-base-uncased"
S2V_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- UTILITIES ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, low_memory=False)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"&amp;", " and ", s)
    s = re.sub(r"[^0-9A-Za-z\.\,\!\?\:\;\-\s/]", " ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_statutes(text: str):
    if not isinstance(text, str) or not text:
        return []
    pats = [
        r'\bsection\s+\d+[A-Za-z0-9\-]*', r'\bsec\.?\s\d+[A-Za-z0-9\-]*', r'\bs\.\s?\d+[A-Za-z0-9\-]*',
        r'\bipc\s*\d+[A-Za-z0-9\-]*', r'\barticle\s+\d+[A-Za-z0-9\-]*'
    ]
    found = set()
    for p in pats:
        for m in re.finditer(p, text.lower()):
            found.add(m.group().replace("  ", " ").strip())
    return list(found)

def map_labels_to_binary(series):
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        'accepted':1,'allow':1,'allowed':1,'allowed in part':1,'yes':1,'petitioner':1,'1':1,'1.0':1,
        'rejected':0,'dismissed':0,'dismiss':0,'no':0,'respondent':0,'0':0,'0.0':0,'refused':0
    }
    out = s.map(mapping)
    # fallback: if numeric and not mapped
    if out.isna().all():
        try:
            out = series.astype(int)
        except Exception:
            out = pd.Series(np.where(s.isin(['1','0']), s.astype(int), np.nan), index=series.index)
    return out.astype('Int64')

# ---------------- LOAD & CLEAN ----------------
@st.cache_data(show_spinner=False)
def load_and_prepare(path=DATASET_PATH, sample_limit=None):
    df = safe_read_csv(path)
    # identify text col
    text_col = None
    for c in df.columns:
        if c.lower() in ("judgment_text","judgement_text","text","case_text","facts","judgement"):
            text_col = c; break
    if text_col is None:
        for c in df.columns:
            if "text" in c.lower() or "judg" in c.lower() or "fact" in c.lower():
                text_col = c; break
    if text_col is None:
        raise RuntimeError("No text-like column found. Rename main text column to 'judgment_text' or similar.")
    # identify label col
    label_col = None
    for c in df.columns:
        if c.lower() in ("label","decision","outcome","result"):
            label_col = c; break
    if label_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 10:
                label_col = c; break
    if label_col is None:
        raise RuntimeError("No label-like column found (label/outcome/decision).")
    # optional sample
    if sample_limit:
        df = df.sample(n=min(sample_limit,len(df)), random_state=42).reset_index(drop=True)
    # keep text + label first
    df = df[[text_col, label_col] + [c for c in df.columns if c not in (text_col,label_col)]].copy()
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates()
    # clean text
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    # map label
    df[label_col] = map_labels_to_binary(df[label_col])
    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)
    # statutes & short summary
    df['statutes'] = df[text_col].apply(lambda t: "|".join(extract_statutes(t)))
    df['short_summary'] = df[text_col].apply(lambda t: " ".join(re.split(r'(?<=[.!?]) +', t)[:3]))
    # save cleaned
    try:
        df.to_csv(CLEANED_DATASET, index=False)
    except Exception:
        pass
    return df, text_col, label_col

# ---------------- BALANCING (DOWNSAMPLE majority) ----------------
def balance_dataset_downsample(df, label_col):
    """
    Balance dataset 50:50 by deleting rows from the majority class.
    Keeps only min_count rows per class (random_sample).
    """
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return df.copy()
    min_count = counts.min()
    df_balanced = (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=42))
          .reset_index(drop=True)
    )
    return df_balanced

# ---------------- TF-IDF & classical model ----------------
def build_tfidf(df, text_col, max_features=100000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,3), analyzer='word', min_df=3)
    X_tfidf = vect.fit_transform(df[text_col].values)
    joblib.dump(vect, VECT_FILE)
    return vect, X_tfidf

def train_stack_model(X_tfidf, y, use_xgb=True):
    base_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=10000)
    estimators = [('lr', base_lr)]
    if use_xgb and XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
        estimators.append(('xgb', xgb))
    final_est = LogisticRegression(max_iter=10000, class_weight='balanced')
    stack = StackingClassifier(estimators=estimators, final_estimator=final_est, cv=3, n_jobs=-1, passthrough=False)
    stack.fit(X_tfidf, y)
    joblib.dump(stack, MODEL_FILE)
    return stack

# ---------------- sentence embeddings & retrieval ----------------
def build_sentence_embeddings(df, text_col, model_name=S2V_MODEL_NAME):
    if not S2V_AVAILABLE:
        return None, None
    s2v = SentenceTransformer(model_name)
    emb = s2v.encode(df[text_col].tolist(), show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_FILE, emb)
    return s2v, emb

def build_knn_index(embeddings, n_neighbors=50):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='auto').fit(embeddings)
    joblib.dump(nbrs, EMB_INDEX_FILE)
    return nbrs

def get_similar_cases(query_emb, nbrs, df, topk=5):
    dists, idx = nbrs.kneighbors([query_emb], n_neighbors=topk)
    idx = idx[0].tolist()
    return df.iloc[idx]

# ---------------- BERT fine-tuning utilities ----------------
if TRANSFORMERS_AVAILABLE:
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(int(self.labels[idx]))
            return item

    def fine_tune_bert(df, text_col, label_col, model_name, output_dir=BERT_MODEL_DIR,
                       epochs=2, batch_size=8, learning_rate=2e-5):
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.1, stratify=labels, random_state=42)
        train_enc = {k: [v[i] for i in train_idx] for k, v in encodings.items()}
        val_enc = {k: [v[i] for i in val_idx] for k, v in encodings.items()}
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        train_dataset = TextDataset(train_enc, train_labels)
        val_dataset = TextDataset(val_enc, val_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            push_to_hub=False,
            fp16=torch.cuda.is_available()
        )
        def compute_metrics(eval_pred):
            preds, labels_np = eval_pred
            preds = np.argmax(preds, axis=1)
            acc = (preds == labels_np).astype(np.float32).mean().item()
            return {"accuracy": acc}
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir

    def load_bert_model(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model

    def bert_predict_proba(tokenizer, model, texts):
        enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
        if torch.cuda.is_available():
            model.to('cuda')
            enc = {k: v.to('cuda') for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

# ---------------- PDF report ----------------
def build_pdf(petitioner, respondent, extra, winner, confidence, summary, top_terms, statutes, similar_cases_df, label_col):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("AI Judgment Report", styles['Title']))
    story.append(Paragraph(f"Date: {datetime.now().isoformat()}", styles['Normal']))
    story.append(Spacer(1,6))
    story.append(Paragraph(f"<b>Predicted Winner:</b> {winner} (confidence: {confidence:.2%})", styles['Heading2']))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Concise Summary:</b>", styles['Heading3']))
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Top contributing terms:</b>", styles['Normal']))
    story.append(Paragraph(", ".join([f"{t}:{v:+.3f}" for t,v in top_terms]) if top_terms else "N/A", styles['Normal']))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Suggested Statutes:</b>", styles['Normal']))
    story.append(Paragraph(statutes or "N/A", styles['Normal']))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>Similar Cases (top):</b>", styles['Heading3']))
    for idx, row in similar_cases_df.head(5).iterrows():
        title = row.get('case_title') or f"Case idx {idx}"
        lab = row.get(label_col) if label_col in row.index else "N/A"
        story.append(Paragraph(f"- {title} | label={lab}", styles['Normal']))
    doc.build(story)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Court Judgment Predictor — BERT + Balanced", layout="wide")
st.title("⚖ Court Judgment Predictor — BERT + 50:50 downsampling")

# Sidebar controls
st.sidebar.markdown("## Options")
use_bert = st.sidebar.checkbox("Use BERT (fine-tune)", value=False)
bert_model_name = st.sidebar.text_input("BERT model name (HuggingFace)", value=BERT_MODEL_NAME_DEFAULT)
bert_epochs = st.sidebar.number_input("BERT epochs", min_value=1, max_value=10, value=2, step=1)
bert_batch = st.sidebar.number_input("BERT batch size", min_value=1, max_value=32, value=8, step=1)
bert_lr = st.sidebar.number_input("BERT learning rate", value=2e-5, format="%.6f", step=1e-6)
sample_limit = st.sidebar.number_input("Sample size (0=all)", value=20000, step=1000)

# Load & prepare dataset
with st.spinner("Loading and cleaning dataset..."):
    try:
        df, text_col, label_col = load_and_prepare(DATASET_PATH, sample_limit=sample_limit if sample_limit>0 else None)
        st.sidebar.info(f"Loaded {len(df)} rows — text col: {text_col} label col: {label_col}")
    except Exception as e:
        st.error(str(e))
        st.stop()

# Balance by deleting extra rows (downsample majority)
with st.spinner("Balancing dataset to 50:50 by downsampling majority class..."):
    before_counts = df[label_col].value_counts().to_dict()
    df = balance_dataset_downsample(df, label_col)
    after_counts = df[label_col].value_counts().to_dict()
    st.sidebar.success(f"Balanced dataset: before={before_counts} → after={after_counts}")

# Build / load TF-IDF
if not os.path.exists(VECT_FILE):
    with st.spinner("Building TF-IDF vectorizer..."):
        vect, X_tfidf = build_tfidf(df, text_col, max_features=100000)
else:
    vect = joblib.load(VECT_FILE)
    X_tfidf = vect.transform(df[text_col].values)

# Sentence embeddings (optional)
s2v, embeddings = None, None
if S2V_AVAILABLE:
    if not os.path.exists(EMB_FILE):
        with st.spinner("Building sentence-transformer embeddings..."):
            s2v, embeddings = build_sentence_embeddings(df, text_col)
    else:
        embeddings = np.load(EMB_FILE)
        try:
            s2v = SentenceTransformer(S2V_MODEL_NAME)
        except Exception:
            s2v = None

# KNN index (optional)
nbrs = None
if embeddings is not None and not os.path.exists(EMB_INDEX_FILE):
    with st.spinner("Building KNN index..."):
        nbrs = build_knn_index(embeddings, n_neighbors=50)
else:
    if os.path.exists(EMB_INDEX_FILE):
        try:
            nbrs = joblib.load(EMB_INDEX_FILE)
        except Exception:
            nbrs = None
    elif embeddings is not None:
        nbrs = build_knn_index(embeddings, n_neighbors=50)

# Decide which classifier to use
bert_tokenizer = None
bert_model = None
if use_bert:
    if not TRANSFORMERS_AVAILABLE:
        st.warning("Transformers/torch not installed — cannot fine-tune BERT. Please install to enable BERT.")
        use_bert = False
    else:
        # if already fine-tuned model folder exists, try to load
        if os.path.exists(os.path.join(BERT_MODEL_DIR, "config.json")):
            try:
                with st.spinner("Loading saved fine-tuned BERT model..."):
                    bert_tokenizer, bert_model = load_bert_model(BERT_MODEL_DIR)
                    st.sidebar.success("Loaded fine-tuned BERT model.")
            except Exception:
                bert_tokenizer, bert_model = None, None
        if bert_model is None:
            # fine-tune now (might take long on CPU)
            try:
                with st.spinner("Fine-tuning BERT (this may take time)..."):
                    model_dir = fine_tune_bert(df, text_col, label_col,
                                              model_name=bert_model_name,
                                              output_dir=BERT_MODEL_DIR,
                                              epochs=int(bert_epochs),
                                              batch_size=int(bert_batch),
                                              learning_rate=float(bert_lr))
                    bert_tokenizer, bert_model = load_bert_model(model_dir)
                    st.sidebar.success("BERT fine-tuned and loaded.")
            except Exception as e:
                st.error(f"BERT fine-tune failed: {e}")
                use_bert = False

# If not using BERT, ensure classical model exists or train it
model = None
if not use_bert:
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Training classical stacked model..."):
            y = df[label_col].astype(int).values
            model = train_stack_model(X_tfidf, y, use_xgb=True)
            try:
                Xtr, Xte, ytr, yte = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)
                preds = model.predict(Xte)
                acc = accuracy_score(yte, preds)
                st.sidebar.success(f"Trained classical model — held-out accuracy: {acc*100:.2f}%")
            except Exception:
                st.sidebar.success("Trained classical model.")
    else:
        model = joblib.load(MODEL_FILE)
        st.sidebar.info("Loaded classical model from disk.")

# ---------------- UI: Input & Predict ----------------
left, right = st.columns([2,1])
with left:
    st.subheader("Input case facts")
    petitioner = st.text_area("Petitioner facts", height=200)
    respondent = st.text_area("Respondent facts", height=200)
    extra = st.text_area("Additional facts / sections (optional)", height=120)
    if st.button("Predict & Generate Report"):
        combined = " ".join([petitioner, respondent, extra]).strip()
        if len(combined) < 50:
            st.error("Please provide more detailed facts (>= 50 chars).")
        else:
            cleaned = clean_text(combined)
            # Predict with BERT if enabled
            if use_bert and bert_model is not None and bert_tokenizer is not None:
                try:
                    probs = bert_predict_proba(bert_tokenizer, bert_model, [cleaned])[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[1]) if len(probs)>1 else 0.0
                    winner = "Petitioner" if pred==1 else "Respondent"
                    st.success(f"[BERT] Predicted winner: {winner}")
                    st.info(f"Estimated confidence (petitioner): {conf:.2%}")
                except Exception as e:
                    st.error(f"BERT prediction failed: {e}")
                    use_bert = False
                    pred = None
                    conf = 0.0
            else:
                # classical
                X_new_tfidf = vect.transform([cleaned])
                pred = int(model.predict(X_new_tfidf)[0])
                probs = None
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_new_tfidf)[0]
                except Exception:
                    probs = None
                conf = float(probs[1]) if probs is not None and len(probs)>1 else 0.0
                winner = "Petitioner" if pred==1 else "Respondent"
                st.success(f"[Classical] Predicted winner: {winner}")
                if conf:
                    st.info(f"Estimated confidence (petitioner): {conf:.2%}")

            # concise summary
            st.markdown("*Concise summary (first 3 sentences of input)*")
            st.write(" ".join(re.split(r'(?<=[.!?]) +', cleaned)[:3]))

            # top contributing terms (classical logistic fallback)
            top_terms = []
            if not use_bert:
                try:
                    if hasattr(model, "named_estimators_"):
                        for name, est in model.named_estimators_.items():
                            if isinstance(est, LogisticRegression):
                                feat_names = vect.get_feature_names_out()
                                coef = est.coef_[0]
                                x = X_new_tfidf.toarray().ravel()
                                contrib = coef * x
                                idx = np.argsort(-np.abs(contrib))[:12]
                                top_terms = [(feat_names[i], float(contrib[i])) for i in idx if contrib[i]!=0]
                                break
                except Exception:
                    top_terms = []

            if top_terms:
                st.markdown("*Top contributing terms*")
                for t,v in top_terms:
                    st.write(f"- {t} : {v:+.4f}")

            # statutes
            statutes = extract_statutes(combined)
            st.markdown("*Suggested statutes found in text*")
            if statutes:
                for s in statutes:
                    st.write(f"- {s}")
            else:
                st.write("- No statute pattern found; showing rule-of-thumb laws")
                st.write(", ".join(["Indian Evidence Act", "Indian Contract Act", "Transfer of Property Act"]))

            # similar cases (if embeddings available)
            similar_df = pd.DataFrame()
            if nbrs is not None and s2v is not None:
                try:
                    q_emb = s2v.encode([cleaned], convert_to_numpy=True)[0]
                    similar_df = get_similar_cases(q_emb, nbrs, df, topk=5)
                    st.markdown("*Similar past cases (top matches)*")
                    for i, row in similar_df.head(5).iterrows():
                        title = row.get('case_title') or f"case_{i}"
                        st.write(f"- {title} | short: {row.get('short_summary')[:300]} | label: {row.get(label_col)}")
                except Exception:
                    st.write("Error computing similar cases.")
            else:
                st.write("Similar cases not available (embeddings/KNN disabled).")

            # PDF
            pdf_b64 = build_pdf(petitioner, respondent, extra, winner, conf or 0.0,
                                " ".join(re.split(r'(?<=[.!?]) +', cleaned)[:3]),
                                top_terms, "|".join(statutes), similar_df if not similar_df.empty else pd.DataFrame(), label_col)
            st.markdown(f'<a href="data:application/pdf;base64,{pdf_b64}" download="Judgment_Report.pdf">📥 Download Judgment Report (PDF)</a>', unsafe_allow_html=True)

with right:
    st.subheader("Dataset & Model Info")
    st.write(f"Rows loaded (after cleaning & balancing): {len(df)}")
    st.write("Label distribution (after balancing)")
    try:
        st.bar_chart(df[label_col].value_counts())
    except Exception:
        st.write(df[label_col].value_counts())
    st.markdown("*Sample*")
    try:
        st.dataframe(df[[text_col, 'short_summary', 'statutes', label_col]].head(5))
    except Exception:
        st.dataframe(df.head(5))
    if use_bert and os.path.exists(BERT_MODEL_DIR):
        st.success("BERT model available.")
    if os.path.exists(MODEL_FILE):
        st.success("Classical model ready.")

st.caption("Notes: Models provide probabilistic support and are not a substitute for legal advice.")
