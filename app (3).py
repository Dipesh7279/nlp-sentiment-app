import streamlit as st
import pickle
import re
import nltk
import numpy as np
import requests

# Ensure required NLTK resources are available (download only if missing)
for resource_path, resource_name in [('corpora/stopwords','stopwords'), ('corpora/wordnet','wordnet')]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preserve negation words so "not", "no" etc. aren't removed
NEGATIONS = {"no", "nor", "not", "never", "n't"}
try:
    stop_words = set(stopwords.words('english')) - NEGATIONS
except LookupError:
    # fallback to a small stopword set if something goes wrong
    stop_words = {w for w in [
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
        'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
        'it','its','itself','they','them','their','theirs','themselves','what','which',
        'who','whom','this','that','these','those','am','is','are','was','were','be','been',
        'being','have','has','had','having','do','does','did','doing','a','an','the','and',
        'but','if','or','because','as','until','while','of','at','by','for','with','about',
        'against','between','into','through','during','before','after','above','below','to',
        'from','up','down','in','out','on','off','over','under','again','further','then','once',
        'here','there','when','where','why','how','all','any','both','each','few','more','most',
        'other','some','such','only','own','same','so','than','too','very','s','t','can','will',
        'just','don','should','now'
    ]}

lemmatizer = WordNetLemmatizer()

# Small contraction-expansion map to keep negation meaning
_CONTRACTION_MAP = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
}

def expand_contractions(text: str) -> str:
    for k, v in _CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text, flags=re.IGNORECASE)
    return text

def clean_text(text: str) -> str:
    # expand contractions first so "can't" -> "can not" and we keep "not"
    text = expand_contractions(text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    # keep only letters and spaces (apostrophes already handled)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

# Load model and vectorizer
with open('vector.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("IMDB Movie Sentiment Analysis")
st.write("Enter a movie review below to see if it is Positive or Negative.")

user_input = st.text_area("Review Text", "")
show_debug = st.checkbox("Show debug info")
skip_preprocessing = st.checkbox("Skip preprocessing (use raw text)")

# Model selection: local sklearn or Hugging Face Inference API
model_choice = st.radio("Choose model", ("sklearn (local)", "huggingface (HF Inference API)"))

# Helper to call HF Inference API
HF_DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def call_hf_api(text: str, hf_api_key: str, model_name: str = HF_DEFAULT_MODEL):
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    payload = {"inputs": text}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Response can be [{'label': 'POSITIVE', 'score': 0.99}] or similar
        if isinstance(data, dict) and data.get('error'):
            return None, f"HF API error: {data.get('error')}"
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            label = item.get('label')
            score = float(item.get('score', 0.0))
            return {'label': label, 'score': score}, None
        return None, "Unexpected HF response format"
    except Exception as e:
        return None, str(e)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        model_input_text = user_input if skip_preprocessing else cleaned

        if model_input_text.strip() == "":
            st.warning("Text was empty after preprocessing — try a longer input or adjust preprocessing.")

        # If user selected HF model, call HF API. Otherwise use local sklearn pipeline.
        predicted_class = None
        predicted_proba = None
        raw_label = None

        if model_choice == "huggingface (HF Inference API)":
            hf_api_key = st.secrets.get('HF_API_KEY') if hasattr(st, 'secrets') else None
            if not hf_api_key:
                hf_api_key = st.text_input('Enter Hugging Face API key', type='password')
            if not hf_api_key:
                st.error('Hugging Face API key is required for this option. Set HF_API_KEY in Streamlit secrets or enter it here.')
            else:
                result, err = call_hf_api(model_input_text, hf_api_key)
                if err:
                    st.error(f"Hugging Face inference failed: {err}\nFalling back to local model.")
                    model_choice = "sklearn (local)"
                else:
                    raw_label = result['label']
                    score = result['score']
                    # Map HF labels to sentiment
                    if raw_label.upper().startswith('POS') or raw_label.upper().startswith('LABEL_1'):
                        predicted_class = 1
                    else:
                        predicted_class = 0
                    predicted_proba = [1 - score, score] if predicted_class == 1 else [score, 1 - score]

        if model_choice == "sklearn (local)":
            # transform to sparse vector (no toarray) to preserve expected format
            vec = tfidf.transform([model_input_text])

            if show_debug:
                st.write("Using raw text:" if skip_preprocessing else "Using preprocessed text:")
                st.write(model_input_text)
                try:
                    nonzero = int(vec.nnz)
                except Exception:
                    try:
                        nonzero = int(np.count_nonzero(vec.toarray()))
                    except Exception:
                        nonzero = None
                st.write("Non-zero TF-IDF features:", nonzero)
                if hasattr(model, 'classes_'):
                    st.write("Model classes:", model.classes_)

            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(vec)[0]
                    # pick the class with highest probability
                    idx = int(np.argmax(probs))
                    predicted_class = int(model.classes_[idx]) if hasattr(model, 'classes_') else int(idx)
                    predicted_proba = probs.tolist()
                else:
                    preds = model.predict(vec)
                    predicted_class = int(preds[0])
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                raise

        # Map predicted_class to human-readable sentiment
        sentiment = None
        if predicted_class is not None:
            # If model.classes_ contains strings, handle accordingly
            try:
                sentiment = "Positive" if int(predicted_class) == 1 else "Negative"
            except Exception:
                # handle string labels like 'POS'/'NEG' or 'pos'/'neg'
                if isinstance(predicted_class, str) and predicted_class.lower().startswith('pos'):
                    sentiment = 'Positive'
                else:
                    sentiment = 'Negative'

        color = "green" if sentiment == "Positive" else "red"

        if show_debug and raw_label is not None:
            st.write("Hugging Face raw label:", raw_label)
            st.write("Hugging Face score:", score)
        if show_debug and predicted_proba is not None:
            st.write("Predicted probabilities:", predicted_proba)

        if sentiment is not None:
            st.markdown(f"### Prediction: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        else:
            st.error('Could not determine sentiment from model output.')
