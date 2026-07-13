import streamlit as st
import pickle
import re
import nltk
import numpy as np

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

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        # choose input based on skip_preprocessing
        model_input_text = user_input if skip_preprocessing else cleaned

        if model_input_text.strip() == "":
            st.warning("Text was empty after preprocessing — try a longer input or adjust preprocessing.")

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

        # Use predict_proba when available to determine predicted class robustly
        predicted_class = None
        predicted_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(vec)
                class_idx = int(np.argmax(probs, axis=1)[0])
                predicted_class = model.classes_[class_idx]
                predicted_proba = probs[0].tolist()
            else:
                preds = model.predict(vec)
                predicted_class = preds[0]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            raise

        # Determine sentiment mapping (assume class '1' -> Positive)
        sentiment = "Positive" if int(predicted_class) == 1 else "Negative"

        color = "green" if sentiment == "Positive" else "red"
        if predicted_proba is not None and show_debug:
            st.write("Predicted probabilities:", predicted_proba)
        st.markdown(f"### Prediction: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
