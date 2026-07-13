import streamlit as st
import pickle
import re
import nltk

# Ensure required NLTK resources are available (download only if missing)
for resource_path, resource_name in [('corpora/stopwords','stopwords'), ('corpora/wordnet','wordnet')]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK variables
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

# Load files
with open('vector.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("IMDB Movie Sentiment Analysis")
st.write("Enter a movie review below to see if it is Positive or Negative.")

user_input = st.text_area("Review Text", "")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned]).toarray()
        prediction = model.predict(vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        color = "green" if sentiment == "Positive" else "red"
        st.markdown(f"### Prediction: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")
