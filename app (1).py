import streamlit as st
import pandas as pd
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from googletrans import Translator
import nltk
import spacy
import gender_guesser.detector as gender
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import os
import wget
model_path = os.path.join(os.getcwd(), "en_core_web_sm")
if not os.path.exists(model_path):
    url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    model_file = wget.download(url)
    os.system(f"pip install {model_file}")

nlp = spacy.load("en_core_web_sm")


s = SentimentIntensityAnalyzer()
d = gender.Detector()

indian_names_df = pd.read_csv("indian_names.csv")
indian_name_gender_map = dict(zip(indian_names_df['name'].str.lower(), indian_names_df['gender'].str.lower()))

model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", cache_dir="./models_cache")
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", cache_dir="./models_cache")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

translator = Translator()
def detect_translate(text):
    detected_lang = detect(text)
    if detected_lang != 'en':
        translated = translator.translate(text, dest='en').text
        return translated, detected_lang
    return text, detected_lang

def analyze_sentiment(text):
    scores = s.polarity_scores(text)
    if scores['compound'] > 0.05:
        return "😊 Positive"
    elif scores['compound'] < -0.05:
        return "😡 Negative"
    else:
        return "😐 Neutral"

def guess_gender_fallback(name):
    name = name.lower()
    if name.endswith(('a', 'i', 'u')):
        return "female"
    elif name.endswith(('n', 'k', 'm', 'd', 'r')):
        return "male"
    return "unknown"

def gender_based_detection(text):
    translated_text, _ = detect_translate(text)
    doc = nlp(translated_text)

    male_targets = {"boy", "man", "guy", "male", "dude", "bro"}
    female_targets = {"girl", "woman", "lady", "female", "chick", "gal"}
    female_insults = {"slut", "whore"}

    male_detected = False
    female_detected = False
    person_names = set()
    
    capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    person_names.update([name.strip().capitalize() for name in capitalized_words])

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_names.add(ent.text.strip().capitalize())
    for token in doc:
        if token.pos_ == "PROPN":
            person_names.add(token.text.strip().capitalize())

    for name in person_names:
        name_lower = name.lower()
        if name_lower in indian_name_gender_map:
            gender_guess = indian_name_gender_map[name_lower]
        else:
            gender_guess = d.get_gender(name)
            if gender_guess == 'unknown':
                gender_guess = guess_gender_fallback(name)
        if gender_guess in ["female", "mostly_female"]:
            female_detected = True
        elif gender_guess in ["male", "mostly_male"]:
            male_detected = True

    for token in doc:
        word = token.lemma_.lower()
        if word in male_targets:
            male_detected = True
        if word in female_targets or word in female_insults:
            female_detected = True
        if word == "bitch":
            if any(w in female_targets for w in [t.lemma_.lower() for t in doc]) or person_names:
                female_detected = True

    if female_detected and not male_detected:
        return "⚠️ Female-targeted language detected"
    elif male_detected and not female_detected:
        return "⚠️ Male-targeted language detected"
    elif male_detected and female_detected:
        return "⚠️ Mixed gender-targeted language detected"
    else:
        return "✅ No gender-targeted language detected"

def has_mixed_sentiment_toxicity(text):
    negative_keywords = ["bitch", "ugly", "stupid", "dumb", "idiot", "fat"]
    positive_keywords = ["beautiful", "smart", "cute", "nice", "lovely", "sweet"]
    text_lower = text.lower()
    has_negative = any(word in text_lower for word in negative_keywords)
    has_positive = any(word in text_lower for word in positive_keywords)
    scores = s.polarity_scores(text)
    if has_negative and has_positive and abs(scores['compound']) < 0.2:
        return True
    return False

def contains_direct_insults(text):
    insults = ["bitch", "ugly", "stupid", "dumb", "idiot", "fat", "loser"]
    return any(word in text.lower() for word in insults)

def predict(text):
    text, lang = detect_translate(text)
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    bullying_prob = probs[0][1].item()
    sentiment = analyze_sentiment(cleaned)
    gender_analysis = gender_based_detection(text)

    if has_mixed_sentiment_toxicity(cleaned):
        result = "🚨 Mixed Sentiment with Possible Toxicity"
    elif contains_direct_insults(cleaned):
        result = "🚨 Cyberbullying Detected"
    elif bullying_prob > 0.3 and sentiment != "😊 Positive":
        result = "🚨 Cyberbullying Detected"
    elif bullying_prob > 0.7:
        result = "🚨 Cyberbullying Detected"
    else:
        result = "✅ Not Cyberbullying"

    return result, sentiment, gender_analysis

st.title("🚨 AI Cyberbullying Detection with HateBERT")
input_text = st.text_area("Enter a comment or tweet:")

if st.button("🔍 Analyze"):
    with st.spinner("Analyzing..."):
        result, sentiment, gender_analysis = predict(input_text)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Prediction")
        st.markdown(f"<div class='report'>{result}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Sentiment")
        st.markdown(f"<div class='report'>{sentiment}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("#### Gender Analysis")
        st.markdown(f"<div class='report'>{gender_analysis}</div>", unsafe_allow_html=True)
