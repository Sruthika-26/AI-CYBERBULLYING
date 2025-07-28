import streamlit as st
import pandas as pd
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
import spacy
import gender_guesser.detector as gender
from nltk.sentiment import SentimentIntensityAnalyzer
import en_core_web_sm

try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass
    
nlp = en_core_web_sm.load()
sentiment_analyzer = SentimentIntensityAnalyzer()
gender_detector = gender.Detector()

@st.cache_resource
def load_hate_detection_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", cache_dir="./models_cache")
        model = AutoModelForSequenceClassification.from_pretrained(
            "GroNLP/hateBERT", 
            cache_dir="./models_cache", 
            torch_dtype=torch.float32
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_hate_detection_model()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def translate_to_english(text):
    try:
        detected_language = detect(text)
        if detected_language != 'en':
            translator = GoogleTranslator(source='auto', target='en')
            translated_text = translator.translate(text)
            return translated_text, detected_language
        return text, detected_language
    except:
        return text, 'en'

def load_indian_names_database():
    try:
        df = pd.read_csv("indian_names.csv")
        df[['name', 'gender']] = df['name,gender'].astype(str).str.split(",", expand=True)
        df['name'] = df['name'].str.lower()
        name_to_gender = dict(zip(df['name'], df['gender']))
        return name_to_gender
    except:
        return {}

indian_names_db = load_indian_names_database()

def get_sentiment_analysis(text):
    """Analyze sentiment of the text"""
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score > 0.05:
        return "😊 Positive"
    elif compound_score < -0.05:
        return "😡 Negative"
    else:
        return "😐 Neutral"

def predict_gender_from_name(name):
    name = name.lower()
    if name in indian_names_db:
        return indian_names_db[name]

    gender_guess = gender_detector.get_gender(name)
    if gender_guess != 'unknown':
        return gender_guess
    if name.endswith(('a', 'i', 'u')):
        return "female"
    elif name.endswith(('n', 'k', 'm', 'd', 'r')):
        return "male"
    
    return "unknown"

def analyze_gender_targeting(text):
    translated_text, _ = translate_to_english(text)
    doc = nlp(translated_text)

   
    male_indicators = {"boy", "man", "guy", "male", "dude", "bro"}
    female_indicators = {"girl", "woman", "lady", "female", "chick", "gal"}
    female_specific_insults = {"slut", "whore"}

    male_content_detected = False
    female_content_detected = False
    detected_names = set()
    
    
    capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    detected_names.update([name.strip().capitalize() for name in capitalized_words])

  
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            detected_names.add(entity.text.strip().capitalize())
    
    for token in doc:
        if token.pos_ == "PROPN":
            detected_names.add(token.text.strip().capitalize())


    for name in detected_names:
        predicted_gender = predict_gender_from_name(name)
        if predicted_gender in ["female", "mostly_female"]:
            female_content_detected = True
        elif predicted_gender in ["male", "mostly_male"]:
            male_content_detected = True

  
    for token in doc:
        word = token.lemma_.lower()
        if word in male_indicators:
            male_content_detected = True
        if word in female_indicators or word in female_specific_insults:
            female_content_detected = True
        if word == "bitch":
            if any(w in female_indicators for w in [t.lemma_.lower() for t in doc]) or detected_names:
                female_content_detected = True

  
    if female_content_detected and not male_content_detected:
        return "⚠️ Female-targeted language detected"
    elif male_content_detected and not female_content_detected:
        return "⚠️ Male-targeted language detected"
    elif male_content_detected and female_content_detected:
        return "⚠️ Mixed gender-targeted language detected"
    else:
        return "✅ No gender-targeted language detected"

def check_mixed_sentiment_toxicity(text):
    """Check for mixed sentiment with potential toxicity"""
    negative_terms = ["bitch", "ugly", "stupid", "dumb", "idiot", "fat"]
    positive_terms = ["beautiful", "smart", "cute", "nice", "lovely", "sweet"]
    
    text_lower = text.lower()
    has_negative_terms = any(term in text_lower for term in negative_terms)
    has_positive_terms = any(term in text_lower for term in positive_terms)

    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    
    if has_negative_terms and has_positive_terms and abs(sentiment_scores['compound']) < 0.2:
        return True
    return False

def contains_direct_harmful_language(text):
    harmful_terms = ["bitch", "ugly", "stupid", "dumb", "idiot", "fat", "loser"]
    return any(term in text.lower() for term in harmful_terms)

def make_cyberbullying_prediction(text):
    """Main function to predict cyberbullying"""
    if not text or not text.strip():
        return "Please enter some text to analyze", "😐 Neutral", "✅ No analysis available"
    
    translated_text, original_language = translate_to_english(text)
    
    cleaned_text = preprocess_text(translated_text)
    
    bullying_probability = 0.5 
    
    if tokenizer and model:
        try:
            inputs = tokenizer(
                cleaned_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                if hasattr(probabilities[0][1], 'item'):
                    bullying_probability = probabilities[0][1].item()
                else:
                    bullying_probability = float(probabilities[0][1])
        except Exception as e:
            st.warning(f"Model prediction failed, using rule-based approach: {e}")
            bullying_probability = 0.5

    sentiment_result = get_sentiment_analysis(cleaned_text)
    gender_analysis_result = analyze_gender_targeting(text)
    
    if check_mixed_sentiment_toxicity(cleaned_text):
        final_result = "🚨 Mixed Sentiment with Possible Toxicity"
    elif contains_direct_harmful_language(cleaned_text):
        final_result = "🚨 Cyberbullying Detected"
    elif bullying_probability > 0.3 and sentiment_result != "😊 Positive":
        final_result = "🚨 Cyberbullying Detected"
    elif bullying_probability > 0.7:
        final_result = "🚨 Cyberbullying Detected"
    else:
        final_result = "✅ Not Cyberbullying"

    return final_result, sentiment_result, gender_analysis_result

# Streamlit UI
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🕵️ AI-Powered Cyberbullying Detection System")
st.markdown("*Using advanced NLP and HateBERT model for comprehensive text analysis*")

# Create input section
st.markdown("### 📝 Enter Text for Analysis")
user_input = st.text_area(
    "Paste a comment, tweet, or message below:",
    placeholder="Enter the text you want to analyze for cyberbullying...",
    height=100
)

# Analysis button
if st.button("🔍 Analyze Text", type="primary"):
    if user_input:
        with st.spinner("🤖 Analyzing text... Please wait"):
            # Get predictions
            prediction_result, sentiment_result, gender_result = make_cyberbullying_prediction(user_input)
        
        # Display results in columns
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Cyberbullying Detection")
            if "🚨" in prediction_result:
                st.error(prediction_result)
            elif "⚠️" in prediction_result:
                st.warning(prediction_result)
            else:
                st.success(prediction_result)
        
        with col2:
            st.markdown("#### 💭 Sentiment Analysis")
            if "😡" in sentiment_result:
                st.error(sentiment_result)
            elif "😐" in sentiment_result:
                st.info(sentiment_result)
            else:
                st.success(sentiment_result)
        
        with col3:
            st.markdown("#### 👥 Gender Targeting Analysis")
            if "⚠️" in gender_result:
                st.warning(gender_result)
            else:
                st.success(gender_result)
                
        st.markdown("---")
        st.markdown("#### ℹ️ How it works")
        st.info("""
        This system uses:
        - **HateBERT Model**: Advanced transformer model trained on hate speech detection
        - **Sentiment Analysis**: VADER sentiment analyzer for emotional tone
        - **Gender Analysis**: Detection of gender-targeted language patterns
        - **Multi-language Support**: Automatic translation for non-English text
        - **Rule-based Checks**: Additional patterns for comprehensive analysis
        """)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛡️ Built for creating safer online spaces | "
    "⚡ Powered by HateBERT & Advanced NLP"
    "</div>", 
    unsafe_allow_html=True
)
