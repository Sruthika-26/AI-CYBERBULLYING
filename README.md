# 🛡️ AI Cyberbullying Detection with HateBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io/)
[![HateBERT](https://img.shields.io/badge/model-HateBERT-green.svg)](https://huggingface.co/GroNLP/hateBERT)

An intelligent AI-powered cyberbullying detection system that combines the power of HateBERT transformer model with advanced sentiment analysis and gender-targeted language detection. The system provides real-time analysis through an interactive Streamlit web interface.

## 🎯 Overview

This project implements a multi-faceted approach to cyberbullying detection that goes beyond simple text classification. It incorporates:

- **HateBERT Model**: Pre-trained transformer model specifically designed for hate speech detection
- **Multilingual Support**: Automatic language detection and translation to English
- **Gender-Targeted Analysis**: Specialized detection of gender-based harassment
- **Sentiment Analysis**: VADER sentiment analysis for emotional context
- **Named Entity Recognition**: SpaCy-powered detection of person names and targets
- **Interactive Web Interface**: Real-time analysis through Streamlit dashboard

## ✨ Key Features

### 🤖 Advanced AI Detection
- **HateBERT Integration**: Leverages GroNLP's HateBERT model for state-of-the-art hate speech detection
- **Smart Thresholding**: Dynamic confidence thresholds based on sentiment context
- **Mixed Sentiment Detection**: Identifies subtle toxicity hidden in seemingly positive messages

### 🌍 Multilingual Capabilities
- **Automatic Language Detection**: Uses `langdetect` to identify input language
- **Real-time Translation**: Google Translator integration for non-English content
- **Cross-cultural Analysis**: Works with diverse linguistic patterns

### 👥 Gender-Targeted Detection
- **Comprehensive Name Recognition**: Supports both Western and Indian name databases
- **Context-Aware Analysis**: Understands cultural and linguistic nuances
- **Multi-gender Support**: Detects targeting of all gender identities

### 📊 Sentiment & Context Analysis
- **VADER Sentiment Analysis**: Nuanced emotional tone detection
- **Direct Insult Detection**: Keyword-based identification of explicit harassment
- **Cultural Sensitivity**: Tailored for diverse social media contexts

## 🚀 Demo

### Screenshots of the System in Action

#### 1. Mixed Sentiment Toxicity Detection
![Mixed Sentiment Detection](Images/AI%201.png)
*Demonstrating detection of subtle harassment hidden in seemingly positive language - "Hey beautiful, you're still really dumb though"*

#### 2. Direct Cyberbullying with Gender Analysis
![Direct Cyberbullying](Images/AI%202.png)
*Clear cyberbullying detection with male-targeted language identification - "you're such a loser and an idiot, Brother"*

#### 3. Multilingual Support
![Multilingual Detection](Images/AI%203.png)
*Hindi text analysis with automatic translation: "तुम बहुत बुरे हो और मूर्ख हो" → "You are very bad and stupid"*

#### 4. Positive Content with Gender Detection
![Positive Content](Images/AI%204.png)
*Correctly identifying positive content while detecting female-targeted context - "Great job on your project Diana! Really impressive work"*

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Interactive Web Interface)
- **AI Model**: HateBERT (GroNLP/hateBERT from Hugging Face)
- **NLP Libraries**: 
  - SpaCy (`en_core_web_sm`) for Named Entity Recognition
  - NLTK with VADER for sentiment analysis
- **Language Processing**: 
  - `langdetect` for language identification
  - `deep_translator` for Google Translate integration
- **Gender Detection**: `gender_guesser` + custom Indian names database
- **Data Processing**: Pandas, PyTorch, Transformers

## 📋 Requirements

Create a `requirements.txt` file with these dependencies:

```txt
streamlit>=1.28.0
pandas>=1.5.0
torch>=1.13.0
transformers>=4.21.0
langdetect>=1.0.9
deep-translator>=1.11.4
nltk>=3.8
spacy>=3.4.0
gender-guesser>=0.4.0
```

### Additional Requirements
```bash
# Download SpaCy English model
python -m spacy download en_core_web_sm

# Required data file
indian_names.csv  # Indian names database for gender detection
```

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Sruthika-26/AI-CYBERBULLYING.git
cd AI-CYBERBULLYING
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Models
```bash
# Download SpaCy English model
python -m spacy download en_core_web_sm

# NLTK data will be downloaded automatically when you first run the app
```

### 4. Prepare Data Files
Ensure you have the `indian_names.csv` file in your project root with the format:
```csv
name,gender
raj,male
priya,female
amit,male
sunita,female
```

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```
## 📁 Project Structure

```
AI-CYBERBULLYING/
│
├── app.py                 # Main Streamlit application
├── indian_names.csv       # Indian names database for gender detection
├── Images/                # Demo screenshots
│   ├── AI 1.png          # Mixed sentiment demo
│   ├── AI 2.png          # Direct cyberbullying demo
│   ├── AI 3.png          # Multilingual demo
│   └── AI 4.png          # Positive content demo
├── models_cache/          # Cached HateBERT model files (auto-created)
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## 🔍 Technical Highlights

### Intelligent Gender Detection
- **Multi-source Analysis**: Combines Western and Indian name databases
- **Linguistic Fallback**: Pattern-based gender guessing for unknown names
- **Context Awareness**: Considers surrounding words and entity relationships

### Advanced Preprocessing
- **Text Normalization**: Consistent cleaning pipeline
- **Multilingual Support**: Seamless translation workflow
- **Entity Extraction**: SpaCy-powered named entity recognition

### Robust Classification
- **Threshold Adaptation**: Dynamic confidence levels based on sentiment
- **Multi-factor Decision**: Combines AI predictions with rule-based patterns
- **Edge Case Handling**: Special detection for mixed sentiment scenarios

## 🌟 Unique Features

1. **Cultural Adaptation**: Includes Indian names database for better regional accuracy
2. **Subtle Toxicity Detection**: Identifies harassment hidden in positive language
3. **Real-time Translation**: Supports global social media content analysis
4. **Gender-Aware Analysis**: Specialized detection of gender-based harassment patterns
5. **Interactive Dashboard**: User-friendly interface for immediate testing and analysis

## 🚀 Quick Start

1. **Clone and install**:
   ```bash
   git clone https://github.com/Sruthika-26/AI-CYBERBULLYING.git
   cd AI-CYBERBULLYING
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Test the system** with sample texts and explore the three-dimensional analysis results!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **GroNLP Team** for the excellent HateBERT model
- **Hugging Face** for the transformers library and model hosting
- **SpaCy Team** for robust NLP capabilities
- **NLTK Contributors** for VADER sentiment analysis
- **Indian Names Dataset Contributors** for cultural inclusivity

## 📞 Contact

**Sruthika** - [@Sruthika-26](https://github.com/Sruthika-26)

Project Link: [https://github.com/Sruthika-26/AI-CYBERBULLYING](https://github.com/Sruthika-26/AI-CYBERBULLYING)

---

⭐ **Star this repository if you find it helpful!** ⭐
