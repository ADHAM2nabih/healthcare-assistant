import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load models
@st.cache_resource
def load_models():
    # Load label encoder
    label_encoder = joblib.load('label_encoder.joblib')
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load LSTM model
    model = load_model('lstm_model.h5')
    
    return label_encoder, tokenizer, model

label_encoder, tokenizer, model = load_models()

# Text preprocessing functions
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])

def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return lemmatize_text(text)

# Prediction function
def predict_specialty(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get class name and probability
    class_name = label_encoder.inverse_transform(predicted_class)[0]
    probability = np.max(prediction)
    
    return class_name, probability

# Streamlit UI
st.title("Medical Specialty Classification")
st.write("This app predicts the medical specialty of a given medical transcript.")

# Input text area
user_input = st.text_area("Enter medical transcription text:", 
                         "Patient presents with fever and cough for 3 days...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            specialty, confidence = predict_specialty(user_input)
            
            # Display results in a more compact format
            col1, col2 = st.columns(2)
            with col1:
                st.header("Predicted Specialty", specialty)

