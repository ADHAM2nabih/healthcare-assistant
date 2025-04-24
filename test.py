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
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_resource
def load_models():
    specialty_label_encoder = joblib.load('label_encoder.joblib')
    with open('tokenizer.pkl', 'rb') as handle:
        specialty_tokenizer = pickle.load(handle)
    specialty_model = load_model('lstm_model.h5')
    
    disease_label_encoder = joblib.load('label_encoder_machine.joblib')
    disease_model = joblib.load('mlp_pipeline_model.pkl')
    
    return (specialty_label_encoder, specialty_tokenizer, specialty_model, 
            disease_label_encoder, disease_model)

(specialty_le, specialty_tokenizer, specialty_model,
 disease_le, disease_model) = load_models()

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

def predict_specialty(text):
    """Predict medical specialty using LSTM model"""
    sequence = specialty_tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = specialty_model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    class_name = specialty_le.inverse_transform(predicted_class)[0]
    probability = np.max(prediction)
    return class_name, probability

def predict_disease(text):
    """Predict disease using MLP model"""
    prediction = disease_model.predict([processed_text])
    probabilities = disease_model.predict_proba([processed_text])
    class_name = disease_le.inverse_transform(prediction)[0]
    probability = np.max(probabilities)
    return class_name, probability

st.title("Medical Diagnosis Assistant")
st.write("This app predicts both the medical specialty and disease based on symptoms description.")

user_input = st.text_area(
    "Enter patient symptoms:", 
    "Patient presents with fever, cough, and headache for 3 days..."
)

if st.button("Analyze"):
    processed_text = preprocess_text(text)
    if user_input.strip() == "":
        st.warning("Please enter symptoms to analyze.")
    else:
        with st.spinner("Analyzing symptoms..."):
            specialty, spec_conf = predict_specialty(user_input)
            disease, dis_conf = predict_disease(user_input)
            
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Specialty", specialty)
            
            with col2:
                st.metric("Predicted Disease", disease)
            
            st.markdown("---")
            
            st.warning(
                "⚠️ Note: This is an AI-assisted prediction. "
                "Always consult with a healthcare professional for accurate diagnosis."
            )
