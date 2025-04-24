import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
import re
import nltk
import requests
import time
import joblib
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Set page config
st.set_page_config(
    page_title="المساعد الطبي الذكي",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #e4e8f0);
        color: #333;
    }
    .stTextInput>div>div>input {
        color: #333;
        background-color: rgba(255, 255, 255, 0.8);
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a7bc8;
    }
    .chat-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .diagnosis-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

# Load models
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

# Text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Prediction functions
def predict_medical_condition(text):
    processed_text = preprocess_text(text)
    
    # Specialty prediction
    sequence = specialty_tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    specialty_pred = specialty_model.predict(padded_sequence)
    specialty_class = specialty_le.inverse_transform([np.argmax(specialty_pred)])[0]
    specialty_conf = np.max(specialty_pred)
    
    # Disease prediction
    disease_pred = disease_model.predict([processed_text])
    disease_class = disease_le.inverse_transform(disease_pred)[0]
    disease_conf = np.max(disease_model.predict_proba([processed_text]))
    
    return {
        'specialty': (specialty_class, specialty_conf),
        'disease': (disease_class, disease_conf)
    }

# Bot response generator with medical restrictions
def get_medical_response(user_input, diagnosis):
medical_prompt = f"""
You are an AI medical assistant. Based on the following symptoms:
"{user_input}"

The preliminary diagnosis suggests:
- Probable medical specialty: {diagnosis['specialty'][0]} (confidence: {diagnosis['specialty'][1]*100:.1f}%)
- Likely condition: {diagnosis['disease'][0]} (confidence: {diagnosis['disease'][1]*100:.1f}%)

Please provide a professional response in English with:
1. A brief welcoming sentence
2. Mention the appropriate medical specialty
3. State the probable condition in simple terms
4. Offer one basic initial advice
5. Conclude by reminding to consult a doctor

The response must:
- Be in clear, simple English
- Not exceed 3-4 short sentences
- Contain only medical information
- Not provide specific treatment prescriptions
- Remain accurate and professional
- Include a disclaimer that this is not a substitute for professional medical advice

Example structure:
"Hello! Based on your symptoms, a {specialty} specialist would be most appropriate. Your symptoms may suggest {condition}. For now, I recommend {general advice}. Please consult a doctor for proper evaluation."

Important constraints:
- DO NOT diagnose beyond the model's predictions
- DO NOT suggest medications
- DO NOT make absolute claims about the condition
- ALWAYS emphasize seeing a real doctor
"""
    
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://medical-chatbot.com",
        "X-Title": "Medical Assistant"
    }
    payload = {
        "model": "qwen/qwen-2.5-7b-instruct:free",
        "messages": [{"role": "user", "content": medical_prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        bot_reply = data['choices'][0]['message']['content']
        return re.sub(r'[{}]', '', re.sub(r'\\boxed\s*', '', bot_reply)).strip()
    except Exception as e:
        return f"عذرًا، حدث خطأ في معالجة طلبك. يرجى المحاولة لاحقًا."

# Main app
st.title("🏥 المساعد الطبي الذكي")
st.write("أهلاً بك! يمكنني مساعدتك في فهم الأعراض الطبية وتوجيهك للتخصص المناسب.")

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.generated:
        for i in range(len(st.session_state.generated)-1, -1, -1):
            message(st.session_state.generated[i], key=str(i), is_user=False)
            message(st.session_state.past[i], is_user=True, key=str(i) + '_user')
    
    if st.session_state.diagnosis:
        with st.expander("🔍 التفاصيل الطبية", expanded=False):
            st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
            st.subheader("نتائج التحليل")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("التخصص الطبي", 
                         st.session_state.diagnosis['specialty'][0],
            
            with col2:
                st.metric("المرض المحتمل", 
                         st.session_state.diagnosis['disease'][0],
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area("💬 صف الأعراض التي تشعر بها...", 
                             key='input', 
                             value="",
                             placeholder="مثال: أعاني من حرارة وسعال منذ يومين")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.form_submit_button("تحليل الأعراض 🩺")
    with col2:
        chat_btn = st.form_submit_button("إرسال رسالة 💬")
    
    if analyze_btn and user_input:
        with st.spinner("🔍 جاري تحليل الأعراض..."):
            diagnosis = predict_medical_condition(user_input)
            st.session_state.diagnosis = diagnosis
            
            bot_response = get_medical_response(user_input, diagnosis)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(bot_response)
            st.rerun()
    
    if chat_btn and user_input:
        st.session_state.past.append(user_input)
        with st.spinner("🤖 جاري الرد..."):
            if st.session_state.diagnosis:
                bot_response = get_medical_response(user_input, st.session_state.diagnosis)
            else:
                bot_response = "من فضلك اضغط على 'تحليل الأعراض' أولاً للحصول على تشخيص أولي"
            
            st.session_state.generated.append(bot_response)
            st.rerun()
