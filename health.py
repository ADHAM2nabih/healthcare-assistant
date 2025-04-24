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
    .result-container {
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
        color: #000000;
    }
    .diagnosis-card h3, .diagnosis-card p, .diagnosis-card div {
        color: #000000 !important;
    }
    .response-box {
        background: #f0f8ff;
        border-left: 4px solid #4a90e2;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }
    .symptoms-box {
        background: #e9f5ee;
        color: #000000 !important;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin-top: 15px;
    }
    .symptoms-box p {
        color: #000000 !important;
    }
    /* Override info alert */
    .stAlert {
        color: #000000 !important;
    }
    .stAlert div {
        color: #000000 !important;
    }
    .stInfo {
        background-color: #e9f5ee !important;
        color: #000000 !important;
    }
    .stInfo div {
        color: #000000 !important;
    }
    /* Override Streamlit's default styling for metrics */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

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
    أنا مساعد طبي ذكي. بناءً على الأعراض التالية: 
    
    تم تشخيص الحالة بأنها:
    - التخصص الطبي المحتمل: {diagnosis['specialty'][0]} 
    - المرض المحتمل: {diagnosis['disease'][0]} 
    
    اكتب ردًا مهنيًا باللغة العربية بطريقة ودية واحترافية:
    1. ابدأ بجملة ترحيب قصيرة
    2. اذكر التخصص الطبي المناسب للحالة
    3. اذكر المرض المحتمل بطريقة بسيطة
    4. قدم نصيحة أولية بسيطة
    5. اختتم بتذكير بضرورة مراجعة الطبيب
    
    الرد يجب أن:
    - يكون باللغة العربية الفصحى أو العامية البسيطة
    - لا يتجاوز 3-4 جمل قصيرة
    - لا يحتوي على أي معلومات غير طبية
    - لا يقدم وصفات علاجية
    - يكون دقيقًا ومهنيًا
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

# Input form
with st.form(key='symptom_form', clear_on_submit=False):
    user_input = st.text_area("💬 صف الأعراض التي تشعر بها...", 
                             key='input', 
                             value="",
                             placeholder="مثال: أعاني من حرارة وسعال منذ يومين")
    
    analyze_btn = st.form_submit_button("تحليل الأعراض 🩺")
    
    if analyze_btn and user_input:
        with st.spinner("🔍 جاري تحليل الأعراض..."):
            diagnosis = predict_medical_condition(user_input)
            
            # Get AI response
            bot_response = get_medical_response(user_input, diagnosis)
            
            # Clear previous results by using a container outside the form
            st.session_state.diagnosis = diagnosis
            st.session_state.response = bot_response
            st.session_state.symptoms = user_input

# Display results (outside the form so it can be cleared on new submission)
if 'diagnosis' in st.session_state and st.session_state.diagnosis:
    with st.container():
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Display user symptoms with BLACK text
        st.subheader("📝 الأعراض المذكورة:")
        st.markdown(f"""
        <div class="symptoms-box">
            <p style="color: #000000; font-size: 1.1rem;">{st.session_state.symptoms}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display AI response
        st.subheader("🩺 تقييم الحالة:")
        st.markdown(f'<div class="response-box">{st.session_state.response}</div>', unsafe_allow_html=True)
        
        # Display medical details with improved visibility - WITHOUT confidence percentages
        with st.expander("🔍 التفاصيل الطبية", expanded=False):
            st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
            
            # Custom heading with black text
            st.markdown('<h3 style="color: #000000; font-weight: bold;">نتائج التحليل</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="color: #000000; padding: 10px; border-radius: 5px; background: #f8f9fa;">
                    <p style="font-weight: bold; color: #000000; margin-bottom: 5px;">التخصص الطبي</p>
                    <p style="font-size: 1.2rem; color: #000000; margin-bottom: 2px;">{st.session_state.diagnosis['specialty'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="color: #000000; padding: 10px; border-radius: 5px; background: #f8f9fa;">
                    <p style="font-weight: bold; color: #000000; margin-bottom: 5px;">المرض المحتمل</p>
                    <p style="font-size: 1.2rem; color: #000000; margin-bottom: 2px;">{st.session_state.diagnosis['disease'][0]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Warning with black text
            st.markdown("""
            <div style="background-color: #fff3cd; color: #000000; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; margin-top: 15px;">
                <p style="color: #000000; margin: 0;">⚠️ <strong>ملاحظة:</strong> هذه النتائج استشارية فقط ولا تغني عن مراجعة الطبيب</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Button to clear results
if 'diagnosis' in st.session_state and st.session_state.diagnosis:
    if st.button("فحص جديد"):
        # Clear session state
        st.session_state.diagnosis = None
        st.session_state.response = None
        st.session_state.symptoms = None
        st.rerun()
