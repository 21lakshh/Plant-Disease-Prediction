import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title=" Kisaan Saathi - Crop Disease Detection & AgriAssist Support",
    page_icon="🌽",
    layout="wide"
)   
# Loading the Gemeni API from the config file
working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, 'config.json')
config_data = json.load(open(config_path))

# loading api key 
api_key = config_data["GOOGLE_API_KEY"]
st.markdown("""
    <style>
    .main {
        background-color: #080505; /* Dark background for the app */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green buttons */
        color: white; /* Ensure button text is white */
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-family: Arial, sans-serif; /* Improve readability */
        font-size: 14px; /* Adjust text size */
    }
    .user-message {
        background: #e3f2fd; /* Light blue background */
        color: #000; /* Black text for user messages */
        margin-left: 20%;
        border: 1px solid #90caf9; /* Optional border for better visibility */
    }
    .bot-message {
        background: #f5f5f5; /* Light grey background */
        color: #000; /* Black text for bot messages */
        margin-right: 20%;
        border: 1px solid #ddd; /* Optional border for better visibility */
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(f"{working_dir}/trained_model/plant_disease_prediction.h5")
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
    
    genai.configure(api_key=api_key) # Your api key goes here
    return model, class_indices, genai.GenerativeModel('gemini-pro')

model, class_indices, gemini = load_resources()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Language mapping
LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "ਪੰਜਾਬੀ": "pa",
    "ಕನ್ನಡ": "kn"
}

def load_and_preprocess_image(image):
    img = Image.open(image).resize((224, 224))
    return np.expand_dims(np.array(img)/255.0, axis=0)

def predict_disease(image):
    processed_image = load_and_preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_indices[str(np.argmax(predictions))]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

st.title("🌽 Kisaan Saathi - Crop Disease Detection & AgriAssist Support")
st.markdown("Upload a plant leaf image for disease detection and chat with our AI expert!")

# Language selector
with st.sidebar:
    st.header("Settings")
    selected_lang = st.selectbox("Choose Language", options=list(LANGUAGES.keys()))
    st.markdown("---")
    st.info("🌐 Select your preferred language for assistance")

# Entering the image and predicting the disease
col1, col2 = st.columns(2)
uploaded_file = col1.file_uploader("Upload plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    if col2.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            disease, confidence = predict_disease(uploaded_file)
            st.session_state.disease = disease
            st.session_state.confidence = confidence
            st.session_state.messages = [] 
            
if 'disease' in st.session_state:
    st.success(f"**Detected Disease:** {st.session_state.disease} ({st.session_state.confidence:.2f}% confidence)")

    # Chat interface
    st.markdown("---")
    st.subheader("🤖 Farmer Support Chatbot")
    
    # Predefined questions
    cols = st.columns(3)
    questions = {
        "What causes this disease?": "Explain the causes of {disease} in {language}",
        "How to prevent it?": "Explain prevention methods for {disease} in {language}",
        "Treatment options?": "Suggest treatment options for {disease} in {language}"
    }
    
    for col, (q, prompt) in zip(cols, questions.items()):
        with col:
            if st.button(q):
                lang_code = LANGUAGES[selected_lang]
                full_prompt = prompt.format(disease=st.session_state.disease, language=selected_lang)
                response = gemini.generate_content(full_prompt)
                
                # Add to chat history
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.messages.append({"role": "assistant", "content": response.text})

    # Display chat history
    for message in st.session_state.messages:
        cls = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(f"""
            <div class="chat-message {cls}">
                <b>{'👤 Farmer' if message['role'] == 'user' else '🤖 AgriAssist'}:</b><br/>
                {message['content']}
            </div>
        """, unsafe_allow_html=True)

    # User input for custom questions
    user_input = st.chat_input("Ask more questions about the disease...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate context-aware prompt
        lang_code = LANGUAGES[selected_lang]
        context = f"The plant has been detected with {st.session_state.disease}. " + \
                  f"Respond in {selected_lang} language. Question: {user_input}"
        
        # Get Gemini response
        response = gemini.generate_content(context)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
        # Rerun to update chat display
        st.rerun()
