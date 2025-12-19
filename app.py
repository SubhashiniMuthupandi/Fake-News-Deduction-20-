import streamlit as st
import pickle
import os

# --- SETTINGS ---
FOLDER_NAME = "Model8"
MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Path logic to find files inside Model8 folder
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, FOLDER_NAME, MODEL_FILE)
vect_path = os.path.join(base_path, FOLDER_NAME, VECTORIZER_FILE)

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Enter the news text below to verify its authenticity.")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open(model_path, 'rb') as f_model:
            model = pickle.load(f_model)
        with open(vect_path, 'rb') as f_vect:
            vectorizer = pickle.load(f_vect)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

if model is not None and vectorizer is not None:
    # --- UI SECTION ---
    news_input = st.text_area("Paste News Content Here:", height=200, placeholder="Type or paste the news article text...")

    if st.button("Analyze News"):
        if news_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # 1. Transform input using the loaded vectorizer
            transformed_input = vectorizer.transform([news_input])
            
            # 2. Predict using the loaded model
            prediction = model.predict(transformed_input)
            
            # 3. Display Results
            st.write("---")
            if prediction[0] == 1:
                st.success("### Result: REAL NEWS ✅")
                st.balloons()
            else:
                st.error("### Result: FAKE NEWS 🚨")
                st.write("Be careful! This article shows characteristics of misinformation.")

else:
    st.error("❌ Model or Vectorizer files not found!")
    st.info(f"Please ensure you have a folder named '{FOLDER_NAME}' containing '{MODEL_FILE}' and '{VECTORIZER_FILE}'.")
