import streamlit as st
import pandas as pd
import joblib  # For loading your saved model
import prediction_model.config as config
from prediction_model.processing.data_handling import load_pipeline

# Page configuration
st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💊",
    layout="centered",
)

# Custom CSS for a medical-themed interface
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            color: #333333;
        }
        .reportview-container {
            background: #f0f8ff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Medical Insurance Premium Prediction 💡")
st.write("Provide the following information to predict the insurance premium.")

# Load the model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return load_pipeline(config.MODEL_NAME)

model = load_model()

# Collecting user input
def user_input_features():
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", 10.0, 50.0, 22.5)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

    return pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

# Get user input
input_data = user_input_features()

# Button to generate prediction
if st.button("Predict Premium"):
    try:
        # Prediction
        prediction = model.predict(input_data[config.FEATURES])[0]
        st.success(f"Predicted Insurance Premium: ${prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
