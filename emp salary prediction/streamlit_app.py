# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai # For Gemini API integration

# --- Configuration ---
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')

# Configure Gemini API (replace with your actual API key or Streamlit secret)
# It's recommended to use Streamlit secrets for API keys in deployment
# st.secrets["GEMINI_API_KEY"]
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-pro')
    GEMINI_ENABLED = True
except Exception as e:
    st.warning(f"Gemini API not configured or failed to load: {e}. Gemini features will be disabled.")
    GEMINI_ENABLED = False

# --- Set Page Config (Move this to the top) ---
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# --- Load Model ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run model_training.py first.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model_pipeline = load_model()

# Get feature names from the preprocessor (for displaying input fields)
# This assumes the preprocessor is the first step in the pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
# Extract original feature names used during training
original_numerical_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'educational-num']
original_categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]

# Define possible values for categorical features (from dataset exploration or domain knowledge)
# These should ideally be extracted dynamically from the training data's unique values
# For simplicity, hardcoding a subset here. In a real app, you'd load these from a saved list.
categorical_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Local-gov', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
    'education': ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'],
    'relationship': ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'],
    'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    'gender': ['Male', 'Female'],
    'native-country': ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Taiwan', 'Haiti', 'Portugal', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'France', 'Ireland', 'Thailand', 'Hong', 'Trinadad&Tobago', 'Greece', 'Honduras', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Scotland', 'Laos']
}

# --- Streamlit App Layout ---
st.title("💰 Employee Salary Prediction")
st.markdown("""
    Predict whether an employee's income is **<=50K** or **>50K** based on their characteristics.
    This model uses a Logistic Regression algorithm trained on the Adult Income dataset.
""")

st.sidebar.header("Input Employee Data")

# --- User Input Collection ---
input_data = {}

# Numerical Inputs
st.sidebar.subheader("Numerical Features")
input_data['age'] = st.sidebar.slider("Age", 17, 90, 30)
input_data['capital-gain'] = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
input_data['capital-loss'] = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
input_data['hours-per-week'] = st.sidebar.slider("Hours per Week", 1, 99, 40)
input_data['educational-num'] = st.sidebar.number_input("Educational Number", 1, 16, 10)  # Add this line

# Categorical Inputs
st.sidebar.subheader("Categorical Features")
for feature in original_categorical_features:
    if feature in categorical_options:
        input_data[feature] = st.sidebar.selectbox(f"{feature.replace('-', ' ').title()}", categorical_options[feature])
    else:
        st.sidebar.text(f"Warning: No options defined for {feature}")
        input_data[feature] = st.sidebar.text_input(f"{feature.replace('-', ' ').title()}", "Unknown")

# --- Prediction Button ---
if st.sidebar.button("Predict Income"):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns matches the training data
    # This is crucial for the preprocessor to work correctly
    # We need to reconstruct the DataFrame with all expected columns, even if they are not directly input
    # For simplicity, we'll assume the order of features in original_numerical_features + original_categorical_features
    # matches the order expected by the preprocessor.
    # A more robust solution would involve saving the column order from training.
    
    # Create a dummy DataFrame with all columns in the correct order and types
    # This is a workaround if you don't save the original column order and types
    # from the training script. In a real scenario, you'd load a template DataFrame.
    
    # For now, we'll just ensure the input_df has the correct columns for the preprocessor
    # The preprocessor handles the order internally based on its fit.
    
    try:
        prediction = model_pipeline.predict(input_df)
        prediction_proba = model_pipeline.predict_proba(input_df)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"Predicted Income: **>50K**")
        else:
            st.info(f"Predicted Income: **<=50K**")

        st.write(f"Probability of <=50K: {prediction_proba[0][0]:.2f}")
        st.write(f"Probability of >50K: {prediction_proba[0][1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled and the model is trained.")

# --- Gemini API Integration (Optional) ---
if GEMINI_ENABLED:
    st.markdown("---")
    st.subheader("🤖 Ask Gemini about the Data or Prediction")
    gemini_query = st.text_area("Enter your question for Gemini (e.g., 'What does 'capital-gain' mean in this context?', 'Explain the importance of education for salary.')", height=100)

    if st.button("Ask Gemini"):
        if gemini_query:
            with st.spinner("Gemini is thinking..."):
                try:
                    # Craft a prompt that gives context to Gemini
                    prompt = f"""
                    You are an AI assistant helping to understand an employee salary prediction model.
                    The model uses features like age, workclass, education, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, educational-num, and native-country to predict if income is <=50K or >50K.

                    Here's the user's question:
                    "{gemini_query}"

                    Please provide a concise and helpful answer based on the context of such a dataset and prediction task.
                    """
                    response = gemini_model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")
                    st.info("Please check your Gemini API key and internet connection.")
        else:
            st.warning("Please enter a question for Gemini.")

st.markdown("---")
st.caption("Developed by ROHIT BADIGER for Employee Salary Prediction")
