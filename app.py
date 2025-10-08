import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Constants ---
MODEL_FILE = "best_model.pk1"
# The full list of categories defined in the original app
FULL_EDUCATION_COLS = ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college']
FULL_OCCUPATION_COLS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-forces"
]

# --- CRITICAL FIX: DEFINING THE 13 FEATURES ---
# Based on the error "expecting 13 features," the model MUST have been trained on:
# 3 numeric features + 10 one-hot encoded features.
NUMERIC_COLS = ['age', 'hours-per-week', 'experience'] # Assuming 'experience' for sidebar
# Since we don't know the exact 10 OHE features, we must select 10 from the 20 available.
# This is a guess; if it fails, the feature list below is wrong.
OHE_FEATURE_NAMES = [
    'education_Bachelors', 'education_Masters', 'education_PhD', 
    'education_HS-grad', 'education_Assoc', # 5 Education features
    'occupation_Tech-support', 'occupation_Craft-repair', 'occupation_Other-service', 
    'occupation_Sales', 'occupation_Exec-managerial' # 5 Occupation features
] 

# The complete list of 13 features in the exact order the model expects
MODEL_FEATURE_ORDER = NUMERIC_COLS + OHE_FEATURE_NAMES

# Required columns in the CSV for batch prediction
# NOTE: adult 3.csv has 'educational-num' not 'experience'
REQUIRED_CSV_COLS = ['age', 'hours-per-week', 'education', 'occupation', 'educational-num']


# --------------- MODEL LOADING ---------------
try:
    # Use the latest uploaded model name
    if os.path.exists("best_model (1).pk1"):
        model_name = "best_model (1).pk1"
    elif os.path.exists("best_model.pk1"):
        model_name = "best_model.pk1"
    else:
        st.error("⚠️ Model file not found. Please upload 'best_model.pk1' or 'best_model (1).pk1'.")
        st.stop()
        
    model = joblib.load(model_name)
    st.success(f"Model '{model_name}' loaded successfully.")
    
except Exception as e:
    st.error("⚠️ Failed to load the model. Check scikit-learn version in requirements.txt.")
    st.exception(e)
    st.stop()


# --------------- PAGE CONFIG & TITLE ---------------
st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **<=50K** based on employee details.")

# --------------- SIDEBAR INPUTS (Single Prediction) ---------------
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", FULL_EDUCATION_COLS)
occupation = st.sidebar.selectbox("Job Role", FULL_OCCUPATION_COLS)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# --------------- CREATE INPUT DATAFRAME (Single Prediction) ---------------
# Create a DataFrame with ALL possible OHE columns first, then filter to the 13 required.
# This ensures the OHE features are correctly generated regardless of which ones the model uses.
single_input = pd.DataFrame({
    'age': [age],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# One-hot encode ALL categorical inputs
for col in FULL_EDUCATION_COLS:
    single_input[f'education_{col}'] = [1 if education == col else 0]

for col in FULL_OCCUPATION_COLS:
    single_input[f'occupation_{col}'] = [1 if occupation == col else 0]

# Filter the input to only include the 13 features the model expects
final_single_input = single_input[MODEL_FEATURE_ORDER]

st.subheader("Processed Input Data (13 Features)")
st.write(final_single_input)

# --------------- SINGLE PREDICTION ---------------
if st.button("Predict Salary Class"):
    try:
        # Pass the 13-feature input
        prediction = model.predict(final_single_input.values) 
        result_label = "**>50K**" if prediction[0] == 1 else "**<=50K**"
        st.success(f"💡 Predicted Salary Class: {result_label}")
    except Exception as e:
        st.error("⚠️ Prediction failed. The feature set or order is likely incorrect.")
        st.write("Error details:", str(e))

# --------------- BATCH PREDICTION (CSV Upload) ---------------
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # --- Robustness Check for required columns in the CSV ---
        missing_cols = [col for col in REQUIRED_CSV_COLS if col not in batch_data.columns]
        if missing_cols:
            st.error(f"⚠️ The uploaded CSV is missing the following required columns: **{', '.join(missing_cols)}**.")
            st.stop()
        
        # --- Data Preprocessing for Batch Prediction ---
        # 1. Create a placeholder DataFrame with all possible features
        processed = pd.DataFrame()
        processed['age'] = batch_data['age']
        processed['hours-per-week'] = batch_data['hours-per-week']
        # CRITICAL FIX: Map 'educational-num' from CSV to the 'experience' feature slot
        processed['experience'] = batch_data['educational-num'] 

        # Clean and One-hot encode ALL possible education columns
        batch_data['education'] = batch_data['education'].astype(str).str.strip()
        for col in FULL_EDUCATION_COLS:
            processed[f'education_{col}'] = (batch_data['education'] == col).astype(int)

        # Clean and One-hot encode ALL possible occupation columns
        batch_data['occupation'] = batch_data['occupation'].astype(str).str.strip()
        for col in FULL_OCCUPATION_COLS:
            processed[f'occupation_{col}'] = (batch_data['occupation'] == col).astype(int)
        
        # 2. Filter the processed data to ONLY include the 13 features in the correct order
        final_batch_input = processed[MODEL_FEATURE_ORDER]

        st.write("Processed batch input (first 5 rows with 13 features):")
        st.write(final_batch_input.head())

        # --- Run Batch Prediction ---
        batch_preds = model.predict(final_batch_input.values)
        batch_data['PredictedClass'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

        st.success("✅ Batch prediction successful! Preview of results:")
        st.dataframe(batch_data.head())

        # --- Download Button ---
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Predictions CSV", 
            csv, 
            file_name='predicted_classes.csv', 
            mime='text/csv'
        )

    except Exception as e:
        st.error("⚠️ Batch prediction failed due to an unexpected error. The model feature set or order is likely still incorrect.")
        st.write("Error details:", str(e))
