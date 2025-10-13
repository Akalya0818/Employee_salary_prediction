import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Constants ---
MODEL_FILE = "best_model (2).pk1"

FULL_EDUCATION_COLS = ['Assoc', 'Bachelors', 'HS-grad', 'Masters', 'PhD']
FULL_OCCUPATION_COLS = [
    "Craft-repair", "Exec-managerial", "Other-service", "Sales", "Tech-support"
]

# ✅ Corrected Alphabetical Feature Order (13 features total)
MODEL_FEATURE_ORDER = [
    'age', 'experience', 'hours-per-week',
    'education_Assoc', 'education_Bachelors', 'education_HS-grad',
    'education_Masters', 'education_PhD',
    'occupation_Craft-repair', 'occupation_Exec-managerial',
    'occupation_Other-service', 'occupation_Sales', 'occupation_Tech-support'
]

# Columns required in uploaded CSV
REQUIRED_CSV_COLS = ['age', 'hours-per-week', 'education', 'occupation', 'educational-num']


# --- Load Model ---
try:
    model_name = MODEL_FILE
    if os.path.exists("best_model (2).pk1"):
        model_name = "best_model (2).pk1"

    model = joblib.load(model_name)
    st.success(f"✅ Model '{model_name}' loaded successfully.")

except Exception as e:
    st.error("⚠ Model file not found or corrupted. Please upload a valid .pk1 file.")
    st.exception(e)
    st.stop()


# --- Page Setup ---
st.set_page_config(page_title="💰 Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns *>50K* or *<=50K* based on personal and professional details.")


# --- Sidebar Input Section ---
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
education = st.sidebar.selectbox("Education Level", FULL_EDUCATION_COLS)
occupation = st.sidebar.selectbox("Occupation", FULL_OCCUPATION_COLS)


# --- Prepare Input Data ---
input_data = pd.DataFrame({
    'age': [age],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# One-hot encode categorical features
for col in FULL_EDUCATION_COLS:
    input_data[f'education_{col}'] = [1 if education == col else 0]

for col in FULL_OCCUPATION_COLS:
    input_data[f'occupation_{col}'] = [1 if occupation == col else 0]

# Filter to model’s expected order
final_input = input_data[MODEL_FEATURE_ORDER]

st.subheader("Processed Input (13 features):")
st.dataframe(final_input)


# --- Predict Single Employee Salary ---
if st.button("🔍 Predict Salary Class"):
    try:
        prediction = model.predict(final_input.values)
        result_label = ">50K" if prediction[0] == 1 else "<=50K"
        emoji = "💸" if prediction[0] == 1 else "🪙"
        st.success(f"{emoji} Predicted Salary Class: *{result_label}*")
    except Exception as e:
        st.error("⚠ Prediction failed — likely due to feature mismatch.")
        st.write(str(e))


# --- Batch Prediction ---
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
        if missing_cols:
            st.error(f"⚠ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Process CSV
        processed = pd.DataFrame()
        processed['age'] = df['age']
        processed['hours-per-week'] = df['hours-per-week']
        processed['experience'] = df['educational-num']  # mapped from NPTEL dataset column

        for col in FULL_EDUCATION_COLS:
            processed[f'education_{col}'] = (df['education'].astype(str).str.strip() == col).astype(int)
        for col in FULL_OCCUPATION_COLS:
            processed[f'occupation_{col}'] = (df['occupation'].astype(str).str.strip() == col).astype(int)

        final_batch_input = processed[MODEL_FEATURE_ORDER]

        # Predict in batch
        preds = model.predict(final_batch_input.values)
        df['PredictedClass'] = [">50K" if p == 1 else "<=50K" for p in preds]

        st.success("✅ Batch prediction successful!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Download Results",
            csv,
            file_name="salary_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("⚠ Error during batch prediction.")
        st.write(str(e))
