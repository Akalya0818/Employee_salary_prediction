import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="💰 Employee Salary Prediction", layout="centered")

# --- TITLE ---
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **<=50K** based on details.")

# --- THEME TOGGLE ---
theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .css-18e3th9 {background-color: #111 !important;}
        .css-1v3fvcr {color: #eee !important;}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- AUTOMATIC MODEL LOADING ---
MODEL_FILES = ["best_model (2).pk1", "best_model.pk1"]
model = None

for mf in MODEL_FILES:
    if os.path.exists(mf):
        model = joblib.load(mf)
        st.success(f"✅ Model '{mf}' loaded successfully!")
        break

if model is None:
    st.error("⚠️ No trained model found in the folder. Please place 'best_model (2).pk1' or 'best_model.pk1' in the same folder as this app.")
    st.stop()

# --- FEATURE LIST ---
FULL_EDUCATION_COLS = ['Assoc', 'Bachelors', 'HS-grad', 'Masters', 'PhD']
FULL_OCCUPATION_COLS = ["Craft-repair", "Exec-managerial", "Other-service", "Sales", "Tech-support"]

MODEL_FEATURE_ORDER = [
    'age', 'experience', 'hours-per-week',
    'education_Assoc', 'education_Bachelors', 'education_HS-grad',
    'education_Masters', 'education_PhD',
    'occupation_Craft-repair', 'occupation_Exec-managerial',
    'occupation_Other-service', 'occupation_Sales', 'occupation_Tech-support'
]

REQUIRED_CSV_COLS = ['age', 'hours-per-week', 'education', 'occupation', 'educational-num']

# --- SINGLE PREDICTION ---
st.sidebar.header("Enter Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
education = st.sidebar.selectbox("Education Level", FULL_EDUCATION_COLS)
occupation = st.sidebar.selectbox("Occupation", FULL_OCCUPATION_COLS)

input_data = pd.DataFrame({
    'age': [age],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

for col in FULL_EDUCATION_COLS:
    input_data[f'education_{col}'] = [1 if education == col else 0]

for col in FULL_OCCUPATION_COLS:
    input_data[f'occupation_{col}'] = [1 if occupation == col else 0]

final_input = input_data[MODEL_FEATURE_ORDER]

st.subheader("Processed Input Data (13 features)")
st.dataframe(final_input)

if st.button("🔍 Predict Salary Class"):
    try:
        pred = model.predict(final_input.values)
        result = ">50K" if pred[0] == 1 else "<=50K"
        emoji = "💎" if pred[0] == 1 else "📘"
        st.success(f"{emoji} Predicted Salary Class: **{result}**")
    except Exception as e:
        st.error("⚠ Prediction failed.")
        st.write(str(e))

# --- BATCH PREDICTION ---
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV for batch predictions", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
        if missing_cols:
            st.error(f"⚠ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        processed = pd.DataFrame()
        processed['age'] = df['age']
        processed['hours-per-week'] = df['hours-per-week']
        processed['experience'] = df['educational-num']

        for col in FULL_EDUCATION_COLS:
            processed[f'education_{col}'] = (df['education'].astype(str).str.strip() == col).astype(int)
        for col in FULL_OCCUPATION_COLS:
            processed[f'occupation_{col}'] = (df['occupation'].astype(str).str.strip() == col).astype(int)

        final_batch_input = processed[MODEL_FEATURE_ORDER]

        preds = model.predict(final_batch_input.values)
        df['PredictedClass'] = np.where(preds == 1, ">50K", "<=50K")

        st.success("✅ Batch prediction complete!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Download Predictions CSV",
            csv,
            file_name="salary_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("⚠ Error during batch prediction.")
        st.write(str(e))
