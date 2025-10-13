import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="💰 Employee Salary Prediction", layout="centered")

# --- TITLE ---
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **<=50K** based on input details.")

# --- MODEL LOADING ---
MODEL_PATH = "best_model_pk2.pkl"  # Adjust if needed (check your GitHub download file name)
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found. Please upload or place it in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success(f"✅ Model '{MODEL_PATH}' loaded successfully.")

# --- SINGLE PREDICTION FORM ---
st.sidebar.header("📋 Enter Employee Details")

def user_input():
    age = st.sidebar.slider("Age", 18, 65, 30)
    workclass = st.sidebar.selectbox("Workclass", 
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
         'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = st.sidebar.selectbox("Education", 
        ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college'])
    occupation = st.sidebar.selectbox("Occupation", 
        ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
         "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
         "Farming-fishing", "Transport-moving", "Protective-serv", "Armed-forces"])
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
    sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
    relationship = st.sidebar.selectbox("Relationship", 
        ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Unmarried', 'Other-relative'])
    native_country = st.sidebar.selectbox("Country", ['United-States', 'India', 'Philippines', 'Germany', 'Other'])

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'occupation': occupation,
        'hours-per-week': hours_per_week,
        'sex': sex,
        'relationship': relationship,
        'native-country': native_country
    }
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("🧾 Input Data")
st.write(input_df)

# --- SINGLE PREDICTION ---
if st.button("Predict Salary Class"):
    try:
        pred = model.predict(input_df)
        result = ">50K" if pred[0] == 1 else "<=50K"
        if result == ">50K":
            st.success(f"💎 Predicted Salary Class: **{result}**")
        else:
            st.info(f"📘 Predicted Salary Class: **{result}**")
    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.write(str(e))

# --- BATCH PREDICTION ---
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        preds = model.predict(data)
        data["PredictedClass"] = np.where(preds == 1, ">50K", "<=50K")

        st.success("✅ Batch prediction complete!")
        st.dataframe(data.head())

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions CSV", csv, "predicted_output.csv", "text/csv")

    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write(str(e))
