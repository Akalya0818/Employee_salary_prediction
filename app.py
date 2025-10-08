import streamlit as st
import pandas as pd
import joblib

# Load model
import os
if not os.path.exists("best_model.pk1"):
    st.error("⚠️ Model file not found. Please upload 'best_model.pkl' to the app folder.")
    st.stop()

model = joblib.load("best_model.pk1")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on employee details.")

# Sidebar Inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
occupation = st.sidebar.selectbox("Job Role", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                               "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                               "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                               "Armed-forces"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Create DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Manual encoding (simple label encoding)
education_map = {
    "HS-grad": 0, "Some-college": 1, "Assoc": 2,
    "Bachelors": 3, "Masters": 4, "PhD": 5
}

occupation_map = {name: idx for idx, name in enumerate([
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-forces"
])}

input_df['education'] = input_df['education'].map(education_map)
input_df['occupation'] = input_df['occupation'].map(occupation_map)

st.subheader("Processed Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💡 Predicted Salary Class: {prediction[0]}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input preprocessing.")
        st.write("Error details:", str(e))

# Batch Prediction Section
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        # Apply same encoding
        batch_data['education'] = batch_data['education'].map(education_map)
        batch_data['occupation'] = batch_data['occupation'].map(occupation_map)
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write("Error details:", str(e))
