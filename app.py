import streamlit as st
import pandas as pd
import joblib
import os

# --------------- MODEL LOADING ---------------
if not os.path.exists("best_model.pk1"):
    st.error("⚠️ Model file not found. Please upload 'best_model.pk1' to the app folder.")
    st.stop()

model = joblib.load("best_model.pk1")

# --------------- PAGE CONFIG ---------------
st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on employee details.")

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-forces"
])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# ----------------- CREATE INPUT DATAFRAME -----------------
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Label encoding mapping - replace one-hot with this 
education_map = {
    "Bachelors": 0,
    "Masters": 1,
    "PhD": 2,
    "HS-grad": 3,
    "Assoc": 4,
    "Some-college": 5
}

occupation_map = {
    "Tech-support": 0,
    "Craft-repair": 1,
    "Other-service": 2,
    "Sales": 3,
    "Exec-managerial": 4,
    "Prof-specialty": 5,
    "Handlers-cleaners": 6,
    "Machine-op-inspct": 7,
    "Adm-clerical": 8,
    "Farming-fishing": 9,
    "Transport-moving": 10,
    "Priv-house-serv": 11,
    "Protective-serv": 12,
    "Armed-forces": 13
}

input_df['education'] = input_df['education'].map(education_map)
input_df['occupation'] = input_df['occupation'].map(occupation_map)

st.subheader("Processed Input Data (matches model training structure)")
st.write(input_df)

# ----------------- SINGLE PREDICTION -----------------
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df.values)
        result_label = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"💡 Predicted Salary Class: {result_label}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model compatibility.")
        st.write("Error details:", str(e))

# ----------------- BATCH PREDICTION -----------------
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        
        # Apply label encoding maps
        batch_data['education'] = batch_data['education'].map(education_map)
        batch_data['occupation'] = batch_data['occupation'].map(occupation_map)

        # Keep only the required columns, columns must match training features exactly 
        batch_data = batch_data[['age', 'education', 'occupation', 'hours-per-week', 'experience']]

        st.write("Processed batch input (first 5 rows):")
        st.write(batch_data.head())

        batch_preds = model.predict(batch_data.values)
        batch_data['PredictedClass'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write("Error details:", str(e))
