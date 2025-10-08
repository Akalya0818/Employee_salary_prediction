import streamlit as st
import pandas as pd
import joblib

# Load trained model or pipeline
# ⚠️ Make sure your file is named correctly: best_model.pkl, not .pk1
model = joblib.load("best_model.pkl")

# Streamlit app setup
st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on employee details.")

# Sidebar input fields
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", 
                                 ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
occupation = st.sidebar.selectbox("Job Role", 
                                  ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                   "Armed-forces"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Convert user input to DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.subheader("Input Data Preview")
st.write(input_df)

# --- Single Prediction ---
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💡 Predicted Salary Class: {prediction[0]}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model and input preprocessing.")
        st.write("Error details:", str(e))
        if hasattr(model, "feature_names_in_"):
            st.write("Expected Features:", list(model.feature_names_in_))

# --- Batch Prediction ---
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", batch_data.head())
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error("⚠️ Batch prediction failed. Please ensure your CSV columns match training data.")
        st.write("Error details:", str(e))

