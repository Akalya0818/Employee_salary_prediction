import streamlit as st
import pandas as pd
import joblib

# Load model safely
import os
if not os.path.exists("best_model.pk1"):
    st.error("⚠️ Model file not found. Please upload 'best_model.pk1' to the app folder.")
    st.stop()

model = joblib.load("best_model.pk1")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on employee details.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
occupation = st.sidebar.selectbox("Job Role", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                               "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                               "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                               "Armed-forces"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Base input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# ---- MANUAL ONE-HOT ENCODING ----
education_cols = ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college']
occupation_cols = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-forces"]

# Create one-hot encoded columns manually
for col in education_cols:
    input_df[f'education_{col}'] = 1 if education == col else 0

for col in occupation_cols:
    input_df[f'occupation_{col}'] = 1 if occupation == col else 0

# Drop original categorical columns
input_df.drop(['education', 'occupation'], axis=1, inplace=True)

st.subheader("Processed Input Data (matches model training structure)")
st.write(input_df)

# --- Prediction ---
if st.button("Predict Salary Class"):
    try:
        # Ensure input columns match model's expected features
        if hasattr(model, "feature_names_in_"):
    # Add missing columns with 0s
           for col in model.feature_names_in_:
               if col not in input_df.columns:
                  input_df[col] = 0

      # Drop extra columns
        input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)
        st.success(f"💡 Predicted Salary Class: {prediction[0]}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model compatibility.")
        st.write("Error details:", str(e))

# --- Batch Prediction Section ---
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Apply same encoding logic
        for col in education_cols:
            batch_data[f'education_{col}'] = batch_data['education'].apply(lambda x: 1 if x == col else 0)

        for col in occupation_cols:
            batch_data[f'occupation_{col}'] = batch_data['occupation'].apply(lambda x: 1 if x == col else 0)

        batch_data.drop(['education', 'occupation'], axis=1, inplace=True)

        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write("Error details:", str(e))

