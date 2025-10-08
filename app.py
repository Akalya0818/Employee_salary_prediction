import streamlit as st
import pandas as pd
import joblib
import os

# --------------- MODEL LOADING ---------------
if not os.path.exists("best_model.pk1"):
    st.error("⚠️ Model file not found. Please upload 'best_model.pk1' to the app folder.")
    st.stop()

model = joblib.load("best_model.pk1")

education_cols = ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college']
occupation_cols = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-forces"
]

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", education_cols)
occupation = st.sidebar.selectbox("Job Role", occupation_cols)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Input dataframe with numeric features
input_df = pd.DataFrame({
    'age': [age],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# One-hot encode education and occupation for single input
for col in education_cols:
    input_df[f'education_{col}'] = [1 if education == col else 0]
for col in occupation_cols:
    input_df[f'occupation_{col}'] = [1 if occupation == col else 0]

st.subheader("Processed Input Data (matches model training structure)")
st.write(input_df)

if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df.values)
        result_label = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"💡 Predicted Salary Class: {result_label}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Please check model compatibility.")
        st.write("Error details:", str(e))

# Batch prediction
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        processed = pd.DataFrame()
        processed['age'] = batch_data['age']
        processed['hours-per-week'] = batch_data['hours-per-week']
        processed['experience'] = batch_data['experience']

        for col in education_cols:
            processed[f'education_{col}'] = (batch_data['education'] == col).astype(int)
        for col in occupation_cols:
            processed[f'occupation_{col}'] = (batch_data['occupation'] == col).astype(int)

        st.write("Processed batch input (first 5 rows):")
        st.write(processed.head())

        batch_preds = model.predict(processed.values)
        batch_data['PredictedClass'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write("Error details:", str(e))


