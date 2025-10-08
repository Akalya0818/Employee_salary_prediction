import streamlit as st
import pandas as pd
import joblib
import os

# ----------------- MODEL LOADING -----------------
if not os.path.exists("best_model.pk1"):
    st.error("⚠️ Model file not found. Please upload 'best_model.pk1' to the app folder.")
    st.stop()

model = joblib.load("best_model.pk1")

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Employee Salary Classification", page_icon="💰", layout="centered")
st.title("💰 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or <=50K based on employee details.")

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
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

# Manual one-hot encoding
education_cols = ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Assoc', 'Some-college']
occupation_cols = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-forces"
]

# Add encoded columns
for col in education_cols:
    input_df[f'education_{col}'] = 1 if education == col else 0

for col in occupation_cols:
    input_df[f'occupation_{col}'] = 1 if occupation == col else 0

# Drop original categorical columns
input_df.drop(['education', 'occupation'], axis=1, inplace=True)

st.subheader("Processed Input Data (matches model training structure)")
st.write(input_df)

# ----------------- SINGLE PREDICTION -----------------
if st.button("Predict Salary Class"):
    try:
        # Predict using NumPy array to avoid feature name mismatch
        prediction = model.predict(input_df.values)
        # Convert numerical output to readable label
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

        # Apply same encoding logic
        for col in education_cols:
            batch_data[f'education_{col}'] = batch_data['education'].apply(lambda x: 1 if x == col else 0)

        for col in occupation_cols:
            batch_data[f'occupation_{col}'] = batch_data['occupation'].apply(lambda x: 1 if x == col else 0)

        batch_data.drop(['education', 'occupation'], axis=1, inplace=True)

        # Predict using NumPy array
        batch_preds = model.predict(batch_data.values)
        batch_data['PredictedClass'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

        st.success("✅ Batch prediction successful!")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error("⚠️ Batch prediction failed.")
        st.write("Error details:", str(e))


