import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm

# Load the trained model and preprocessor
lgbm_tuned_model = joblib.load('lgbm_tuned_model.pkl')
preprocessor = joblib.load('preprocessor.pkl') # Load the preprocessor

st.title('Diabetes Prediction App')

st.write("Please enter the patient's details to predict the likelihood of diabetes.")

# Create input fields for user to enter patient details
with st.form("prediction_form"):
    st.header("Patient Information")
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    glucose_level = st.number_input('Glucose Level', min_value=50, max_value=250, value=100)
    blood_pressure = st.number_input('Blood Pressure', min_value=50, max_value=200, value=90)
    insulin = st.number_input('Insulin', min_value=10.0, max_value=300.0, value=100.0, step=0.1)
    physical_activity = st.number_input('Physical Activity (minutes/week)', min_value=0, max_value=1000, value=150)
    family_history = st.selectbox('Family History of Diabetes', ['No', 'Yes'])
    
    submitted = st.form_submit_button("Predict Diabetes")

if submitted:
    # Create new features based on input values
    # Physical_State
    if physical_activity <= 150:
        physical_state = 'Insufficiently Active'
    else:
        physical_state = 'Highly Active'

    # Glucose_Status
    if glucose_level <= 99.9:
        glucose_status = 'Normal'
    elif 100 <= glucose_level <= 125.9:
        glucose_status = 'Prediabetic'
    else:
        glucose_status = 'Diabetic'

    # Insulin_category
    if insulin < 35:
        insulin_category = 'Optimal'
    elif 35 <= insulin < 70:
        insulin_category = 'Normal'
    elif 70 <= insulin < 105:
        insulin_category = 'Early-Resistance'
    else:
        insulin_category = 'High Resistance'

    # glucose_group (same logic as Glucose_Status, potentially redundant but included for consistency with notebook)
    if glucose_level <= 99.9:
        glucose_group = 'Normal'
    elif 100 <= glucose_level <= 125.9:
        glucose_group = 'Prediabetic'
    else:
        glucose_group = 'Diabetic'

    # Create a DataFrame from the user input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Glucose_Level': glucose_level,
        'Blood_Pressure': blood_pressure,
        'Insulin': insulin,
        'Physical_Activity': physical_activity,
        'Family_History': family_history,
        'Physical_State': physical_state,
        'Glucose_Status': glucose_status,
        'Insulin_category': insulin_category,
        'glucose_group': glucose_group
    }])

    # Preprocess the input data
    # The preprocessor expects columns in the same order as X during training
    # Need to make sure column names match exactly as they were during training.
    # Get original numerical and categorical feature names from the preprocessor
    numerical_features_from_preprocessor = ['Age', 'BMI', 'Glucose_Level', 'Blood_Pressure', 'Insulin', 'Physical_Activity']
    categorical_features_from_preprocessor = ['Gender', 'Family_History', 'Physical_State', 'Glucose_Status', 'Insulin_category', 'glucose_group']

    # Reorder columns to match the training data's original column order before preprocessing
    # Ensure 'Family_History' and 'Gender' are correctly handled as categorical features
    input_data_ordered = input_data[numerical_features_from_preprocessor + categorical_features_from_preprocessor]

    processed_input = preprocessor.transform(input_data_ordered)

    # Make prediction
    prediction = lgbm_tuned_model.predict(processed_input)
    prediction_proba = lgbm_tuned_model.predict_proba(processed_input)[:, 1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"The patient is predicted to have Diabetes with a probability of {prediction_proba[0]:.2f}")
    else:
        st.success(f"The patient is predicted NOT to have Diabetes with a probability of {1 - prediction_proba[0]:.2f}")
