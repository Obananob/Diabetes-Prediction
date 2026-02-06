import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm

lgbm_tuned_model = joblib.load('lgbm_tuned_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title('Diabetes Prediction App')

st.write("Please enter the patient's details to predict the likelihood of diabetes.")

with st.form("prediction_form"):
    st.header("Patient Information")
    
    age = st.number_input('Age', min_value=1, max_value=120, value=None, placeholder="Enter age")
    gender = st.selectbox('Gender', ['Select...', 'Female', 'Male'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=None, step=0.1, placeholder="Enter BMI ")
    glucose_level = st.number_input('Glucose Level (mg/dL)', min_value=50, max_value=250, value=None, placeholder="Enter glucose level")
    blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=50, max_value=200, value=None, placeholder="Enter blood pressure")
    insulin = st.number_input('Insulin (μU/mL)', min_value=10.0, max_value=300.0, value=None, step=0.1, placeholder="Enter insulin level")
    physical_activity = st.number_input('Physical Activity (minutes/week)', min_value=0, max_value=1000, value=None, placeholder="Enter minutes per week")
    family_history = st.selectbox('Family History of Diabetes', ['Select...', 'No', 'Yes'])
    
    submitted = st.form_submit_button("Predict Diabetes")

if submitted:
    
    if (age is None or bmi is None or glucose_level is None or 
        blood_pressure is None or insulin is None or physical_activity is None or
        gender == 'Select...' or family_history == 'Select...'):
        st.error("⚠️ Please fill in all fields before submitting.")
    else:
        if physical_activity <= 150:
            physical_state = 'Insufficiently Active'
        else:
            physical_state = 'Highly Active'

        if glucose_level <= 99.9:
            glucose_status = 'Normal'
        elif 100 <= glucose_level <= 125.9:
            glucose_status = 'Prediabetic'
        else:
            glucose_status = 'Diabetic'

        if insulin < 35:
            insulin_category = 'Optimal'
        elif 35 <= insulin < 70:
            insulin_category = 'Normal'
        elif 70 <= insulin < 105:
            insulin_category = 'Early-Resistance'
        else:
            insulin_category = 'High Resistance'

        if glucose_level <= 99.9:
            glucose_group = 'Normal'
        elif 100 <= glucose_level <= 125.9:
            glucose_group = 'Prediabetic'
        else:
            glucose_group = 'Diabetic'

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

        numerical_features_from_preprocessor = ['Age', 'BMI', 'Glucose_Level', 'Blood_Pressure', 'Insulin', 'Physical_Activity']
        categorical_features_from_preprocessor = ['Gender', 'Family_History', 'Physical_State', 'Glucose_Status', 'Insulin_category', 'glucose_group']

        input_data_ordered = input_data[numerical_features_from_preprocessor + categorical_features_from_preprocessor]

        processed_input = preprocessor.transform(input_data_ordered)

        prediction = lgbm_tuned_model.predict(processed_input)
        prediction_proba = lgbm_tuned_model.predict_proba(processed_input)[:, 1]

        st.subheader("Prediction Result:")
        
        
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: This patient shows a {prediction_proba[0]*100:.1f}% likelihood of having diabetes. Please consult a healthcare professional for proper diagnosis.") 
    else:
      st.success(f"✓ Low Risk: This patient shows a {(1 - prediction_proba[0])*100:.1f}% likelihood of NOT having diabetes. Regular monitoring is still recommended.")
