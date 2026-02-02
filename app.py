import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from math import exp
from typing import Optional

# Page Configuration (must be before other UI calls)
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Disease Prediction")
st.write("Enter patient details below to generate a risk assessment.")

# --- Load the Model ---
@st.cache_resource
def load_model(path: str = "diabetes_model.pkl"):
    """
    Try joblib first (common for sklearn), then pickle as fallback.
    Returns the model or None if loading fails.
    """
    try:
        try:
            model = joblib.load(path)
        except Exception:
            # fallback to pickle
            with open(path, "rb") as f:
                model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure it is in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_probability(model, X: pd.DataFrame) -> Optional[float]:
    """
    Return probability for positive class if available.
    If not available, try to use decision_function and sigmoid it.
    Returns None if no probability-like score is available.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # assume binary classification, positive class at index 1
            return float(proba[0][1])
        if hasattr(model, "decision_function"):
            score = model.decision_function(X)
            # handle array-like
            if isinstance(score, (np.ndarray, list, tuple)):
                score = float(np.ravel(score)[0])
            return float(sigmoid(score))
    except Exception:
        # fallthrough to None
        return None
    return None

# --- Input Form using st.form so inputs are submitted together ---
st.header("Patient Information")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender_display = st.selectbox("Gender", ["Male", "Female"])
        # Confirm these mappings match your training encoding
        gender_map = {"Male": 1, "Female": 0}
        gender = int(gender_map[gender_display])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=500, value=100)
    with col2:
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=30)
        activity_display = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        activity_map = {"Low": 0, "Moderate": 1, "High": 2}
        activity = int(activity_map[activity_display])
        history_display = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        history_map = {"No": 0, "Yes": 1}
        family_history = int(history_map[history_display])

    submitted = st.form_submit_button("Predict Risk")

# --- Prediction Logic ---
if submitted:
    if model is None:
        st.error("No model loaded. Fix the model file and reload the app.")
    else:
        # assemble input; ensure column names and order match training
        input_data = pd.DataFrame({
            'Age': [int(age)],
            'Gender': [int(gender)],
            'BMI': [float(bmi)],
            'Glucose_Level': [float(glucose)],
            'Blood_Pressure': [float(bp)],
            'Insulin': [float(insulin)],
            'Physical_Activity': [int(activity)],
            'Family_History': [int(family_history)]
        })

        # If the model exposes feature_names_in_, check for a mismatch and warn
        try:
            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
                input_features = list(input_data.columns)
                if model_features != input_features:
                    st.warning(
                        "Warning: The input features do not exactly match the model's expected features.\n\n"
                        f"Model expects: {model_features}\n\n"
                        f"App sends:   {input_features}\n\n"
                        "If these differ, predictions may be incorrect. Adjust column names/order to match."
                    )
        except Exception:
            # ignore if introspection not supported
            pass

        # Make prediction with error handling
        try:
            pred = model.predict(input_data)[0]
            prob = get_probability(model, input_data)  # may be None

            st.markdown("---")

            if pred == 1:
                st.error("Result: High Risk of Diabetes Detected")
                if prob is not None:
                    st.write(f"Confidence (probability of diabetes): {prob:.1%}")
                else:
                    st.info("Model does not provide a probability score for this prediction.")
                st.warning("⚠️ This patient should consult a specialist for further testing.")
            else:
                st.success("Result: Low Risk of Diabetes Detected")
                if prob is not None:
                    st.write(f"Confidence (probability of NO diabetes): {(1 - prob):.1%}")
                else:
                    st.info("Model does not provide a probability score for this prediction.")

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.info("Check if your feature names, column order, and data types match what the model expects.")
            st.exception(e)  # shows stack trace in the app for debugging (remove in production)
