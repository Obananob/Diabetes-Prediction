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
def load_model(path: str = "Log_reg.pkl"):
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
        # assemble raw input (these are the app's "raw" columns)
        input_data = pd.DataFrame({
            'Age': [int(age)],
            'Gender': [int(gender)],  # numeric fallback if needed
            'BMI': [float(bmi)],
            'Glucose_Level': [float(glucose)],
            'Blood_Pressure': [float(bp)],
            'Insulin': [float(insulin)],
            'Physical_Activity': [int(activity)],
            'Family_History': [int(family_history)]
        })

        # Keep the display strings around so we can create proper one-hot columns if model expects them
        input_data_display = input_data.copy()
        input_data_display['Gender_display'] = gender_display
        input_data_display['Family_History_display'] = history_display
        input_data_display['Physical_Activity_display'] = activity_display

        # If the model exposes feature_names_in_, try to align/reindex the app input to match exactly.
        try:
            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
                input_features = list(input_data.columns)
                if model_features != input_features:
                    st.warning(
                        "Warning: The input features do not exactly match the model's expected features.\n\n"
                        f"Model expects: {model_features}\n\n"
                        f"App sends:   {input_features}\n\n"
                        "The app will attempt to align the inputs automatically. If this fails, use a saved preprocessing pipeline."
                    )

                # Start with a copy and then add any missing one-hot / dummy columns the model expects.
                X = input_data_display.copy()

                # If model expects Gender_M / Gender_F style columns, create them
                for feat in model_features:
                    if feat not in X.columns:
                        # handle Gender_M / Gender_F
                        if feat.startswith("Gender_"):
                            # expected token after underscore, e.g., 'M' or 'Male' or 'Female'
                            token = feat.split("_", 1)[1]
                            token_norm = token.lower()
                            if token_norm in ["m", "male", "man"]:
                                X[feat] = 1 if gender_display.lower().startswith("m") else 0
                            elif token_norm in ["f", "female", "woman"]:
                                X[feat] = 1 if gender_display.lower().startswith("f") else 0
                            else:
                                # generic compare
                                X[feat] = 1 if token_norm in gender_display.lower() else 0

                        # handle Family_History_Yes style
                        elif feat.lower() == "family_history_yes":
                            X[feat] = 1 if history_display == "Yes" else 0

                        # handle Physical_Activity_... dummies
                        elif feat.startswith("Physical_Activity_"):
                            token = feat.split("_", 2)[2] if feat.count("_") >= 2 else feat.split("_", 1)[1]
                            X[feat] = 1 if token.lower() in activity_display.lower() else 0

                        else:
                            # generic numeric column missing -> fill with 0
                            X[feat] = 0

                # Reindex to the exact order expected by the model (fill any remaining missing with 0)
                X = X.reindex(columns=model_features, fill_value=0)
            else:
                X = input_data
        except Exception:
            # if we can't auto-align, fall back to original and let sklearn raise a clear error
            X = input_data

        # Make prediction with error handling
        try:
            pred = model.predict(X)[0]
            prob = get_probability(model, X)  # may be None

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
