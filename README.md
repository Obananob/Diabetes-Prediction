# Diabetes Prediction App

A machine learning-powered web application that predicts the likelihood of diabetes based on patient health metrics. Built with Streamlit and LightGBM.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)

## Overview

This application uses a trained LightGBM model to predict diabetes risk based on various patient health indicators including age, BMI, glucose levels, blood pressure, insulin levels, physical activity, and family history.

## Features

- **Interactive Web Interface**: Easy-to-use form for entering patient information
- **Real-time Predictions**: Instant diabetes risk assessment
- **Probability Scores**: Detailed likelihood percentages for predictions
- **Feature Engineering**: Automatically generates derived features for enhanced accuracy
- **Professional UI**: Clean and intuitive design with color-coded results

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Obananob/Diabetes-Prediction
   cd diabetes-prediction-app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   
   Make sure these files are in your project directory:
   - `lgbm_tuned_model.pkl` - Trained LightGBM model
   - `preprocessor.pkl` - Data preprocessor

## Dependencies

```
streamlit
pandas
numpy
scikit-learn==1.6.1
joblib
lightgbm
```

## Project Structure

```
diabetes-prediction-app/
│
├── streamlitapp.py           # Main Streamlit application
├── requirements.txt          # Python dependencies
├── lgbm_tuned_model.pkl     # Trained LightGBM model
├── preprocessor.pkl          # Data preprocessor
├── notebook.ipynb            # Model training & analysis notebook
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
```

## Usage

1. **Run the application**
   ```bash
   streamlit run streamlit app.py
   ```

2. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

3. **Enter patient information**
   - Age (1-120 years)
   - Gender (Male/Female)
   - BMI (10.0-60.0)
   - Glucose Level (50-250 mg/dL)
   - Blood Pressure (50-200 mmHg)
   - Insulin (10.0-300.0 μU/mL)
   - Physical Activity (0-1000 minutes/week)
   - Family History of Diabetes (Yes/No)

4. **Get prediction**
   
   Click "Predict Diabetes" to receive a risk assessment

## How It Works

### Feature Engineering

The app automatically creates derived features from user inputs:

1. **Physical_State**: Categorizes activity level
   - Insufficiently Active: ≤150 min/week
   - Highly Active: >150 min/week

2. **Glucose_Status**: Blood glucose classification
   - Normal: ≤99.9 mg/dL
   - Prediabetic: 100-125.9 mg/dL
   - Diabetic: ≥126 mg/dL

3. **Insulin_category**: Insulin resistance levels
   - Optimal: <35 μU/mL
   - Normal: 35-69 μU/mL
   - Early-Resistance: 70-104 μU/mL
   - High Resistance: ≥105 μU/mL

### Prediction Process

1. User inputs are collected via the web form
2. Derived features are computed
3. Data is preprocessed using the saved preprocessor
4. LightGBM model generates prediction and probability
5. Results are displayed with risk assessment

## Model Information

- **Algorithm**: LightGBM (Light Gradient Boosting Machine)
- **Model Type**: Binary Classification
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Output**: Binary prediction (0 = No Diabetes, 1 = Diabetes) with probability score

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Local Deployment

The app runs locally by default. For production deployment, consider:
- Docker containerization
- Cloud platforms (AWS, GCP, Azure)
- Heroku or similar PaaS providers

## Disclaimer

**This application is for educational and informational purposes only.**

- Not a substitute for professional medical advice
- Predictions should be verified by qualified healthcare professionals
- Always consult with a doctor for proper diagnosis and treatment
- Do not make medical decisions based solely on this tool

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Project Team

**Awibi Diabetes Disease Prediction Group**

### Dataset Acquisition & Selection
- Mr. Alli
- Miss Elizabeth

### Data Processing & Feature Engineering
- **Miss Opeyemi Omotola**
- **Mr. Oyedele** *(Phase Lead)*
- **Mr. Duke Okechukwu**

*Data cleaning, reshaping, wrangling, exploratory data analysis, and feature selection*

### Model Development & Optimization
- **Miss Oni Bashir**
- **Mr. Abdulkareem** *(Phase Lead)*
- **Mr. Duke Okechukwu**

*Model selection, train-test split, training, evaluation, comparison, and optimization*

### Deployment & Documentation
- **Miss Oni Bashir** *(Phase Lead)*
- **Mr. Abdulkareem**
- **Miss Elizabeth**

*Result interpretation, model deployment (Streamlit app), and project documentation*

---

*A collaborative machine learning project for diabetes risk prediction*

## Acknowledgments

- Thanks to the open-source community
- Streamlit for the amazing framework
- LightGBM developers

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [team email or contact person](obananob91@gmail.com)

