# Diabetes Prediction App

A machine learning-powered web application that predicts the likelihood of diabetes based on patient health metrics. Built with Streamlit and LightGBM.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)

## 🎯 Overview

This application uses a trained LightGBM model to predict diabetes risk based on various patient health indicators including age, BMI, glucose levels, blood pressure, insulin levels, physical activity, and family history.

## ✨ Features

- **Interactive Web Interface**: Easy-to-use form for entering patient information
- **Real-time Predictions**: Instant diabetes risk assessment
- **Probability Scores**: Detailed likelihood percentages for predictions
- **Feature Engineering**: Automatically generates derived features for enhanced accuracy
- **Professional UI**: Clean and intuitive design with color-coded results

## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Obananob/Diabetes-Prediction>
   cd diabetes-prediction-app
Create a virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Ensure model files are present
Make sure these files are in your project directory:
lgbm_tuned_model.pkl - Trained LightGBM model
preprocessor.pkl - Data preprocessor
📦 Dependencies
streamlit
pandas
numpy
scikit-learn==1.6.1
joblib
lightgbm
🎮 Usage
Run the application
streamlit run streamlit app.py
Open your browser
The app will automatically open at http://localhost:8501
Enter patient information
Age (1-120 years)
Gender (Male/Female)
BMI (10.0-60.0)
Glucose Level (50-250 mg/dL)
Blood Pressure (50-200 mmHg)
Insulin (10.0-300.0 μU/mL)
Physical Activity (0-1000 minutes/week)
Family History of Diabetes (Yes/No)
Get prediction
Click "Predict Diabetes" to receive a risk assessment
🔧 How It Works
Feature Engineering
The app automatically creates derived features from user inputs:
Physical_State: Categorizes activity level
Insufficiently Active: ≤150 min/week
Highly Active: >150 min/week
Glucose_Status: Blood glucose classification
Normal: ≤99.9 mg/dL
Prediabetic: 100-125.9 mg/dL
Diabetic: ≥126 mg/dL
Insulin_category: Insulin resistance levels
Optimal: <35 μU/mL
Normal: 35-69 μU/mL
Early-Resistance: 70-104 μU/mL
High Resistance: ≥105 μU/mL
Prediction Process
User inputs are collected via the web form
Derived features are computed
Data is preprocessed using the saved preprocessor
LightGBM model generates prediction and probability
Results are displayed with risk assessment
📊 Model Information
Algorithm: LightGBM (Light Gradient Boosting Machine)
Model Type: Binary Classification
Preprocessing: StandardScaler for numerical features, OneHotEncoder for categorical features
Output: Binary prediction (0 = No Diabetes, 1 = Diabetes) with probability score
🌐 Deployment
Streamlit Cloud
Push your code to GitHub
Connect your repository to Streamlit Cloud
Deploy with one click
Local Deployment
The app runs locally by default. For production deployment, consider:
Docker containerization
Cloud platforms (AWS, GCP, Azure)
Heroku or similar PaaS providers
⚠️ Disclaimer
This application is for educational and informational purposes only.
Not a substitute for professional medical advice
Predictions should be verified by qualified healthcare professionals
Always consult with a doctor for proper diagnosis and treatment
Do not make medical decisions based solely on this tool
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Project Team

**Awibi Diabetes Disease Prediction Group**

| Phase | Team Members | Lead |
|-------|-------------|------|
| **Dataset Acquisition** | Mr. Alli, Miss Elizabeth | - |
| **Data Processing & EDA** | Miss Opeyemi Omotola, Mr. Duke Okechukwu | Mr. Oyedele |
| **Model Development** | Miss Oni Bashir, Mr. Duke Okechukwu | Mr. Abdulkareem |
| **Deployment & Documentation** | Mr. Abdulkareem, Miss Elizabeth | Miss Oni Bashir |

### Key Responsibilities

- **Data Acquisition**: Dataset research and selection
- **Data Processing**: Cleaning, EDA, feature engineering
- **Model Development**: Training, evaluation, and optimization
- **Deployment**: Streamlit application and documentation
🙏 Acknowledgments
Thanks to the open-source community
Streamlit for the amazing framework
LightGBM developers
📞 Support
For issues, questions, or suggestions:
Open an issue on GitHub
Contact: your.email@example.com
