import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)

# Create Stacking Classifier
stacked_model = StackingClassifier(
    estimators=[('rf', rf), ('gbm', gbm)], 
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
)

# Train the stacked model
stacked_model.fit(X_train, y_train)

# FUNCTION TO GET USER INPUT
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)
    
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# GET USER DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Standardize user input
user_data_scaled = scaler.transform(user_data)

# PREDICTION & PROBABILITIES
user_result = stacked_model.predict(user_data_scaled)
user_proba = stacked_model.predict_proba(user_data_scaled)[0]  # Probabilities of class 0 and 1

# DEBUGGING OUTPUT
st.subheader("Debugging Information")
st.write(f"Raw User Data: {user_data}")
st.write(f"Scaled User Data: {user_data_scaled}")
st.write(f"Prediction Probability (0=Not Diabetic, 1=Diabetic): {user_proba}")

# THRESHOLD ADJUSTMENT: Predict `1` if probability of being diabetic is > 0.4 instead of 0.5
threshold = 0.4
final_prediction = 1 if user_proba[1] > threshold else 0

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
color = 'red' if final_prediction == 1 else 'blue'

# OUTPUT
st.subheader('Your Report:')
output = 'You are Diabetic' if final_prediction == 1 else 'You are not Diabetic'
st.title(output)

# MODEL ACCURACY CHECK
y_pred = stacked_model.predict(X_test)
st.subheader('Model Accuracy:')
st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

# CHECK PREDICTION DISTRIBUTION
unique, counts = np.unique(y_pred, return_counts=True)
st.subheader("Model Predictions in Test Data")
st.write(dict(zip(unique, counts)))
