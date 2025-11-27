import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib
import os

st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚲", layout="centered")
# === Load and preprocess the dataset ===
@st.cache_resource
def train_and_save_model():
    df = pd.read_csv(r"C:\Users\vinay\OneDrive\Desktop\ML_WebApp_Project\Bike_Rental.csv")

    # Convert datetime and extract new time-based features
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday

    # Drop unwanted columns
    df.drop(columns=['datetime', 'atemp', 'casual', 'registered'], inplace=True)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Define feature columns and target
    feature_cols = ['season', 'holiday', 'workingday', 'weather',
                    'temp', 'humidity', 'windspeed', 'hour', 'day', 'month', 'weekday']
    X_raw = df_imputed[feature_cols]
    y = df_imputed['count']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, 'bike_rental_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Accuracy
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return feature_cols, model, scaler, r2

# Load or train model
feature_cols, model, scaler, r2 = train_and_save_model()


st.title("🚲 Bike Rental Prediction")
st.write("Enter details below to predict the number of bike rentals.")

# Input fields
user_input = []
for field in feature_cols:
    val = st.number_input(f"{field.replace('_', ' ').capitalize()}", step=1.0)
    user_input.append(val)

if st.button("Predict"):
    try:
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Bike Rentals: {int(prediction)}")
    except Exception as e:
        st.error(f"Error: {e}")

# Show model accuracy
st.sidebar.header("Model Info")
st.sidebar.write(f"R² Score: **{r2:.3f}**")
