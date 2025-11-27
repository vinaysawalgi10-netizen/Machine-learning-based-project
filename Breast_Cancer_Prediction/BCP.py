import streamlit as st
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load & preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\vinay\Downloads\data.csv")
    df.drop(columns=["id", "Unnamed: 32"], inplace=True)
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])  # M=1, B=0
    return df

df = load_data()
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Define models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Breast Cancer Classification App")
st.write("Classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using ML models.")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Model comparison
st.subheader("🔍 Cross-Validation (F1-score, cv=5)")
results = {}
for name, clf in models.items():
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="f1")
    results[name] = scores.mean()
st.write(pd.DataFrame(results.items(), columns=["Model", "Mean F1-score"]))

# Choose final model
chosen_model = st.selectbox("Choose Final Model", list(models.keys()))
final_model = models[chosen_model]
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = final_model.predict(X_test_scaled)
st.subheader("📊 Test Set Performance")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# Random sample prediction (NO true label)
st.subheader("🎲 Predict on Random Test Sample")
if st.button("Pick Random Sample"):
    rand_index = random.randint(0, X_test.shape[0]-1)
    sample = X_test.iloc[rand_index]
    scaled_sample = scaler.transform([sample])
    pred = final_model.predict(scaled_sample)[0]

    st.write("**Predicted Result:**", "🟥 Malignant (M)" if pred==1 else "🟩 Benign (B)")
    st.write("🔎 Sample Features:", sample.to_dict())
