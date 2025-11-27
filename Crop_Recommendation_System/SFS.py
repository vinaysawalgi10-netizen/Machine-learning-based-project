import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cohere

st.set_page_config(page_title="FarmAI – Crop Recommendation", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\vinay\OneDrive\Desktop\PROJECTS__\Crop_recommendationV2.csv")
    return df

df = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# Train Models
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )
}

train_accuracies = {}
test_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_accuracies[name] = accuracy_score(y_train, model.predict(X_train))
    test_accuracies[name] = accuracy_score(y_test, model.predict(X_test))

# -------------------------------
# Cohere AI Crop Advice
# -------------------------------
def get_crop_advice(crop, features_dict):
    prompt = f"""
A farmer has the following soil and climate conditions:
{features_dict}.

The recommended crop is: {crop}.

Explain in 10-15 points why this crop is suitable for these conditions,
and provide a small tip for best farming practices and advantages as well.
"""

    co = cohere.Client("wvLUF48gLNbd3tb7Je33mRyN13mj10c5gPT6yMK5")

    try:
        response = co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.7
        )
        # Access the response text directly
        if hasattr(response, "text"):
            return response.text.strip()
        else:
            return "⚠️ AI Suggestion failed: no response received"
    except Exception as e:
        return f"⚠️ AI Suggestion failed: {str(e)}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 FarmAI – Intelligent Crop Recommendation System")

# Model selection
model_choice = st.selectbox("Choose a Model", list(models.keys()))
chosen_model = models[model_choice]

# Show training and test accuracy
st.subheader("📊 Model Accuracy")
st.write(f"Training Accuracy: **{train_accuracies[model_choice]*100:.2f}%**")
st.write(f"Test Accuracy (Full Test Set): **{test_accuracies[model_choice]*100:.2f}%**")

# Random prediction
if st.button("🎲 Get Random Prediction"):
    idx = np.random.randint(0, X_test.shape[0])
    sample = X_test[idx].reshape(1, -1)
    true_label = le.inverse_transform([y_test[idx]])[0]
    pred_label = le.inverse_transform(chosen_model.predict(sample))[0]

    # Sample accuracy
    test_accuracy_sample = 1.0 if pred_label == true_label else 0.0

    st.subheader("🌾 Prediction Result (Random Sample)")
    st.write(f"**Predicted Crop:** {pred_label}")
    st.write(f"**Actual Crop (Ground Truth):** {true_label}")
    st.write(f"Test Accuracy for this sample: **{test_accuracy_sample*100:.2f}%**")

    # Show features
    features_dict = {col: float(X.iloc[idx][col]) for col in X.columns}
    st.subheader("🔬 Sample Features")
    st.dataframe(pd.DataFrame([features_dict]))

    # AI suggestion with spinner
    st.subheader("💡 AI-based Crop Suggestion")
    with st.spinner("Generating AI advice..."):
        advice = get_crop_advice(pred_label, features_dict)
    st.success(advice)
