import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import os

# Load and clean data
df = pd.read_csv(r"C:\Users\vinay\Downloads\Titanic-Dataset.csv")
df["Age"] = SimpleImputer(strategy="mean").fit_transform(df[["Age"]])
df[["Cabin", "Embarked"]] = SimpleImputer(strategy="most_frequent").fit_transform(df[["Cabin", "Embarked"]])
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df.drop(columns=["Name", "PassengerId"], inplace=True)

for col in ['Ticket', 'Cabin']:
    df[col] = le.fit_transform(df[col].astype(str))

df = pd.get_dummies(df, columns=["Embarked"], prefix="embarked")
scaler = MinMaxScaler()
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_scaled = scaler.fit_transform(X)
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create the 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the model, scaler, and feature order
feature_order = X.columns.tolist()
with open("model/titanic_model.pkl", "wb") as file:
    pickle.dump((model, scaler, feature_order), file)

print("Model trained and saved!")

from flask import Flask, request, render_template_string
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize app
app = Flask(__name__)

# Load model, scaler, and feature order
with open("model/titanic_model.pkl", "rb") as file:
    model, scaler, feature_order = pickle.load(file)

# Preprocess function
def preprocess_input(data_dict, scaler, feature_order):
    df = pd.DataFrame([data_dict])

    # Same hardcoded mapping as during training
    df["Sex"] = 0 if df["Sex"].iloc[0] == "female" else 1
    for col in ['Ticket', 'Cabin']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df = pd.get_dummies(df, columns=["Embarked"], prefix="embarked")

    # Ensure correct column order and handle missing columns
    df = df.reindex(columns=feature_order, fill_value=0)
    scaled_data = scaler.transform(df)
    return scaled_data

# HTML form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Titanic Survival Prediction</title>
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            background-color: #0077b6;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .container {
            margin: 40px auto;
            background: white;
            padding: 40px 30px;
            border-radius: 12px;
            width: 400px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        h2 {
            text-align: center;
            margin-bottom: 30px;
            color: #023e8a;
        }
        input, select {
            width: 100%;
            padding: 12px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 14px;
        }
        input[type=submit] {
            background-color: #0077b6;
            color: white;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        input[type=submit]:hover {
            background-color: #023e8a;
        }
        .result {
            text-align: center;
            margin-top: 25px;
            font-size: 20px;
            color: #0077b6;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        Titanic Survival Prediction
    </div>
    <div class="container">
        <h2>Enter Passenger Details</h2>
        <form action="/predict" method="post">
            <input type="number" name="Pclass" placeholder="Passenger Class (1,2,3)" required>
            <select name="Sex" required>
                <option value="">Select Gender</option>
                <option>male</option>
                <option>female</option>
            </select>
            <input type="number" name="Age" placeholder="Age" step="any" required>
            <input type="number" name="SibSp" placeholder="Siblings/Spouses Aboard" required>
            <input type="number" name="Parch" placeholder="Parents/Children Aboard" required>
            <input type="number" name="Fare" placeholder="Fare" step="any" required>
            <input type="text" name="Ticket" placeholder="Ticket Number" required>
            <input type="text" name="Cabin" placeholder="Cabin Number" required>
            <select name="Embarked" required>
                <option value="">Select Embarked Port</option>
                <option>C</option>
                <option>Q</option>
                <option>S</option>
            </select>
            <input type="submit" value="Predict">
        </form>

        {% if prediction %}
        <div class="result">
            Prediction: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "Pclass": int(request.form["Pclass"]),
        "Sex": request.form["Sex"],
        "Age": float(request.form["Age"]),
        "SibSp": int(request.form["SibSp"]),
        "Parch": int(request.form["Parch"]),
        "Fare": float(request.form["Fare"]),
        "Ticket": request.form["Ticket"],
        "Cabin": request.form["Cabin"],
        "Embarked": request.form["Embarked"]
    }
    processed_input = preprocess_input(input_data, scaler, feature_order)
    prediction = model.predict(processed_input)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    return render_template_string(HTML_TEMPLATE, prediction=result)

# Run server
if __name__ == "__main__":
    app.run(port=5000, debug=True)