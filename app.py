from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("heart_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    df = pd.DataFrame([features], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal"
    ])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        result = f"⚠️ High Risk of Heart Disease! (Probability: {probability:.2f})"
    else:
        result = f"✅ Low Risk of Heart Disease (Probability: {probability:.2f})"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)