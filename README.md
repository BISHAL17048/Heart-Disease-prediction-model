# Heart Disease Prediction

Live demo: **[https://heart-diesease-prediction.onrender.com](https://heart-diesease-prediction.onrender.com)**

> A simple web application that predicts the likelihood of heart disease based on user input. The app provides an interactive UI for entering health parameters and returns a prediction along with a Pie Chart.

---

## Demo

Open the live application here: [https://heart-diesease-prediction.onrender.com](https://heart-diesease-prediction.onrender.com)


---

---

## Dataset
Dataset: [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)
---

## Features

* Interactive form to enter patient health metrics (age, sex, blood pressure, cholesterol, etc.).
* Client-side and server-side validation of inputs.
* ML model (trained offline) provides a probability or class prediction for heart disease risk.
* Clean, responsive UI suitable for desktop and mobile.
* Option to view prediction details and recommended next steps.

---

## Tech stack

* Frontend: HTML, CSS, JavaScript (or the framework used in your project)
* Backend: Python (Flask)
* Machine learning: scikit-learn, joblib
* Deployment: Render (live demo hosted at the URL above)

---

## Run locally

1. Clone the repo:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add/verify model and config files exist (for example `heart_model.pkl` and `schema.json`).

5. Run the application:

```bash
python app.py
```

6. Visit `http://127.0.0.1:5000` in your browser to use the app locally.

---

## Project structure

```
.
├── app.py              # Flask web app
├── heart_model.pkl     # Trained ML model
├── schema.json         # Feature schema
├── templates/
│   └── index.html      # UI template
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---

## Example: app.py

```python
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Load artifacts
MODEL_PATH = "heart_model.pkl"
SCHEMA_PATH = "schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

features = schema["features"]

# Flask app
app = Flask(__name__)

# Mapping for categorical features
sex_map = {"male": 1, "female": 0}

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []
        for feat in features:
            val = request.form.get(feat, "")

            if feat == "sex":
                val = sex_map.get(val.lower(), 0)
            elif feat in ["cp", "restecg", "slope", "ca", "thal", "fbs", "exang"]:
                val = int(val) if val != "" else np.nan
            else:
                val = float(val) if val != "" else np.nan

            input_data.append(val)

        # Convert to DataFrame
        X_input = pd.DataFrame([input_data], columns=features)

        # Predict
        prob = model.predict_proba(X_input)[0, 1]
        pred = model.predict(X_input)[0]

        result = "❤️ No Heart Disease" if pred == 0 else "💔 Heart Disease Detected"
        return render_template(
            "index.html",
            features=features,
            result=result,
            probability=f"{prob:.2f}"
        )
    except Exception as e:
        return render_template("index.html", features=features, error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
```

---

## requirements.txt

```
blinker==1.9.0
click==8.2.1
colorama==0.4.6
Flask==3.1.1
gunicorn==20.1.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.2
MarkupSafe==3.0.2
numpy==2.1.3
pandas==2.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.6.1
scipy==1.16.1
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
Werkzeug==3.1.3
```

---

## API (example)

If your app exposes a prediction API endpoint, document it like this (adjust to actual endpoints):

**POST** `/predict`

Request body (JSON):

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Response (JSON):

```json
{
  "prediction": 1,
  "probability": 0.83,
  "message": "High risk of heart disease"
}
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request with a clear description of the change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-change`)
3. Commit your changes
4. Push to your branch and open a PR

---

## License

Specify your license here (e.g., MIT). Example:

```
MIT License
```

---

## Contact

For questions or feedback, contact the project owner or maintainer.
