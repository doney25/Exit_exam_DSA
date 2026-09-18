import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"

with MODEL_PATH.open("rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
feature_order = artifact["feature_order"]
label_map = artifact["label_map"]


@app.route("/")
def home():
    return "Shopping Preference Prediction API is running. Use /predict with JSON input."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Ensure all expected fields are present; fill missing with None
        record = {key: payload.get(key, None) for key in feature_order}
        df = pd.DataFrame([record], columns=feature_order)

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(object).where(df[col].notna(), "Missing")

        pred_code = int(model.predict(df)[0])
        prediction = label_map.get(pred_code, str(pred_code))

        return jsonify({
            "prediction": prediction,
            "class_code": pred_code,
            "model": "shopping_preference_model"
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
