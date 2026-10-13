import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

MIN_CHARS = 10
MAX_CHARS = 10_000
LABELS = {0: "negative", 1: "positive"}


def validate_review(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text is required.")
    text = text.strip()
    if len(text) < MIN_CHARS:
        raise ValueError(f"Review must be at least {MIN_CHARS} characters.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Review must be at most {MAX_CHARS} characters.")
    return text


def create_app(model_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    path = Path(model_path) if model_path else Path(__file__).parent / "model.pkl"

    try:
        model = joblib.load(path)
    except Exception as exc:  # missing or corrupt model: app still boots, health reports it
        app.logger.error("Could not load model at %s: %s", path, exc)
        model = None

    def predict(text: str) -> dict:
        if model is None:
            raise RuntimeError("Model is not loaded.")
        proba = model.predict_proba([text])[0]
        idx = int(proba.argmax())
        return {
            "sentiment": LABELS[idx],
            "confidence": round(float(proba[idx]), 4),
            "probabilities": {
                "negative": round(float(proba[0]), 4),
                "positive": round(float(proba[1]), 4),
            },
        }

    @app.get("/")
    def index():
        return render_template("index.html", result=None, error=None, review="")

    @app.get("/health")
    def health():
        ok = model is not None
        return jsonify(status="ok" if ok else "unavailable", model_loaded=ok), (200 if ok else 503)

    @app.post("/predict")
    def predict_form():
        review = request.form.get("review", "")
        try:
            result = predict(validate_review(review))
            return render_template("index.html", result=result, error=None, review=review)
        except ValueError as exc:
            return render_template("index.html", result=None, error=str(exc), review=review), 400
        except RuntimeError as exc:
            return render_template("index.html", result=None, error=str(exc), review=review), 503

    @app.post("/api/predict")
    def predict_api():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Body must be JSON: {\"review\": \"...\"}"), 400
        try:
            return jsonify(predict(validate_review(payload.get("review"))))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)), debug=False)