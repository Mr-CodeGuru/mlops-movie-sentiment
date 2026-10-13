import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    texts = ["great wonderful amazing film", "loved this brilliant movie",
             "terrible awful boring film", "hated this worst movie"]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    model.fit(texts, [1, 1, 0, 0])
    path = tmp_path / "model.pkl"
    joblib.dump(model, path)
    return create_app(path).test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["model_loaded"] is True


def test_index(client):
    assert client.get("/").status_code == 200


def test_api_predict_positive(client):
    r = client.post("/api/predict", json={"review": "a great and wonderful movie"})
    body = r.get_json()
    assert r.status_code == 200
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


def test_api_rejects_empty(client):
    assert client.post("/api/predict", json={"review": "  "}).status_code == 400


def test_api_rejects_bad_body(client):
    assert client.post("/api/predict", data="nope").status_code == 400


def test_form_predict(client):
    r = client.post("/predict", data={"review": "terrible boring awful movie"})
    assert r.status_code == 200 and b"Negative" in r.data


def test_missing_model_health(tmp_path):
    c = create_app(tmp_path / "missing.pkl").test_client()
    assert c.get("/health").status_code == 503