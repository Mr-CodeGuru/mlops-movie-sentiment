import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA = BASE / "data" / "imdb.csv"
MODEL = BASE / "model.pkl"
METRICS = BASE / "metrics.json"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run: python download_data.py")

    df = pd.read_csv(DATA).dropna(subset=["review", "sentiment"])
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    print(f"Rows: {len(df)} | Positive: {df['label'].mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # "br" removes the leftover <br /> HTML tags in the reviews
    stop_words = list(ENGLISH_STOP_WORDS | {"br"})
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=stop_words,
                    ngram_range=(1, 2),
                    max_features=100_000,
                    min_df=3,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=4.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))
    print(metrics)

    joblib.dump(pipeline, MODEL, compress=3)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL.name} ({MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()