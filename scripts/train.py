#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TEXT_FIELDS = ("type_prefix", "vendor", "name", "model", "description")


def combine_text(frame: pd.DataFrame) -> pd.Series:
    columns = [frame[field].fillna("").astype(str) for field in TEXT_FIELDS]
    return columns[0].str.cat(columns[1:], sep=" ")


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=80_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.5,
                    max_iter=300,
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(train_path: Path, test_path: Path | None, output: Path) -> None:
    frame = pd.read_parquet(train_path)
    texts = combine_text(frame)
    target = frame["category_ind"].astype(int)
    train_x, validation_x, train_y, validation_y = train_test_split(
        texts,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    metrics = {
        "validation_accuracy": float(accuracy_score(validation_y, predictions)),
        "validation_f1_macro": float(f1_score(validation_y, predictions, average="macro")),
    }
    print(json.dumps(metrics, indent=2))

    model.fit(texts, target)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "pipeline.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if test_path:
        test = pd.read_parquet(test_path)
        test_predictions = model.predict(combine_text(test))
        submission = pd.DataFrame(
            {"ID": range(len(test_predictions)), "category_ind": test_predictions.astype(int)}
        )
        submission.to_csv(output / "submission.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train product category model")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training-output"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.train_data, arguments.test_data, arguments.output)
