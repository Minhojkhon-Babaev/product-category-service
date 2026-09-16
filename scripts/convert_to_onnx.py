#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

TEXT_FIELDS = ("type_prefix", "vendor", "name", "model", "description")


def combine_text(frame: pd.DataFrame) -> pd.Series:
    columns = [frame[field].fillna("").astype(str) for field in TEXT_FIELDS]
    return columns[0].str.cat(columns[1:], sep=" ")


def reference_stats(frame: pd.DataFrame) -> dict:
    text_lengths = combine_text(frame).str.len().to_numpy()
    _, edges = np.histogram(text_lengths, bins=10)
    edges[0] = -0.5
    edges[-1] = 1e12
    counts, _ = np.histogram(text_lengths, bins=edges)

    category_counts = frame["category_ind"].value_counts(normalize=True).sort_index()
    return {
        "text_length": {
            "bin_edges": edges.tolist(),
            "probabilities": (counts / counts.sum()).tolist(),
        },
        "category_ind": {
            "labels": category_counts.index.astype(int).tolist(),
            "probabilities": category_counts.tolist(),
        },
    }


def convert(pipeline_path: Path, train_path: Path, output: Path) -> None:
    with pipeline_path.open("rb") as source:
        pipeline = pickle.load(source)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_count = len(vectorizer.vocabulary_)
    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=13,
    )
    onnx.checker.check_model(onnx_model)

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output / "vectorizer.joblib")
    (output / "model.onnx").write_bytes(onnx_model.SerializeToString())

    train = pd.read_parquet(train_path)
    (output / "reference_stats.json").write_text(
        json.dumps(reference_stats(train), ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = {
        "model_name": "tfidf-logistic-regression",
        "model_version": "1",
        "feature_count": feature_count,
        "class_count": len(classifier.classes_),
        "classes": classifier.classes_.astype(int).tolist(),
        "kaggle_public_score": 0.32413,
        "kaggle_private_score": 0.36728,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sklearn classifier to ONNX")
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert(arguments.pipeline, arguments.train_data, arguments.output)
