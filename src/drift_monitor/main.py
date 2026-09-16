import json
import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import redis
from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)

DRIFT_SCORE = Gauge(
    "product_drift_score",
    "Latest drift score by feature",
    ["feature", "method"],
)
SAMPLE_SIZE = Gauge("product_drift_sample_size", "Events used in the latest drift calculation")
DRIFT_DETECTED = Gauge(
    "product_drift_detected",
    "Whether feature drift exceeds its configured threshold",
    ["feature"],
)
CALCULATIONS = Counter(
    "product_drift_calculations_total",
    "Number of periodic drift calculations",
    ["status"],
)


def population_stability_index(
    values: list[float],
    edges: list[float],
    expected: list[float],
) -> float:
    actual_counts, _ = np.histogram(values, bins=np.asarray(edges))
    actual = actual_counts / max(actual_counts.sum(), 1)
    expected_array = np.asarray(expected)
    epsilon = 1e-6
    return float(
        np.sum(
            (actual - expected_array)
            * np.log((actual + epsilon) / (expected_array + epsilon))
        )
    )


def jensen_shannon(
    observed: list[int],
    labels: list[int],
    expected: list[float],
) -> float:
    counts = {label: 0 for label in labels}
    for value in observed:
        if value in counts:
            counts[value] += 1
    total = max(sum(counts.values()), 1)
    actual = np.asarray([counts[label] / total for label in labels], dtype=float)
    reference = np.asarray(expected, dtype=float)
    midpoint = 0.5 * (actual + reference)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return math.sqrt(0.5 * (kl_divergence(actual, midpoint) + kl_divergence(reference, midpoint)))


class DriftMonitor:
    def __init__(self) -> None:
        self.redis = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True,
        )
        self.event_key = os.getenv("EVENT_LIST_KEY", "prediction_events")
        self.window_size = int(os.getenv("DRIFT_WINDOW_SIZE", "1000"))
        self.min_samples = int(os.getenv("DRIFT_MIN_SAMPLES", "20"))
        self.interval = int(os.getenv("DRIFT_INTERVAL_SECONDS", "60"))
        reference_path = Path(os.getenv("REFERENCE_STATS_PATH", "/artifacts/reference_stats.json"))
        self.reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def read_events(self) -> list[dict]:
        records = self.redis.lrange(self.event_key, 0, self.window_size - 1)
        return [json.loads(record) for record in records]

    def calculate(self) -> None:
        events = self.read_events()
        SAMPLE_SIZE.set(len(events))
        if len(events) < self.min_samples:
            CALCULATIONS.labels("insufficient_data").inc()
            return

        text_reference = self.reference["text_length"]
        text_score = population_stability_index(
            [event["text_length"] for event in events],
            text_reference["bin_edges"],
            text_reference["probabilities"],
        )
        category_reference = self.reference["category_ind"]
        category_score = jensen_shannon(
            [event["category_ind"] for event in events],
            category_reference["labels"],
            category_reference["probabilities"],
        )

        DRIFT_SCORE.labels("text_length", "psi").set(text_score)
        DRIFT_SCORE.labels("category_ind", "jensen_shannon").set(category_score)
        DRIFT_DETECTED.labels("text_length").set(text_score >= 0.2)
        DRIFT_DETECTED.labels("category_ind").set(category_score >= 0.2)
        CALCULATIONS.labels("success").inc()

    def run_forever(self) -> None:
        while True:
            try:
                self.calculate()
            except Exception:
                CALCULATIONS.labels("error").inc()
                logger.exception("Drift calculation failed")
            time.sleep(self.interval)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    monitor = DriftMonitor()
    start_http_server(int(os.getenv("MONITOR_PORT", "8001")))
    thread = threading.Thread(target=monitor.run_forever, daemon=False)
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
