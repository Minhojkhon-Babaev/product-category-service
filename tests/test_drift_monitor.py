import json

import pytest

from drift_monitor.main import DriftMonitor, jensen_shannon, population_stability_index


def test_population_stability_index_is_zero_for_equal_distribution():
    score = population_stability_index(
        values=[0.2, 0.8, 1.2, 1.8],
        edges=[0.0, 1.0, 2.0],
        expected=[0.5, 0.5],
    )
    assert score == pytest.approx(0.0)


def test_population_stability_index_detects_shift():
    score = population_stability_index(
        values=[0.2, 0.3, 0.4, 0.5],
        edges=[0.0, 1.0, 2.0],
        expected=[0.5, 0.5],
    )
    assert score > 1.0


def test_jensen_shannon_is_zero_for_equal_distribution():
    score = jensen_shannon(
        observed=[1, 1, 2, 2],
        labels=[1, 2],
        expected=[0.5, 0.5],
    )
    assert score == pytest.approx(0.0)


def test_jensen_shannon_detects_category_shift():
    score = jensen_shannon(
        observed=[1, 1, 1, 1],
        labels=[1, 2],
        expected=[0.5, 0.5],
    )
    assert score > 0.4


@pytest.fixture
def monitor(mocker, tmp_path, monkeypatch):
    reference = {
        "text_length": {
            "bin_edges": [0.0, 10.0, 100.0],
            "probabilities": [0.5, 0.5],
        },
        "category_ind": {"labels": [1, 2], "probabilities": [0.5, 0.5]},
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setenv("REFERENCE_STATS_PATH", str(reference_path))
    monkeypatch.setenv("DRIFT_MIN_SAMPLES", "2")
    redis_client = mocker.patch("drift_monitor.main.redis.Redis.from_url").return_value
    instance = DriftMonitor()
    return instance, redis_client


def test_read_events_decodes_redis_records(monitor):
    instance, redis_client = monitor
    redis_client.lrange.return_value = [
        '{"category_ind": 1, "text_length": 5}',
        '{"category_ind": 2, "text_length": 20}',
    ]

    assert instance.read_events() == [
        {"category_ind": 1, "text_length": 5},
        {"category_ind": 2, "text_length": 20},
    ]


def test_calculate_handles_small_window(monitor, mocker):
    instance, _ = monitor
    mocker.patch.object(instance, "read_events", return_value=[{"category_ind": 1}])

    instance.calculate()


def test_calculate_updates_scores(monitor, mocker):
    instance, _ = monitor
    mocker.patch.object(
        instance,
        "read_events",
        return_value=[
            {"category_ind": 1, "text_length": 2},
            {"category_ind": 1, "text_length": 3},
        ],
    )

    instance.calculate()
