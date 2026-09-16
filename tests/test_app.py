import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import redis
from fastapi.testclient import TestClient

from product_service.app import create_app, load_metadata
from product_service.settings import Settings


class FakePipeline:
    def __init__(self):
        self.commands = []

    def lpush(self, *args):
        self.commands.append(("lpush", args))
        return self

    def ltrim(self, *args):
        self.commands.append(("ltrim", args))
        return self

    def execute(self):
        self.commands.append(("execute", ()))


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.pipeline_instance = FakePipeline()
        self.closed = False
        self.fail = False

    def ping(self):
        if self.fail:
            raise redis.RedisError("down")
        return True

    def get(self, key):
        if self.fail:
            raise redis.RedisError("down")
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = str(value)

    def pipeline(self):
        return self.pipeline_instance

    def close(self):
        self.closed = True


@pytest.fixture
def components(mocker, tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"model_version": "1"}')
    config = Settings(
        vectorizer_path=tmp_path / "vectorizer.joblib",
        metadata_path=metadata_path,
        redis_url="redis://test",
    )
    fake_redis = FakeRedis()
    classifier = Mock()
    classifier.is_ready.return_value = True
    classifier.predict.return_value = 42
    mocker.patch("product_service.app.joblib.load", return_value=Mock())
    mocker.patch("product_service.app.redis.Redis.from_url", return_value=fake_redis)
    classifier_class = mocker.patch(
        "product_service.app.TritonClassifier",
        return_value=classifier,
    )
    app = create_app(config)
    return app, config, fake_redis, classifier, classifier_class


@pytest.fixture
def client(components):
    app, *_ = components
    with TestClient(app) as test_client:
        yield test_client


def payload(name="Кровать"):
    return {
        "url": "https://example.test/product",
        "texts": {
            "name": name,
            "description": "Мягкая мебель",
            "model": "",
            "type_prefix": "Кровать",
            "vendor": "Орматек",
        },
        "image_url": "https://example.test/image.jpg",
    }


def test_load_metadata(tmp_path):
    path = Path(tmp_path) / "metadata.json"
    path.write_text('{"version": 3}')
    assert load_metadata(path) == {"version": 3}


def test_lifespan_initializes_and_closes_dependencies(client, components):
    _, config, fake_redis, classifier, classifier_class = components

    classifier_class.assert_called_once()
    assert classifier_class.call_args.kwargs["url"] == config.triton_url
    assert client.app.state.classifier is classifier
    assert client.app.state.metadata == {"model_version": "1"}
    assert fake_redis.closed is False


def test_health_is_ok_when_dependencies_are_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True, "redis_ready": True}


def test_health_is_degraded_on_dependency_errors(client, components):
    _, _, fake_redis, classifier, _ = components
    classifier.is_ready.side_effect = RuntimeError("down")
    fake_redis.fail = True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model_ready": False,
        "redis_ready": False,
    }


def test_predict_calls_model_and_records_event(client, components):
    _, config, fake_redis, classifier, _ = components

    response = client.post("/predict", json=payload())

    assert response.status_code == 200
    assert response.json() == {"category_ind": 42}
    classifier.predict.assert_called_once_with("Кровать Орматек Кровать Мягкая мебель")
    assert list(fake_redis.values.values()) == ["42"]
    commands = fake_redis.pipeline_instance.commands
    assert commands[0][0] == "lpush"
    event = json.loads(commands[0][1][1])
    assert event["category_ind"] == 42
    assert event["vendor"] == "Орматек"
    assert commands[1] == ("ltrim", (config.event_list_key, 0, config.event_list_max_size - 1))


def test_predict_uses_cache(client, components):
    _, _, fake_redis, classifier, _ = components

    first = client.post("/predict", json=payload())
    second = client.post("/predict", json=payload())

    assert first.json() == second.json() == {"category_ind": 42}
    classifier.predict.assert_called_once()
    assert len(fake_redis.pipeline_instance.commands) == 3


def test_predict_returns_503_when_redis_fails(client, components):
    _, _, fake_redis, _, _ = components
    fake_redis.fail = True

    response = client.post("/predict", json=payload())

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference service unavailable"}


def test_predict_validates_contract(client):
    response = client.post("/predict", json={"url": "", "image_url": ""})
    assert response.status_code == 422
