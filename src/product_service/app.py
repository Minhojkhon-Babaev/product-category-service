import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import redis
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from .inference import TritonClassifier
from .metrics import CACHE_OPERATIONS, DEPENDENCY_READY, PREDICTIONS, REQUEST_LATENCY, REQUESTS
from .preprocessing import cache_key, combine_text
from .schemas import HealthResponse, PredictRequest, PredictResponse
from .settings import Settings, settings

logger = logging.getLogger(__name__)


def load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def create_app(config: Settings = settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        vectorizer = joblib.load(config.vectorizer_path)
        app.state.classifier = TritonClassifier(
            vectorizer=vectorizer,
            url=config.triton_url,
            model_name=config.triton_model_name,
            timeout=config.triton_timeout_seconds,
        )
        app.state.redis = redis.Redis.from_url(config.redis_url, decode_responses=True)
        app.state.metadata = load_metadata(config.metadata_path)
        yield
        app.state.redis.close()

    app = FastAPI(title=config.service_name, version="1.0.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model_ready = False
        redis_ready = False
        try:
            model_ready = request.app.state.classifier.is_ready()
        except Exception:
            logger.exception("Triton readiness check failed")
        try:
            redis_ready = bool(request.app.state.redis.ping())
        except redis.RedisError:
            logger.exception("Redis readiness check failed")
        DEPENDENCY_READY.labels("triton").set(model_ready)
        DEPENDENCY_READY.labels("redis").set(redis_ready)
        status = "ok" if model_ready and redis_ready else "degraded"
        return HealthResponse(status=status, model_ready=model_ready, redis_ready=redis_ready)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        started = time.perf_counter()
        status = "success"
        try:
            key = cache_key(payload)
            cached = request.app.state.redis.get(key)
            if cached is not None:
                CACHE_OPERATIONS.labels("hit").inc()
                category_ind = int(cached)
            else:
                CACHE_OPERATIONS.labels("miss").inc()
                text = combine_text(payload)
                category_ind = request.app.state.classifier.predict(text)
                request.app.state.redis.setex(key, config.cache_ttl_seconds, category_ind)
                event = json.dumps(
                    {
                        "timestamp": time.time(),
                        "category_ind": category_ind,
                        "text_length": len(text),
                        "vendor": payload.texts.vendor or "",
                    },
                    ensure_ascii=False,
                )
                pipeline = request.app.state.redis.pipeline()
                pipeline.lpush(config.event_list_key, event)
                pipeline.ltrim(config.event_list_key, 0, config.event_list_max_size - 1)
                pipeline.execute()
            PREDICTIONS.labels(str(category_ind)).inc()
            return PredictResponse(category_ind=category_ind)
        except (redis.RedisError, RuntimeError, ValueError) as error:
            status = "error"
            logger.exception("Prediction failed")
            raise HTTPException(status_code=503, detail="Inference service unavailable") from error
        finally:
            REQUESTS.labels("/predict", status).inc()
            REQUEST_LATENCY.labels("/predict").observe(time.perf_counter() - started)

    return app


app = create_app()
