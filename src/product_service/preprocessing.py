import hashlib
import json

from .schemas import PredictRequest

TEXT_FIELDS = ("type_prefix", "vendor", "name", "model", "description")


def combine_text(request: PredictRequest) -> str:
    parts = [getattr(request.texts, field) or "" for field in TEXT_FIELDS]
    return " ".join(part.strip() for part in parts if part.strip())


def cache_key(request: PredictRequest) -> str:
    normalized = request.model_dump(mode="json")
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()
    return f"prediction:{hashlib.sha256(payload).hexdigest()}"
