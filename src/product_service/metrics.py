from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "product_api_requests_total",
    "Number of prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "product_api_request_duration_seconds",
    "Prediction request latency",
    ["endpoint"],
)
CACHE_OPERATIONS = Counter(
    "product_api_cache_operations_total",
    "Redis cache hits and misses",
    ["result"],
)
PREDICTIONS = Counter(
    "product_api_predictions_total",
    "Predicted category counts",
    ["category_ind"],
)
DEPENDENCY_READY = Gauge(
    "product_api_dependency_ready",
    "Dependency readiness status",
    ["dependency"],
)
