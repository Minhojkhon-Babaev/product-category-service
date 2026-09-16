#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

DOCKER_DESKTOP_BIN="/Applications/Docker.app/Contents/Resources/bin"
if [[ -d "$DOCKER_DESKTOP_BIN" ]]; then
  export PATH="$DOCKER_DESKTOP_BIN:$PATH"
fi

export SERVICE_PORT="${SERVICE_PORT:-8000}"
export TRITON_PORT="${TRITON_PORT:-8001}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
export ALERTMANAGER_PORT="${ALERTMANAGER_PORT:-9093}"
export GRAFANA_PORT="${GRAFANA_PORT:-3000}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  for env_file in .env /tmp/hw4-minio.env; do
    if [[ -f "$env_file" ]]; then
      set -a
      # Local development credentials are not committed to git.
      source "$env_file"
      set +a
      break
    fi
  done
fi

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Docker Compose is required" >&2
  exit 1
fi

"${COMPOSE[@]}" up -d --build --remove-orphans

echo "Waiting for API on port ${SERVICE_PORT}..."
for _ in {1..90}; do
  if curl --fail --silent "http://localhost:${SERVICE_PORT}/health" | grep -q '"status":"ok"'; then
    echo "Service is ready: http://localhost:${SERVICE_PORT}"
    echo "API docs: http://localhost:${SERVICE_PORT}/docs"
    echo "Grafana: http://localhost:${GRAFANA_PORT} (admin/admin by default)"
    echo "Prometheus: http://localhost:${PROMETHEUS_PORT}"
    exit 0
  fi
  sleep 2
done

echo "Service did not become ready" >&2
"${COMPOSE[@]}" ps
"${COMPOSE[@]}" logs --tail=100 api triton artifact-init
exit 1
