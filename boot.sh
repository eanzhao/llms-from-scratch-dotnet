#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="${ROOT_DIR}/web"
PORT="${PORT:-4321}"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required but was not found in PATH." >&2
  exit 1
fi

if [ ! -d "${WEB_DIR}" ]; then
  echo "Web directory not found: ${WEB_DIR}" >&2
  exit 1
fi

if [ ! -f "${WEB_DIR}/package.json" ]; then
  echo "package.json not found in ${WEB_DIR}" >&2
  exit 1
fi

cd "${WEB_DIR}"

if [ ! -d node_modules ]; then
  echo "Installing web dependencies with pnpm..."
  pnpm install
fi

EXISTING_PIDS="$(lsof -ti tcp:"${PORT}" || true)"
if [ -n "${EXISTING_PIDS}" ]; then
  echo "Killing existing process(es) on port ${PORT}: ${EXISTING_PIDS}"
  kill ${EXISTING_PIDS} 2>/dev/null || true
  sleep 1

  REMAINING_PIDS="$(lsof -ti tcp:"${PORT}" || true)"
  if [ -n "${REMAINING_PIDS}" ]; then
    echo "Force killing remaining process(es) on port ${PORT}: ${REMAINING_PIDS}"
    kill -9 ${REMAINING_PIDS}
    sleep 1
  fi
fi

echo "Starting Astro dev server on http://127.0.0.1:${PORT}"
exec pnpm dev --host 0.0.0.0 --port "${PORT}"
