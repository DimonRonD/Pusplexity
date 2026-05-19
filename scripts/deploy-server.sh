#!/usr/bin/env bash
# Обновление репозитория и перезапуск Pusplexity через Docker Compose на сервере.
#
# Использование:
#   ./scripts/deploy-server.sh
#   ./scripts/deploy-server.sh /path/to/Pusplexity
#   DEPLOY_DIR=/path/to/Pusplexity ./scripts/deploy-server.sh
#
# Требования: git, docker compose v2, доступ к origin и к Docker.

set -euo pipefail

APP_DIR="${1:-${DEPLOY_DIR:-/opt/Pusplexity}}"

if [[ ! -d "$APP_DIR" ]]; then
  echo "Ошибка: каталог не найден: $APP_DIR" >&2
  echo "Укажите путь: $0 /path/to/repo  или  DEPLOY_DIR=/path/to/repo $0" >&2
  exit 1
fi

cd "$APP_DIR"
echo "==> Каталог: $(pwd)"

echo "==> git pull origin main"
git pull origin main

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> docker compose ps"
docker compose ps

echo "==> Готово."
