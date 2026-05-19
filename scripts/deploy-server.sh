#!/usr/bin/env bash
# Обновление репозитория и перезапуск Pusplexity через Docker Compose на сервере.
#
# По умолчанию: сборка образов с использованием кэша Docker (инкрементально).
# Полная пересборка без кэша: флаг --full (или переменная FULL_BUILD=1).
#
# Использование:
#   ./scripts/deploy-server.sh
#   ./scripts/deploy-server.sh /path/to/Pusplexity
#   DEPLOY_DIR=/path/to/Pusplexity ./scripts/deploy-server.sh
#   ./scripts/deploy-server.sh --full
#   ./scripts/deploy-server.sh --full /path/to/Pusplexity
#   FULL_BUILD=1 ./scripts/deploy-server.sh
#
# Требования: git, docker compose v2, доступ к origin и к Docker.

set -euo pipefail

FULL_BUILD="${FULL_BUILD:-0}"
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    --full|--no-cache)
      FULL_BUILD=1
      ;;
    -h|--help)
      cat <<'EOF'
Обновление репозитория и перезапуск через docker compose.

  deploy-server.sh [путь]              — pull + up --build (кэш Docker по умолчанию)
  deploy-server.sh --full [путь]     — pull + build --no-cache + up
  FULL_BUILD=1 deploy-server.sh       — то же, что --full

Путь по умолчанию: /opt/Pusplexity. Иначе: первый аргумент или DEPLOY_DIR.
EOF
      exit 0
      ;;
    -*)
      echo "Неизвестный флаг: $arg (допустимо: --full, --no-cache, -h)" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done

APP_DIR="${POSITIONAL[0]:-${DEPLOY_DIR:-/opt/Pusplexity}}"

if [[ ! -d "$APP_DIR" ]]; then
  echo "Ошибка: каталог не найден: $APP_DIR" >&2
  echo "Укажите путь: $0 [/path/to/repo]  или  DEPLOY_DIR=/path/to/repo $0" >&2
  echo "Полная пересборка: $0 --full [/path/to/repo]" >&2
  exit 1
fi

cd "$APP_DIR"
echo "==> Каталог: $(pwd)"

echo "==> git pull origin main"
git pull origin main

if [[ "$FULL_BUILD" == "1" ]]; then
  echo "==> docker compose build --no-cache (полная пересборка образов)"
  docker compose build --no-cache
  echo "==> docker compose up -d"
  docker compose up -d
else
  echo "==> docker compose up -d --build (сборка из кэша Docker)"
  docker compose up -d --build
fi

echo "==> docker compose ps"
docker compose ps

echo "==> Готово."
