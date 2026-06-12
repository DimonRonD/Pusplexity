#!/usr/bin/env bash
# Обновление репозитория и применение изменений Pusplexity на сервере.
#
# По умолчанию: git pull + перезапуск контейнеров (код монтируется с хоста).
# Сборка образа — только если изменились Dockerfile или requirements.txt.
#
# Использование:
#   ./scripts/deploy-server.sh              — быстрый деплой (pull + restart при изменении кода)
#   ./scripts/deploy-server.sh --build      — принудительная сборка образа (кэш Docker)
#   ./scripts/deploy-server.sh --full       — полная пересборка без кэша
#   ./scripts/deploy-server.sh /path/to/repo
#
# Требования: git, docker compose v2, доступ к origin и к Docker.

set -euo pipefail

FULL_BUILD="${FULL_BUILD:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    --full|--no-cache)
      FULL_BUILD=1
      ;;
    --build)
      FORCE_BUILD=1
      ;;
    -h|--help)
      cat <<'EOF'
Обновление репозитория и применение изменений через docker compose.

  deploy-server.sh [путь]           — pull; restart при изменении .py/templates/static;
                                      build только если менялись Dockerfile/requirements.txt
  deploy-server.sh --build [путь]   — pull + docker compose build + up -d (кэш Docker)
  deploy-server.sh --full [путь]    — pull + build --no-cache + up -d
  FORCE_BUILD=1 / FULL_BUILD=1      — то же, что флаги выше

Код приложения смонтирован с хоста — для обычных обновлений пересборка не нужна.

Путь по умолчанию: /opt/Pusplexity.
EOF
      exit 0
      ;;
    -*)
      echo "Неизвестный флаг: $arg (допустимо: --build, --full, --no-cache, -h)" >&2
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
  exit 1
fi

cd "$APP_DIR"
echo "==> Каталог: $(pwd)"

PREV_HEAD="$(git rev-parse HEAD)"
echo "==> git pull origin main"
git pull origin main
POST_HEAD="$(git rev-parse HEAD)"

if [[ "$FULL_BUILD" == "1" ]]; then
  echo "==> docker compose build --no-cache (полная пересборка)"
  DOCKER_BUILDKIT=1 docker compose build --no-cache
  echo "==> docker compose up -d"
  docker compose up -d
elif [[ "$FORCE_BUILD" == "1" ]]; then
  echo "==> docker compose build (принудительная сборка, кэш Docker)"
  DOCKER_BUILDKIT=1 docker compose build
  echo "==> docker compose up -d"
  docker compose up -d
elif [[ "$PREV_HEAD" == "$POST_HEAD" ]]; then
  echo "==> Уже актуально (новых коммитов нет)"
else
  mapfile -t CHANGED < <(git diff --name-only "$PREV_HEAD" "$POST_HEAD")

  NEED_REBUILD=0
  NEED_RESTART=0
  NEED_UP=0

  for f in "${CHANGED[@]}"; do
    case "$f" in
      Dockerfile|requirements.txt)
        NEED_REBUILD=1
        ;;
      docker-compose.yml|.env.example)
        NEED_UP=1
        ;;
      *.py|templates/*|static/*)
        NEED_RESTART=1
        ;;
    esac
  done

  if [[ "$NEED_REBUILD" == "1" ]]; then
    echo "==> Изменены Dockerfile/requirements.txt — сборка образа (кэш Docker)"
    DOCKER_BUILDKIT=1 docker compose build
    docker compose up -d
  elif [[ "$NEED_RESTART" == "1" ]]; then
    echo "==> Изменён код приложения — перезапуск контейнеров (без сборки)"
    docker compose up -d
    docker compose restart pusplexity pusplexity-web
  elif [[ "$NEED_UP" == "1" ]]; then
    echo "==> Изменён docker-compose.yml — пересоздание контейнеров"
    docker compose up -d
  else
    echo "==> Изменения не затрагивают приложение: ${CHANGED[*]}"
  fi
fi

echo "==> docker compose ps"
docker compose ps

echo "==> Готово."
