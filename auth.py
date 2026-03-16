"""
Простая аутентификация по email и паролю.
Пары email:пароль хранятся в users.txt.
"""

import os
from pathlib import Path

USERS_FILE = Path(__file__).parent / "users.txt"


def load_users() -> dict[str, str]:
    """Загружает пары email:пароль из users.txt."""
    result = {}
    path = Path(os.environ.get("USERS_FILE", USERS_FILE))
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            email, password = line.split(":", 1)
            result[email.strip().lower()] = password.strip()
    return result


def verify_credentials(email: str, password: str) -> bool:
    """Проверяет email и пароль. Возвращает True при успехе."""
    users = load_users()
    return users.get(email.strip().lower()) == password
