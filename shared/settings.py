import os
from dataclasses import dataclass

from shared.aws_secrets import get_secret

LLM_SECRET_NAME = os.environ.get("LLM_SECRET_NAME", "LLMKeys")
MYSQL_SECRET_NAME = os.environ.get("MYSQL_SECRET_NAME", "MySQL")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str


def _safe_get_secret(name: str) -> dict:
    """Return the secret if reachable, else an empty dict.

    MySQL upload of test-slide rows is opt-in; a missing secret means the
    caller runs without an upload sink rather than failing extraction.
    """
    try:
        return get_secret(name)
    except Exception:
        return {}


def get_settings() -> Settings:
    """Centralized settings accessor backed by AWS Secrets Manager."""
    llm = get_secret(LLM_SECRET_NAME)
    mysql = _safe_get_secret(MYSQL_SECRET_NAME)
    return Settings(
        openai_api_key=llm["OPENAI_API_KEY"],
        openai_model=llm.get("OPENAI_MODEL", "gpt-4o"),
        mysql_host=str(mysql.get("MYSQL_HOST", "")),
        mysql_port=int(mysql.get("MYSQL_PORT", 3306) or 3306),
        mysql_user=str(mysql.get("MYSQL_USER", "")),
        mysql_password=str(mysql.get("MYSQL_PASSWORD", "")),
        mysql_database=str(mysql.get("MYSQL_DATABASE", "cro")),
    )
