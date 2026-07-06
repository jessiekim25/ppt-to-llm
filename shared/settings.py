from dataclasses import dataclass

from shared.aws_secrets import secrets


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str


def get_settings() -> Settings:
    """Centralized settings accessor backed by AWS Secrets Manager."""
    return Settings(
        openai_api_key=secrets["OPENAI_API_KEY"],
        openai_model=secrets.get("OPENAI_MODEL") or "gpt-4o",
        mysql_host=secrets.get("MYSQL_HOST") or "localhost",
        mysql_port=int(secrets.get("MYSQL_PORT") or 3306),
        mysql_user=secrets["MYSQL_USER"],
        mysql_password=secrets["MYSQL_PASSWORD"],
        mysql_database=secrets.get("MYSQL_DATABASE") or "jihwi",
    )
