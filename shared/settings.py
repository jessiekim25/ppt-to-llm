import os
from dataclasses import dataclass

from shared.aws_secrets import get_secret

MYSQL_SECRET_NAME = os.environ.get("MYSQL_SECRET_NAME", "MySQL")
LLM_SECRET_NAME = os.environ.get("LLM_SECRET_NAME", "LLMKeys")


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
    mysql = get_secret(MYSQL_SECRET_NAME)
    llm = get_secret(LLM_SECRET_NAME)
    return Settings(
        openai_api_key=llm["OPENAI_API_KEY"],
        openai_model="gpt-4o",
        mysql_host=mysql["RDS_HOSTNAME"],
        mysql_port=int(3306),
        mysql_user=mysql["RDS_USERNAME_TESTDB"],
        mysql_password=mysql["RDS_PASSWORD_TESTDB"],
        mysql_database="jihwi",
    )