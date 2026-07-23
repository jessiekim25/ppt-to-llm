import os
from dataclasses import dataclass

from shared.aws_secrets import get_secret

LLM_SECRET_NAME = os.environ.get("LLM_SECRET_NAME", "LLMKeys")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str


def get_settings() -> Settings:
    """Centralized settings accessor backed by AWS Secrets Manager."""
    llm = get_secret(LLM_SECRET_NAME)
    return Settings(
        openai_api_key=llm["OPENAI_API_KEY"],
        openai_model=llm.get("OPENAI_MODEL", "gpt-4o"),
    )
