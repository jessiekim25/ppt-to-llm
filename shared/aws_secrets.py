"""Lazy dict-like proxy over a single AWS Secrets Manager secret.

The secret is expected to be a JSON object of string -> string. Values are
loaded on first access and cached for the life of the process.

Override the secret name with the PPT_TO_LLM_SECRET_NAME environment
variable; region follows the standard AWS chain (AWS_REGION, ~/.aws/config).
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

DEFAULT_SECRET_NAME = "ppt-to-llm"


class _Secrets:
    def __init__(self, name: str) -> None:
        self._name = name
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is None:
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=self._name)
            payload = response.get("SecretString")
            if payload is None:
                raise RuntimeError(
                    f"Secret {self._name!r} has no SecretString (binary secrets not supported)."
                )
            self._cache = json.loads(payload)
        return self._cache

    def __getitem__(self, key: str) -> str:
        return self._load()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._load()


secrets = _Secrets(os.environ.get("PPT_TO_LLM_SECRET_NAME", DEFAULT_SECRET_NAME))
