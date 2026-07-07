"""Fetch JSON secrets from AWS Secrets Manager, cached per name for process lifetime.

Each secret is expected to be a JSON object of string -> string. AWS credentials
and region come from the standard boto3 chain (AWS_PROFILE, env vars, IAM role,
~/.aws/config).
"""

from __future__ import annotations

import json
from functools import lru_cache

import boto3


@lru_cache(maxsize=None)
def get_secret(name: str) -> dict[str, str]:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=name)
    payload = response.get("SecretString")
    if payload is None:
        raise RuntimeError(f"Secret {name!r} has no SecretString (binary secrets not supported).")
    return json.loads(payload)
