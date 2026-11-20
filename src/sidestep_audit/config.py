import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    key_env: str = ""
    model_alias: str = ""
    filter_risk: int = Field(default=50, ge=0, le=100)
    price_usd_per_1m_tokens_in: float = Field(default=0.0, ge=0)
    price_usd_per_1m_tokens_out: float = Field(default=0.0, ge=0)
    enabled: bool = True
    supports_json_schema: bool = False
    supports_n: bool = True
    extra_headers: dict[str, str] = {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIDESTEP_", env_file=".env", extra="ignore")

    token: str = ""
    database_path: str = ""
    upstream_base_url: str = ""
    provider_name: str = ""
    upstream_key_env: str = ""
    timeout_seconds: float | None = None
    providers: list[ProviderConfig] = Field(default_factory=list)
    default_provider: str = ""
    observe_only: bool | None = None
    max_attempts: int = 12
    max_cost_usd: float = 0.25
    deadline_ms: int = 45000


def load_settings() -> Settings:
    config_path = Path(os.getenv("SIDESTEP_CONFIG", "config.yaml"))
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    if not isinstance(raw, dict):
        raw = {}
    gateway = raw.get("gateway") or {}
    upstream = raw.get("upstream") or {}
    configured = Settings()
    provider_rows = raw.get("providers") or []
    providers = [ProviderConfig.model_validate(item) for item in provider_rows]
    if not providers and upstream:
        providers = [ProviderConfig(
            name=str(upstream.get("name", "upstream")),
            base_url=str(upstream.get("base_url", "http://127.0.0.1:4100/v1")),
            key_env=str(upstream.get("key_env", "")),
            model_alias=str(upstream.get("model_alias", "")),
            filter_risk=int(upstream.get("filter_risk", 50)),
            price_usd_per_1m_tokens_in=float(upstream.get("price_usd_per_1m_tokens_in", 0.0)),
            price_usd_per_1m_tokens_out=float(upstream.get("price_usd_per_1m_tokens_out", 0.0)),
        )]
    default_provider = configured.default_provider or str(raw.get("default_provider", ""))
    if not default_provider and upstream:
        default_provider = str(upstream.get("name", "upstream"))
    fallback_provider = next((provider for provider in providers if provider.name == default_provider), None)
    if fallback_provider is None and providers:
        fallback_provider = providers[0]
    legacy_base_url = upstream.get("base_url") or (fallback_provider.base_url if fallback_provider else "")
    legacy_provider_name = upstream.get("name") or (fallback_provider.name if fallback_provider else default_provider)
    legacy_key_env = upstream.get("key_env") or (fallback_provider.key_env if fallback_provider else "")
    observe_only = configured.observe_only
    if observe_only is None:
        observe_only = bool(gateway.get("observe_only", False))
    timeout = configured.timeout_seconds or gateway.get("request_timeout_seconds", 45.0)
    return Settings(
        token=configured.token,
        database_path=configured.database_path or str(gateway.get("database_path", "/tmp/sidestep.sqlite3")),
        upstream_base_url=(configured.upstream_base_url or str(legacy_base_url)).rstrip("/"),
        provider_name=configured.provider_name or str(legacy_provider_name or ""),
        upstream_key_env=configured.upstream_key_env or str(legacy_key_env),
        timeout_seconds=float(timeout),
        providers=providers,
        default_provider=default_provider,
        observe_only=observe_only,
        max_attempts=max(1, min(16, configured.max_attempts)),
        max_cost_usd=max(0.0, configured.max_cost_usd),
        deadline_ms=max(1, configured.deadline_ms),
    )
