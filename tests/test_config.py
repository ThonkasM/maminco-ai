"""Settings validation: the service must refuse to boot with a weak token."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_token_is_required(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_SERVICE_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        make()


@pytest.mark.parametrize("token", ["", "short", "x" * 31, "change-me", "Change-Me" + "x" * 40])
def test_weak_or_placeholder_token_is_rejected(token):
    with pytest.raises(ValidationError):
        make(internal_service_token=token)


def test_valid_token_is_accepted():
    assert make(internal_service_token="a" * 32).internal_service_token == "a" * 32


def test_timezone_defaults_to_la_paz_and_validates():
    assert make(internal_service_token="a" * 32).timezone == "America/La_Paz"
    assert make(internal_service_token="a" * 32, timezone="America/Lima").timezone == "America/Lima"
    with pytest.raises(ValidationError):
        make(internal_service_token="a" * 32, timezone="Mars/Olympus")
