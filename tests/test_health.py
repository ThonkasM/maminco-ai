"""Health endpoint tests (no auth required)."""

from __future__ import annotations


def test_health_returns_ok_without_token(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "maminco-ai"
    assert "version" in body
    assert body["models_loaded"] is False
