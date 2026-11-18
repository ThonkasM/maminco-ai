"""Internal-token authentication tests."""

from __future__ import annotations

from tests.conftest import AUTH_HEADERS

PROTECTED_ROUTES = [
    "/forecast/sales",
    "/forecast/products",
    "/forecast/peak-hours",
    "/insights/anomalies",
    "/metrics/summary",
]


def test_missing_token_is_rejected(client):
    for route in PROTECTED_ROUTES:
        response = client.get(route)
        assert response.status_code == 401, route
        assert "X-Internal-Token" in response.json()["detail"]


def test_wrong_token_is_rejected(client):
    response = client.get("/forecast/sales", headers={"X-Internal-Token": "nope"})
    assert response.status_code == 401


def test_valid_token_is_accepted(client):
    for route in PROTECTED_ROUTES:
        response = client.get(route, headers=AUTH_HEADERS)
        assert response.status_code == 200, route


def test_train_requires_token(client):
    assert client.post("/train").status_code == 401
    assert client.post("/train", headers=AUTH_HEADERS).status_code == 200


def test_empty_token_is_rejected(client):
    response = client.get("/forecast/sales", headers={"X-Internal-Token": ""})
    assert response.status_code == 401


def test_non_ascii_token_is_rejected_not_500(client):
    response = client.get(
        "/forecast/sales",
        headers={"X-Internal-Token": "tóken-ñ".encode("latin-1")},
    )
    assert response.status_code == 401
