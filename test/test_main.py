import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "documentation" in data
    assert data["documentation"] == "/docs"
    assert "endpoints" in data
    assert "ask" in data["endpoints"]
    assert "rebuild" in data["endpoints"]
    assert "health" in data["endpoints"]


def test_cors_middleware(client):
    response = client.options("/health")
    assert response.status_code in [200, 405]


def test_app_structure(client):
    response = client.get("/")
    assert response.status_code == 200

    response_health = client.get("/health")
    assert response_health.status_code == 200
