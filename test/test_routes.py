import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock


class TestHealthEndpoint:

    def test_health_check_with_rag(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["message"] == "API RAG opérationnelle"
        assert data["rag_loaded"] is True

    def test_health_check_without_rag(self, client_no_rag):
        response = client_no_rag.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["rag_loaded"] is False


class TestAskEndpoint:
    def test_ask_question_success(self, client):
        payload = {
            "question": "Je cherche un concert ce week-end",
            "top_k": 3,
            "model": "devstral-small"
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model_used" in data
        assert "nb_sources" in data
        assert "question" in data

        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        assert data["model_used"] == "devstral-small"

    def test_ask_question_with_defaults(self, client):
        payload = {
            "question": "Quel événement ce soir ?"
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] is not None

    def test_ask_question_missing_field(self, client):
        payload = {
            "top_k": 3
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 422

    def test_ask_question_invalid_top_k(self, client):
        payload = {
            "question": "Test question",
            "top_k": 0
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 422

    def test_ask_question_top_k_too_high(self, client):
        payload = {
            "question": "Test question",
            "top_k": 15 
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 422

    def test_ask_question_without_rag(self, client_no_rag):
        payload = {
            "question": "Test question"
        }

        response = client_no_rag.post("/ask", json=payload)
        assert response.status_code == 503
        assert "RAG non chargé" in response.json()["detail"]

    def test_ask_question_rag_error(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = ValueError("Erreur de validation")

        payload = {
            "question": "Test question"
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 400
        assert "Erreur de validation" in response.json()["detail"]

    def test_ask_question_rag_exception(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = Exception("Erreur interne")

        payload = {
            "question": "Test question"
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 500
        assert "Erreur lors de la génèration de la réponse" in response.json()["detail"]

    def test_ask_question_short_question(self, client):
        payload = {
            "question": "Hi" 
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 422


class TestRebuildEndpoint:

    def test_rebuild_success(self, client):
        response = client.post("/rebuild")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        assert "nb_events" in data
        assert "nb_chunks" in data
        assert "dimension" in data

        assert data["nb_events"] == 100
        assert data["nb_chunks"] == 500
        assert data["dimension"] == 1024

    def test_rebuild_without_rag(self, client_no_rag):
        response = client_no_rag.post("/rebuild")
        assert response.status_code == 503
        assert "RAG non chargé" in response.json()["detail"]

    def test_rebuild_rag_error(self, client, mock_rag_system):
        mock_rag_system.rebuild.return_value = {
            'status': 'error',
            'message': 'Erreur lors de la reconstruction',
            'nb_events': 0,
            'nb_chunks': 0,
            'dimension': 0
        }

        response = client.post("/rebuild")
        assert response.status_code == 500
        assert "Erreur lors de la reconstruction" in response.json()["detail"]

    def test_rebuild_exception(self, client, mock_rag_system):
        mock_rag_system.rebuild.side_effect = Exception("Erreur inattendue")

        response = client.post("/rebuild")
        assert response.status_code == 500
        assert "Erreur lors de la reconstruction" in response.json()["detail"]
