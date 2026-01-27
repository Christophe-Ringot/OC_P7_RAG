import pytest
from pydantic import ValidationError
from api.models import (
    QuestionRequest,
    AnswerResponse,
    RebuildResponse,
    HealthResponse,
    Source
)


class TestQuestionRequest:

    def test_valid_question_request_with_defaults(self):
        request = QuestionRequest(question="Quel concert ce soir ?")

        assert request.question == "Quel concert ce soir ?"
        assert request.top_k == 3 
        assert request.model == "devstral-small"

    def test_valid_question_request_with_custom_values(self):
        request = QuestionRequest(
            question="Concert de jazz à Lille",
            top_k=5,
            model="devstral-small"
        )

        assert request.question == "Concert de jazz à Lille"
        assert request.top_k == 5
        assert request.model == "devstral-small"

    def test_question_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            QuestionRequest(question="Hi")

        errors = exc_info.value.errors()
        assert any(error['loc'] == ('question',) for error in errors)

    def test_missing_question(self):
        with pytest.raises(ValidationError) as exc_info:
            QuestionRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'] == ('question',) for error in errors)

    def test_top_k_below_minimum(self):
        with pytest.raises(ValidationError) as exc_info:
            QuestionRequest(question="Test question", top_k=0)

        errors = exc_info.value.errors()
        assert any(error['loc'] == ('top_k',) for error in errors)

    def test_top_k_above_maximum(self):
        with pytest.raises(ValidationError) as exc_info:
            QuestionRequest(question="Test question", top_k=15)

        errors = exc_info.value.errors()
        assert any(error['loc'] == ('top_k',) for error in errors)

    def test_top_k_boundary_values(self):
        request_min = QuestionRequest(question="Test", top_k=1)
        assert request_min.top_k == 1

        request_max = QuestionRequest(question="Test", top_k=10)
        assert request_max.top_k == 10


class TestSource:
    def test_valid_source(self):
        source = Source(
            event_id=123,
            title="Concert de Jazz",
            location="Lille"
        )

        assert source.event_id == 123
        assert source.title == "Concert de Jazz"
        assert source.location == "Lille"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Source(event_id=123)

        with pytest.raises(ValidationError):
            Source(title="Concert")

        with pytest.raises(ValidationError):
            Source(location="Lille")


class TestAnswerResponse:
    def test_valid_answer_response(self):
        sources = [
            Source(event_id=1, title="Concert A", location="Lille"),
            Source(event_id=2, title="Concert B", location="Lille")
        ]

        response = AnswerResponse(
            answer="Voici les concerts disponibles",
            sources=sources,
            model_used="devstral-small",
            nb_sources=2,
            question="Concerts ce week-end ?"
        )

        assert response.answer == "Voici les concerts disponibles"
        assert len(response.sources) == 2
        assert response.model_used == "devstral-small"
        assert response.nb_sources == 2
        assert response.question == "Concerts ce week-end ?"

    def test_answer_response_empty_sources(self):
        response = AnswerResponse(
            answer="Aucun résultat",
            sources=[],
            model_used="devstral-small",
            nb_sources=0,
            question="Question test"
        )

        assert response.sources == []
        assert response.nb_sources == 0

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AnswerResponse(
                answer="Test",
                sources=[]
            )


class TestRebuildResponse:
    def test_valid_rebuild_response_success(self):
        response = RebuildResponse(
            status="success",
            message="Base vectorielle reconstruite",
            nb_events=100,
            nb_chunks=500,
            dimension=1024
        )

        assert response.status == "success"
        assert response.message == "Base vectorielle reconstruite"
        assert response.nb_events == 100
        assert response.nb_chunks == 500
        assert response.dimension == 1024

    def test_valid_rebuild_response_error(self):
        response = RebuildResponse(
            status="error",
            message="Erreur lors de la reconstruction",
            nb_events=0,
            nb_chunks=0,
            dimension=0
        )

        assert response.status == "error"
        assert response.nb_events == 0

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            RebuildResponse(
                status="success",
                message="Test"
            )


class TestHealthResponse:
    def test_valid_health_response_ok(self):
        response = HealthResponse(
            status="ok",
            message="API RAG opérationnelle",
            rag_loaded=True
        )

        assert response.status == "ok"
        assert response.message == "API RAG opérationnelle"
        assert response.rag_loaded is True

    def test_valid_health_response_rag_not_loaded(self):
        response = HealthResponse(
            status="ok",
            message="API en cours de chargement",
            rag_loaded=False
        )

        assert response.status == "ok"
        assert response.rag_loaded is False

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="ok")

        with pytest.raises(ValidationError):
            HealthResponse(
                status="ok",
                message="Test"
            )
