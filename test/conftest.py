import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['mistralai'] = MagicMock()
sys.modules['requests'] = MagicMock()


@pytest.fixture
def mock_rag_system():
    """Mock du système RAG pour les tests"""
    mock = Mock()

    mock.query.return_value = {
        'answer': 'Voici une réponse de test',
        'sources': [
            {
                'event_id': 1,
                'title': 'Concert Test',
                'location': 'Paris'
            }
        ],
        'model_name': 'devstral-small',
        'nb_sources': 1,
        'question': 'Question de test'
    }

    mock.rebuild.return_value = {
        'status': 'success',
        'message': 'Base vectorielle reconstruite avec succès',
        'nb_events': 100,
        'nb_chunks': 500,
        'dimension': 1024
    }

    return mock


@pytest.fixture
def client(mock_rag_system):
    from api.routes import set_rag_system
    from api.main import app

    set_rag_system(mock_rag_system)

    yield TestClient(app)


@pytest.fixture
def client_no_rag():
    from api.routes import set_rag_system
    from api.main import app

    set_rag_system(None)

    yield TestClient(app)
