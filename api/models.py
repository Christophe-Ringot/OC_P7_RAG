from pydantic import BaseModel, Field
from typing import List


class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        description="Question à poser au système RAG",
        min_length=3,
        examples=["Je cherche un concert de musique ce week-end"]
    )
    top_k: int = Field(
        default=3,
        description="Nombre de documents à récupèrer pour le contexte",
        ge=1,
        le=10
    )
    model: str = Field(
        default="devstral-small",
        description="Modèle Mistral à utiliser pour la génération",
        examples=["devstral-small"]
    )


class Source(BaseModel):

    event_id: int = Field(..., description="ID de l'événement")
    title: str = Field(..., description="Titre de l'événement")
    location: str = Field(..., description="Ville de l'événement")


class AnswerResponse(BaseModel):

    answer: str = Field(..., description="Réponse générée par le système RAG")
    sources: List[Source] = Field(..., description="Sources utilisées pour générer la rèponse")
    model_used: str = Field(..., description="Nom du modèle Mistral utilisé")
    nb_sources: int = Field(..., description="Nombre de sources récupérées")
    question: str = Field(..., description="Question posée par l'utilisateur")


class RebuildResponse(BaseModel):
    status: str = Field(..., description="Statut de l'opèration (success/error)")
    message: str = Field(..., description="Message descriptif de l'opèration")
    nb_events: int = Field(..., description="Nombre d'évènements récupérès")
    nb_chunks: int = Field(..., description="Nombre de chunks créés")
    dimension: int = Field(..., description="Dimension des vecteurs d'embeddings")


class HealthResponse(BaseModel):

    status: str = Field(..., description="Statut de l'API")
    message: str = Field(..., description="Message de santé")
    rag_loaded: bool = Field(..., description="Système RAG chargé ou non")
