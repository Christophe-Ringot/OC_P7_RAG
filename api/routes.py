from fastapi import APIRouter, HTTPException
from api.models import (
    QuestionRequest,
    AnswerResponse,
    RebuildResponse,
    HealthResponse,
    Source
)

router = APIRouter()

rag_system = None


def set_rag_system(system):
    global rag_system
    rag_system = system


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        message="API RAG opérationnelle",
        rag_loaded=rag_system is not None
    )


@router.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="Systéme RAG non chargé. Veuillez réessayer plus tard."
        )

    try:
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            model_key=request.model
        )

        # Conversion des sources au format Pydantic
        sources = [Source(**source) for source in result['sources']]

        return AnswerResponse(
            answer=result['answer'],
            sources=sources,
            model_used=result['model_name'],
            nb_sources=result['nb_sources'],
            question=result['question']
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génèration de la réponse : {str(e)}"
        )


@router.post("/rebuild", response_model=RebuildResponse, tags=["Admin"])
async def rebuild_vector_store():
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="Systéme RAG non chargé. Veuillez réessayer plus tard."
        )

    try:
        result = rag_system.rebuild()

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])

        return RebuildResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la reconstruction : {str(e)}"
        )
