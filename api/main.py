from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from rag.rag_system import RAGSystem
from api.routes import router, set_rag_system


rag_system_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charge le systéme au démarrage et néttoie à l'arrêt
    global rag_system_instance

    try:
        rag_system_instance = RAGSystem()
        set_rag_system(rag_system_instance)
        print("Documentation : http://localhost:8000/docs")

    except Exception as e:
        print(f"Erreur : {e}")

    yield
    rag_system_instance = None

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Endpoint racine pour rediriger vers la documentation
@app.get("/", tags=["Root"])
async def root():
    return {
        "documentation": "/docs",
        "endpoints": {
            "ask": "POST /ask",
            "rebuild": "POST /rebuild",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
