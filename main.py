"""
Chatbot Éducatif Vocal - Point d'entrée principal
=================================================

Application FastAPI pour un chatbot éducatif destiné aux personnes peu alphabétisées.
Architecture RAG avec Claude API, ElevenLabs TTS, et Whisper STT.

Auteur: Votre nom
Version: 1.0.0
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.api.routes import api_router
from app.utils.logger import setup_logger
from app.utils.exceptions import ChatbotException
from app.dependencies import get_openai_client, get_vectorstore


# Configuration des logs
logger = setup_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application FastAPI.
    
    Initialise les ressources au démarrage et les nettoie à l'arrêt.
    """
    logger.info("🚀 Démarrage du Chatbot Éducatif Vocal...")
    
    try:
        # ✅ Initialisation des services critiques
        logger.info("🔧 Initialisation des services...")
        
        # Test de connexion OpenAI API
        openai_client = get_openai_client()
        logger.info("✅ OpenAI API client initialisé")
        
        # Initialisation du vectorstore (si des documents existent)
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                logger.info("✅ Vectorstore FAISS initialisé")
            else:
                logger.warning("⚠️ Aucun vectorstore trouvé - pensez à ingérer des documents")
        except Exception as e:
            logger.warning(f"⚠️ Vectorstore non disponible: {e}")
        
        # Vérification des clés API
        if not settings.OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY manquante dans .env")
            # Ne pas utiliser sys.exit() dans le lifespan, lever une exception
            raise RuntimeError("OPENAI_API_KEY manquante dans la configuration")
            
        if not settings.has_elevenlabs:
            logger.warning("⚠️ ELEVENLABS_API_KEY manquante - TTS désactivé")
        
        logger.info("🎉 Application démarrée avec succès!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        # Lever l'exception au lieu d'utiliser sys.exit()
        raise RuntimeError(f"Impossible d'initialiser l'application: {e}")
    
    finally:
        # 🧹 Nettoyage des ressources
        logger.info("🛑 Arrêt du Chatbot Éducatif Vocal...")
        logger.info("✅ Nettoyage terminé")


# 🏗️ Création de l'application FastAPI
app = FastAPI(
    title="Chatbot Éducatif Vocal",
    description="""
    API pour un chatbot éducatif vocal destiné aux personnes peu alphabétisées.
    
    ## Fonctionnalités principales
    
    * **Chat intelligent** : Réponses basées sur Claude API + RAG
    * **Interaction vocale** : Speech-to-Text et Text-to-Speech
    * **Base documentaire** : Ingestion et recherche dans documents Markdown
    * **Santé système** : Monitoring et diagnostics
    
    ## Utilisation typique
    
    1. Ingérer des documents éducatifs via `/ingest`
    2. Transcriber la voix utilisateur via `/stt` 
    3. Obtenir une réponse intelligente via `/chat`
    4. Convertir la réponse en audio via `/tts`
    """,
    version="1.0.0",
    contact={
        "name": "Équipe Chatbot Éducatif",
        "email": "contact@chatbot-educatif.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


# 🌐 Configuration CORS étendue pour mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://10.0.2.2:8000",  # Émulateur Android
        "http://localhost:8000",  # Tests locaux
        "*"  # TEMPORAIRE pour développement - À restreindre en production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# 🛡️ Gestionnaires d'erreurs globaux

@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException) -> JSONResponse:
    """Gestionnaire pour les erreurs métier du chatbot."""
    logger.error(f"Erreur chatbot: {exc.message} | Détails: {exc.details}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "chatbot_error",
                "message": exc.message,
                "details": exc.details,
                "timestamp": exc.timestamp.isoformat(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Gestionnaire pour les erreurs HTTP standard."""
    logger.warning(f"Erreur HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Gestionnaire pour les erreurs de validation Pydantic."""
    logger.warning(f"Erreur de validation: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "validation_error",
                "message": "Données de requête invalides",
                "details": exc.errors(),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Gestionnaire pour toutes les autres erreurs non gérées."""
    logger.error(f"Erreur inattendue: {type(exc).__name__}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_error",
                "message": "Une erreur interne s'est produite",
                "details": str(exc) if settings.DEBUG else "Contactez l'administrateur",
                "request_id": getattr(request.state, "request_id", None)
            }
        }
    )


# 📡 Middleware de logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logger les requêtes HTTP."""
    import time
    import uuid
    
    # Génération d'un ID unique pour la requête
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Log de la requête entrante
    logger.info(
        f"🔵 [{request_id}] {request.method} {request.url.path} "
        f"| IP: {request.client.host if request.client else 'unknown'}"
    )
    
    # Traitement de la requête
    response = await call_next(request)
    
    # Calcul du temps de traitement
    process_time = time.time() - start_time
    
    # Log de la réponse
    status_emoji = "🟢" if response.status_code < 400 else "🔴"
    logger.info(
        f"{status_emoji} [{request_id}] {response.status_code} | "
        f"Temps: {process_time:.3f}s"
    )
    
    # Ajout de l'ID dans les headers de réponse
    response.headers["X-Request-ID"] = request_id
    
    return response


# 🔗 Inclusion des routes
app.include_router(api_router, prefix="/api/v1")


# 🏠 Route racine
@app.get(
    "/",
    summary="Page d'accueil",
    description="Point d'entrée principal de l'API",
    response_model=Dict[str, Any]
)
async def root() -> Dict[str, Any]:
    """
    Page d'accueil de l'API avec informations de base.
    """
    return {
        "message": "🎓 Bienvenue sur le Chatbot Éducatif Vocal",
        "description": "API pour l'éducation vocale des personnes peu alphabétisées",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat",
            "stt": "/api/v1/stt", 
            "tts": "/api/v1/tts",
            "ingest": "/api/v1/ingest",
            "health": "/api/v1/health"
        }
    }


# 🚀 Point d'entrée principal
if __name__ == "__main__":
    logger.info("🎯 Lancement en mode développement...")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG,
    )