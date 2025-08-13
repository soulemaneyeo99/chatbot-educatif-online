"""
Dépendances FastAPI du Chatbot Éducatif Vocal
=============================================

Gestion centralisée des dépendances pour l'injection dans les routes FastAPI.
Singletons, cache, et initialisation paresseuse des services.

Utilisation:
    from app.dependencies import get_claude_client, get_vectorstore
    
    @app.get("/chat")
    async def chat(claude_client = Depends(get_claude_client)):
        pass
"""

from functools import lru_cache
from typing import Optional, Dict, Any
import os
from pathlib import Path

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.utils.logger import setup_logger, get_claude_logger, get_rag_logger
from app.utils.exceptions import (
    ConfigurationError,
    ClaudeAPIError,
    VectorstoreError,
    ChatbotException
)

# Import des services (seront créés dans les prochaines étapes)
# from app.services.openai_client import OpenAIClient
# from app.services.vectorstore import VectorStore
# from app.services.tts import TTSService
# from app.services.stt import STTService


# ═══════════════════════════════════════════════════════════════
# 🔐 SÉCURITÉ ET AUTHENTIFICATION
# ═══════════════════════════════════════════════════════════════

# Schéma Bearer token (optionnel pour la v1)
security = HTTPBearer(auto_error=False)

logger = setup_logger(__name__)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Authentification optionnelle via Bearer token.
    
    Pour la v1, l'authentification n'est pas obligatoire.
    Cette fonction peut être étendue plus tard pour une vraie auth.
    
    Args:
        credentials: Token Bearer optionnel
        
    Returns:
        Dict: Informations utilisateur ou None si pas authentifié
    """
    
    if not credentials:
        return None
    
    # TODO: Implémenter la vraie authentification
    # Pour l'instant, on accepte tout token non vide
    if credentials.credentials:
        return {
            "user_id": "anonymous",
            "token": credentials.credentials[:10] + "...",
            "authenticated": True
        }
    
    return None


def require_auth() -> Depends:
    """
    Dépendance pour forcer l'authentification.
    
    Returns:
        Depends: Dépendance qui lève une erreur si non authentifié
    """
    async def _require_auth(user = Depends(get_current_user)):
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentification requise",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user
    
    return Depends(_require_auth)


# ═══════════════════════════════════════════════════════════════
# ⚙️ CONFIGURATION ET VALIDATION
# ═══════════════════════════════════════════════════════════════

@lru_cache()
def get_validated_settings():
    """
    Retourne les paramètres validés avec vérifications supplémentaires.
    
    Returns:
        Settings: Configuration validée
        
    Raises:
        ConfigurationError: Si la configuration est invalide
    """
    settings = get_settings()
    
    # Vérifications critiques
    if not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY manquante",
            config_key="OPENAI_API_KEY",
            details="Cette clé est obligatoire pour utiliser l'assistant IA"
        )
    
    if not settings.OPENAI_API_KEY.startswith("sk-"):
        raise ConfigurationError(
            "OPENAI_API_KEY invalide",
            config_key="OPENAI_API_KEY", 
            details="La clé doit commencer par 'sk-'"
        )
           
    
    # Vérification des répertoires
    if not settings.DOCUMENTS_DIR.exists():
        logger.warning(f"Répertoire documents introuvable: {settings.DOCUMENTS_DIR}")
    
    if not settings.VECTORSTORE_DIR.exists():
        logger.warning(f"Répertoire vectorstore introuvable: {settings.VECTORSTORE_DIR}")
    
    return settings


# ═══════════════════════════════════════════════════════════════
# 🤖 OPENAI API CLIENT
# ═══════════════════════════════════════════════════════════════

# Cache global pour le client OpenAI
_openai_client_cache: Optional[Any] = None


def get_openai_client():
    """
    Retourne une instance singleton du client OpenAI API.
    
    Returns:
        OpenAIClient: Instance configurée du client
        
    Raises:
        ClaudeAPIError: Si impossible d'initialiser le client
    """
    global _openai_client_cache
    
    if _openai_client_cache is not None:
        return _openai_client_cache
    
    try:
        settings = get_validated_settings()
        claude_logger = get_claude_logger()  # On garde le même logger
        
        # Import dynamique pour éviter les dépendances circulaires
        from app.services.openai_client import OpenAIClient
        
        _openai_client_cache = OpenAIClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
            timeout=settings.OPENAI_TIMEOUT,
            logger=claude_logger
        )
        
        claude_logger.info(f"✅ Client OpenAI initialisé - Modèle: {settings.OPENAI_MODEL}")
        return _openai_client_cache
        
    except Exception as e:
        error_msg = f"Impossible d'initialiser le client OpenAI: {e}"
        logger.error(error_msg)
        
        raise ClaudeAPIError(
            message="Assistant IA indisponible",
            details=error_msg
        )


async def get_openai_client_async():
    """Version async de get_openai_client pour FastAPI."""
    return get_openai_client()


# Alias pour compatibilité
def get_claude_client():
    """Alias pour compatibilité - retourne le client OpenAI."""
    return get_openai_client()


async def get_claude_client_async():
    """Alias pour compatibilité - version async."""
    return get_openai_client()


# ═══════════════════════════════════════════════════════════════
# 📚 VECTORSTORE ET RAG
# ═══════════════════════════════════════════════════════════════

# Cache global pour le vectorstore
_vectorstore_cache: Optional[Any] = None


def get_vectorstore():
    """
    Retourne une instance singleton du vectorstore FAISS.
    
    Returns:
        VectorStore: Instance du vectorstore ou None si pas initialisé
        
    Raises:
        VectorstoreError: Si erreur lors de l'initialisation
    """
    global _vectorstore_cache
    
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    
    try:
        settings = get_validated_settings()
        rag_logger = get_rag_logger()
        
        # Vérifier si un index existe
        index_path = settings.vectorstore_path
        if not index_path.exists():
            rag_logger.warning("Aucun index FAISS trouvé - RAG non disponible")
            return None
        
        # Import dynamique
        from app.services.vectorstore import VectorStore
        
        _vectorstore_cache = VectorStore(
            vectorstore_dir=settings.VECTORSTORE_DIR,
            embedding_model=settings.EMBEDDING_MODEL,
            logger=rag_logger
        )
        
        rag_logger.info("✅ Vectorstore FAISS initialisé")
        return _vectorstore_cache
        
    except Exception as e:
        error_msg = f"Impossible d'initialiser le vectorstore: {e}"
        logger.error(error_msg)
        
        # En cas d'erreur, on ne lève pas d'exception
        # Le RAG devient simplement indisponible
        logger.warning("RAG indisponible - Mode fallback activé")
        return None


async def get_vectorstore_async():
    """Version async de get_vectorstore pour FastAPI."""
    return get_vectorstore()


def require_vectorstore():
    """
    Dépendance qui exige un vectorstore fonctionnel.
    
    Returns:
        Depends: Dépendance qui lève une erreur si vectorstore indisponible
    """
    def _require_vectorstore(vectorstore = Depends(get_vectorstore_async)):
        if vectorstore is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Base de connaissances indisponible. Ingérez des documents d'abord."
            )
        return vectorstore
    
    return _require_vectorstore


# ═══════════════════════════════════════════════════════════════
# 🔊 SERVICES AUDIO
# ═══════════════════════════════════════════════════════════════

# Cache global pour TTS
_tts_service_cache: Optional[Any] = None


def get_tts_service():
    """
    Retourne une instance singleton du service TTS.
    
    Returns:
        TTSService: Instance du service TTS ou None si pas configuré
    """
    global _tts_service_cache
    
    if _tts_service_cache is not None:
        return _tts_service_cache
    
    try:
        settings = get_validated_settings()
        
        if not settings.has_elevenlabs:
            logger.warning("ElevenLabs non configuré - TTS indisponible")
            return None
        
        # Import dynamique
        from app.services.tts import TTSService
        
        _tts_service_cache = TTSService(
            api_key=settings.ELEVENLABS_API_KEY,
            voice_id=settings.ELEVENLABS_VOICE_ID,
            model=settings.ELEVENLABS_MODEL,
            stability=settings.ELEVENLABS_STABILITY,
            similarity=settings.ELEVENLABS_SIMILARITY
        )
        
        logger.info("✅ Service TTS (ElevenLabs) initialisé")
        return _tts_service_cache
        
    except Exception as e:
        logger.error(f"Impossible d'initialiser TTS: {e}")
        return None


async def get_tts_service_async():
    """Version async de get_tts_service pour FastAPI."""
    return get_tts_service()


# Cache global pour STT
_stt_service_cache: Optional[Any] = None


def get_stt_service():
    """
    Retourne une instance singleton du service STT.
    
    Returns:
        STTService: Instance du service STT
    """
    global _stt_service_cache
    
    if _stt_service_cache is not None:
        return _stt_service_cache
    
    try:
        settings = get_validated_settings()
        
        # Import dynamique
        from app.services.stt import STTService
        
        _stt_service_cache = STTService(
            model=settings.WHISPER_MODEL,
            language=settings.WHISPER_LANGUAGE,
            timeout=settings.WHISPER_TIMEOUT
        )
        
        logger.info(f"✅ Service STT (Whisper) initialisé - Modèle: {settings.WHISPER_MODEL}")
        return _stt_service_cache
        
    except Exception as e:
        logger.error(f"Impossible d'initialiser STT: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de transcription indisponible"
        )


async def get_stt_service_async():
    """Version async de get_stt_service pour FastAPI."""
    return get_stt_service()


# ═══════════════════════════════════════════════════════════════
# 📊 MONITORING ET MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

from app.utils.logger import log_metrics


def get_metrics():
    """
    Retourne les métriques actuelles de l'application.
    
    Returns:
        Dict: Métriques système
    """
    return log_metrics.get_metrics()


async def get_health_info():
    """
    Retourne les informations de santé des services.
    
    Returns:
        Dict: État de santé de chaque service
    """
    settings = get_validated_settings()
    
    health_info = {
        "timestamp": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {}
    }
    
    # Test OpenAI API
    try:
        openai_client = get_openai_client()
        health_info["services"]["openai"] = {
            "status": "healthy",
            "model": settings.OPENAI_MODEL
        }
    except Exception as e:
        health_info["services"]["openai"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Test Vectorstore
    try:
        vectorstore = get_vectorstore()
        if vectorstore:
            health_info["services"]["vectorstore"] = {
                "status": "healthy",
                "documents_count": getattr(vectorstore, 'document_count', 0)
            }
        else:
            health_info["services"]["vectorstore"] = {
                "status": "not_configured",
                "message": "Aucun document ingéré"
            }
    except Exception as e:
        health_info["services"]["vectorstore"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Test TTS
    try:
        tts_service = get_tts_service()
        if tts_service:
            health_info["services"]["tts"] = {
                "status": "healthy",
                "provider": "elevenlabs"
            }
        else:
            health_info["services"]["tts"] = {
                "status": "not_configured",
                "message": "ElevenLabs non configuré"
            }
    except Exception as e:
        health_info["services"]["tts"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Test STT
    try:
        stt_service = get_stt_service()
        health_info["services"]["stt"] = {
            "status": "healthy",
            "provider": "whisper",
            "model": settings.WHISPER_MODEL
        }
    except Exception as e:
        health_info["services"]["stt"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_info


# ═══════════════════════════════════════════════════════════════
# 🧹 NETTOYAGE ET FERMETURE
# ═══════════════════════════════════════════════════════════════

def cleanup_dependencies():
    """
    Nettoie toutes les dépendances cachées.
    À appeler lors de l'arrêt de l'application.
    """
    global _openai_client_cache, _vectorstore_cache, _tts_service_cache, _stt_service_cache
    
    logger.info("🧹 Nettoyage des dépendances...")
    
    # Nettoyage des caches
    if _openai_client_cache:
        try:
            if hasattr(_openai_client_cache, 'close'):
                _openai_client_cache.close()
            _openai_client_cache = None
            logger.info("✅ Client OpenAI fermé")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture OpenAI: {e}")
    
    if _vectorstore_cache:
        try:
            if hasattr(_vectorstore_cache, 'close'):
                _vectorstore_cache.close()
            _vectorstore_cache = None
            logger.info("✅ Vectorstore fermé")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture vectorstore: {e}")
    
    if _tts_service_cache:
        try:
            if hasattr(_tts_service_cache, 'close'):
                _tts_service_cache.close()
            _tts_service_cache = None
            logger.info("✅ Service TTS fermé")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture TTS: {e}")
    
    if _stt_service_cache:
        try:
            if hasattr(_stt_service_cache, 'close'):
                _stt_service_cache.close()
            _stt_service_cache = None
            logger.info("✅ Service STT fermé")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture STT: {e}")
    
    logger.info("🎉 Nettoyage terminé")


# ═══════════════════════════════════════════════════════════════
# 🔄 VALIDATION ET REDÉMARRAGE
# ═══════════════════════════════════════════════════════════════

def reset_all_caches():
    """
    Remet à zéro tous les caches. Utile pour le rechargement à chaud.
    """
    global _openai_client_cache, _vectorstore_cache, _tts_service_cache, _stt_service_cache
    
    logger.info("🔄 Reset des caches...")
    
    _openai_client_cache = None
    _vectorstore_cache = None
    _tts_service_cache = None
    _stt_service_cache = None
    
    # Reset aussi le cache des settings
    get_validated_settings.cache_clear()
    
    logger.info("✅ Caches remis à zéro")


def validate_all_services() -> Dict[str, Any]:
    """
    Valide que tous les services essentiels sont fonctionnels.
    
    Returns:
        Dict: Rapport de validation
    """
    logger.info("🔍 Validation de tous les services...")
    
    validation_report = {
        "timestamp": "2024-01-01T00:00:00Z",  # Sera remplacé
        "overall_status": "healthy",
        "services": {}
    }
    
    services_to_test = [
        ("openai", get_openai_client),
        ("vectorstore", get_vectorstore),
        ("tts", get_tts_service),
        ("stt", get_stt_service)
    ]
    
    failed_services = []
    
    for service_name, service_getter in services_to_test:
        try:
            service = service_getter()
            if service is not None:
                validation_report["services"][service_name] = {
                    "status": "healthy",
                    "initialized": True
                }
                logger.info(f"✅ {service_name}: OK")
            else:
                validation_report["services"][service_name] = {
                    "status": "not_configured",
                    "initialized": False
                }
                logger.warning(f"⚠️ {service_name}: Non configuré")
        except Exception as e:
            validation_report["services"][service_name] = {
                "status": "unhealthy",
                "error": str(e),
                "initialized": False
            }
            failed_services.append(service_name)
            logger.error(f"❌ {service_name}: {e}")
    
    # Statut global
    if failed_services:
        if "openai" in failed_services:
            validation_report["overall_status"] = "critical"
        else:
            validation_report["overall_status"] = "degraded"
    
    validation_report["failed_services"] = failed_services
    
    logger.info(f"🎯 Validation terminée - Statut: {validation_report['overall_status']}")
    
    return validation_report


# ═══════════════════════════════════════════════════════════════
# 🎛️ DÉPENDANCES DE DÉVELOPPEMENT
# ═══════════════════════════════════════════════════════════════

def get_debug_info() -> Dict[str, Any]:
    """
    Retourne des informations de débogage. Uniquement en mode DEBUG.
    
    Returns:
        Dict: Informations de debug
    """
    settings = get_validated_settings()
    
    if not settings.DEBUG:
        return {"message": "Debug mode disabled"}
    
    debug_info = {
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith("CLAUDE") or key.startswith("ELEVENLABS") or key.startswith("DEBUG")
        },
        "cache_status": {
            "openai_client": _openai_client_cache is not None,
            "vectorstore": _vectorstore_cache is not None,
            "tts_service": _tts_service_cache is not None,
            "stt_service": _stt_service_cache is not None
        },
        "settings": {
            "openai_model": settings.OPENAI_MODEL,
            "whisper_model": settings.WHISPER_MODEL,
            "has_elevenlabs": settings.has_elevenlabs,
            "documents_dir": str(settings.DOCUMENTS_DIR),
            "vectorstore_dir": str(settings.VECTORSTORE_DIR)
        },
        "paths": {
            "documents_exists": settings.DOCUMENTS_DIR.exists(),
            "vectorstore_exists": settings.VECTORSTORE_DIR.exists(),
            "index_exists": settings.vectorstore_path.exists()
        }
    }
    
    return debug_info


# ═══════════════════════════════════════════════════════════════
# 📝 DÉPENDANCES POUR LA VALIDATION DES REQUÊTES
# ═══════════════════════════════════════════════════════════════

async def validate_audio_file(
    content_type: str,
    content_length: int
) -> None:
    """
    Valide un fichier audio uploadé.
    
    Args:
        content_type: Type MIME du fichier
        content_length: Taille du fichier en bytes
        
    Raises:
        HTTPException: Si le fichier est invalide
    """
    settings = get_validated_settings()
    
    # Vérification du type MIME
    allowed_mimes = [
        "audio/mpeg",  # MP3
        "audio/wav",   # WAV
        "audio/ogg",   # OGG
        "audio/mp4",   # M4A
        "audio/flac"   # FLAC
    ]
    
    if content_type not in allowed_mimes:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Format audio non supporté: {content_type}"
        )
    
    # Vérification de la taille
    max_size = settings.MAX_UPLOAD_SIZE
    if content_length > max_size:
        size_mb = content_length / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux ({size_mb:.1f}MB). Maximum: {max_mb:.1f}MB"
        )


async def validate_text_input(text: str) -> str:
    """
    Valide et nettoie un texte d'entrée.
    
    Args:
        text: Texte à valider
        
    Returns:
        str: Texte nettoyé
        
    Raises:
        HTTPException: Si le texte est invalide
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le texte ne peut pas être vide"
        )
    
    # Nettoyage
    cleaned_text = text.strip()
    
    # Vérification de la longueur
    max_length = 5000  # 5000 caractères max
    if len(cleaned_text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Texte trop long ({len(cleaned_text)} caractères). Maximum: {max_length}"
        )
    
    # Filtrage de contenu basique (peut être étendu)
    inappropriate_patterns = [
        # Ajoutez ici des patterns à filtrer si nécessaire
    ]
    
    text_lower = cleaned_text.lower()
    for pattern in inappropriate_patterns:
        if pattern in text_lower:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Contenu inapproprié détecté"
            )
    
    return cleaned_text


# ═══════════════════════════════════════════════════════════════
# 🚀 INITIALISATION FORCÉE (POUR LES TESTS)
# ═══════════════════════════════════════════════════════════════

def force_initialize_all_services():
    """
    Force l'initialisation de tous les services. Utile pour les tests.
    
    Returns:
        Dict: Résultat de l'initialisation
    """
    logger.info("🚀 Initialisation forcée de tous les services...")
    
    results = {}
    
    try:
        openai_client = get_openai_client()
        results["openai"] = "success"
    except Exception as e:
        results["openai"] = f"failed: {e}"
    
    try:
        vectorstore = get_vectorstore()
        results["vectorstore"] = "success" if vectorstore else "not_configured"
    except Exception as e:
        results["vectorstore"] = f"failed: {e}"
    
    try:
        tts_service = get_tts_service()
        results["tts"] = "success" if tts_service else "not_configured"
    except Exception as e:
        results["tts"] = f"failed: {e}"
    
    try:
        stt_service = get_stt_service()
        results["stt"] = "success"
    except Exception as e:
        results["stt"] = f"failed: {e}"
    
    logger.info(f"🎯 Initialisation terminée: {results}")
    return results


# ═══════════════════════════════════════════════════════════════
# 🧪 UTILITAIRES DE TEST
# ═══════════════════════════════════════════════════════════════

def create_test_dependencies():
    """
    Crée des mocks de dépendances pour les tests.
    
    Returns:
        Dict: Dépendances de test
    """
    from unittest.mock import Mock
    
    mock_openai = Mock()
    mock_openai.generate_response.return_value = "Réponse de test"
    
    mock_vectorstore = Mock()
    mock_vectorstore.search.return_value = [
        {"content": "Document de test", "score": 0.9}
    ]
    
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b"audio_data_mock"
    
    mock_stt = Mock()
    mock_stt.transcribe.return_value = "Transcription de test"
    
    return {
        "openai_client": mock_openai,
        "vectorstore": mock_vectorstore,
        "tts_service": mock_tts,
        "stt_service": mock_stt
    }


# ═══════════════════════════════════════════════════════════════
# 📋 EXPORTS PRINCIPAUX
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "get_validated_settings",
    
    # Authentification
    "get_current_user",
    "require_auth",
    
    # Services principaux
    "get_openai_client",
    "get_openai_client_async",
    "get_claude_client",  # Alias pour compatibilité
    "get_claude_client_async",  # Alias pour compatibilité
    "get_vectorstore",
    "get_vectorstore_async",
    "require_vectorstore",
    "get_tts_service",
    "get_tts_service_async",
    "get_stt_service",
    "get_stt_service_async",
    
    # Monitoring
    "get_metrics",
    "get_health_info",
    
    # Validation
    "validate_audio_file",
    "validate_text_input",
    
    # Utilitaires
    "cleanup_dependencies",
    "reset_all_caches",
    "validate_all_services",
    "force_initialize_all_services"
]