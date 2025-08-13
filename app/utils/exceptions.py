"""
Exceptions Métier du Chatbot Éducatif Vocal
===========================================

Classes d'exceptions personnalisées pour gérer les erreurs spécifiques
au domaine de l'application éducative.

Utilisation:
    from app.utils.exceptions import ChatbotException, ClaudeAPIError
    
    try:
        # Code métier
        pass
    except ClaudeAPIError as e:
        logger.error(f"Erreur Claude: {e.message}")
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# 🏷️ ÉNUMÉRATIONS D'ERREURS
# ═══════════════════════════════════════════════════════════════

class ErrorCode(str, Enum):
    """Codes d'erreur standardisés pour le chatbot."""
    
    # Erreurs générales
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # Erreurs API externes
    CLAUDE_API_ERROR = "CLAUDE_API_ERROR"
    CLAUDE_TIMEOUT = "CLAUDE_TIMEOUT"
    CLAUDE_QUOTA_EXCEEDED = "CLAUDE_QUOTA_EXCEEDED"
    CLAUDE_INVALID_MODEL = "CLAUDE_INVALID_MODEL"
    
    # Erreurs Audio
    ELEVENLABS_ERROR = "ELEVENLABS_ERROR"
    ELEVENLABS_QUOTA_EXCEEDED = "ELEVENLABS_QUOTA_EXCEEDED"
    STT_ERROR = "STT_ERROR"
    TTS_ERROR = "TTS_ERROR"
    AUDIO_FORMAT_ERROR = "AUDIO_FORMAT_ERROR"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    
    # Erreurs RAG et Documents
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    VECTORSTORE_ERROR = "VECTORSTORE_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    INGESTION_ERROR = "INGESTION_ERROR"
    SEARCH_ERROR = "SEARCH_ERROR"
    
    # Erreurs de contenu
    CONTENT_INAPPROPRIATE = "CONTENT_INAPPROPRIATE"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"
    LANGUAGE_NOT_SUPPORTED = "LANGUAGE_NOT_SUPPORTED"
    
    # Erreurs utilisateur
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    UNAUTHORIZED = "UNAUTHORIZED"


class ErrorSeverity(str, Enum):
    """Niveaux de gravité des erreurs."""
    
    LOW = "low"          # Erreur récupérable, n'affecte pas l'utilisateur
    MEDIUM = "medium"    # Erreur qui affecte une fonctionnalité
    HIGH = "high"        # Erreur qui empêche l'utilisation normale
    CRITICAL = "critical" # Erreur qui rend l'application inutilisable


# ═══════════════════════════════════════════════════════════════
# 🏗️ EXCEPTION DE BASE
# ═══════════════════════════════════════════════════════════════

class ChatbotException(Exception):
    """
    Exception de base pour toutes les erreurs métier du chatbot.
    
    Attributes:
        message: Message d'erreur lisible par l'utilisateur
        code: Code d'erreur standardisé
        details: Détails techniques supplémentaires
        severity: Niveau de gravité
        status_code: Code de statut HTTP
        timestamp: Moment de l'erreur
        context: Contexte supplémentaire
    """
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details
        self.severity = severity
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
        self.context = context or {}
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'exception en dictionnaire."""
        return {
            "message": self.message,
            "code": self.code.value,
            "details": self.details,
            "severity": self.severity.value,
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"code={self.code.value}, "
            f"severity={self.severity.value})"
        )


# ═══════════════════════════════════════════════════════════════
# 🤖 EXCEPTIONS CLAUDE API
# ═══════════════════════════════════════════════════════════════

class ClaudeAPIError(ChatbotException):
    """Erreur lors des appels à l'API Claude."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CLAUDE_API_ERROR,
        details: Optional[str] = None,
        api_response: Optional[Dict[str, Any]] = None,
        status_code: int = 502
    ):
        context = {"api_response": api_response} if api_response else {}
        
        super().__init__(
            message=message,
            code=code,
            details=details,
            severity=ErrorSeverity.HIGH,
            status_code=status_code,
            context=context
        )


class ClaudeTimeoutError(ClaudeAPIError):
    """Timeout lors d'un appel Claude."""
    
    def __init__(self, timeout_duration: float):
        super().__init__(
            message=f"L'assistant IA n'a pas répondu dans les {timeout_duration}s",
            code=ErrorCode.CLAUDE_TIMEOUT,
            details=f"Timeout après {timeout_duration} secondes",
            status_code=504
        )


class ClaudeQuotaExceededError(ClaudeAPIError):
    """Quota Claude dépassé."""
    
    def __init__(self, quota_type: str = "tokens"):
        super().__init__(
            message="Limite de l'assistant IA atteinte. Réessayez plus tard.",
            code=ErrorCode.CLAUDE_QUOTA_EXCEEDED,
            details=f"Quota {quota_type} dépassé",
            status_code=429
        )


class ClaudeInvalidModelError(ClaudeAPIError):
    """Modèle Claude invalide."""
    
    def __init__(self, model_name: str):
        super().__init__(
            message="Configuration de l'assistant IA invalide",
            code=ErrorCode.CLAUDE_INVALID_MODEL,
            details=f"Modèle '{model_name}' non supporté",
            status_code=400
        )


# ═══════════════════════════════════════════════════════════════
# 🔊 EXCEPTIONS AUDIO
# ═══════════════════════════════════════════════════════════════

class AudioError(ChatbotException):
    """Erreur générale liée à l'audio."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[str] = None,
        status_code: int = 422
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            severity=ErrorSeverity.MEDIUM,
            status_code=status_code
        )


class ElevenLabsError(AudioError):
    """Erreur ElevenLabs (TTS)."""
    
    def __init__(
        self,
        message: str = "Erreur lors de la synthèse vocale",
        details: Optional[str] = None,
        api_response: Optional[Dict[str, Any]] = None
    ):
        context = {"api_response": api_response} if api_response else {}
        
        super().__init__(
            message=message,
            code=ErrorCode.ELEVENLABS_ERROR,
            details=details
        )
        self.context = context


class ElevenLabsQuotaError(ElevenLabsError):
    """Quota ElevenLabs dépassé."""
    
    def __init__(self):
        super().__init__(
            message="Limite de synthèse vocale atteinte. Réessayez plus tard.",
            details="Quota ElevenLabs dépassé"
        )
        self.code = ErrorCode.ELEVENLABS_QUOTA_EXCEEDED
        self.status_code = 429


class STTError(AudioError):
    """Erreur Speech-to-Text."""
    
    def __init__(
        self,
        message: str = "Erreur lors de la transcription audio",
        details: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.STT_ERROR,
            details=details
        )


class TTSError(AudioError):
    """Erreur Text-to-Speech."""
    
    def __init__(
        self,
        message: str = "Erreur lors de la synthèse vocale",
        details: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.TTS_ERROR,
            details=details
        )


class AudioFormatError(AudioError):
    """Format audio non supporté."""
    
    def __init__(self, format_received: str, formats_allowed: list):
        super().__init__(
            message=f"Format audio '{format_received}' non supporté",
            code=ErrorCode.AUDIO_FORMAT_ERROR,
            details=f"Formats autorisés: {', '.join(formats_allowed)}",
            status_code=400
        )


class AudioTooLargeError(AudioError):
    """Fichier audio trop volumineux."""
    
    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            message=f"Fichier audio trop volumineux ({size_mb:.1f}MB)",
            code=ErrorCode.AUDIO_TOO_LARGE,
            details=f"Taille maximum autorisée: {max_size_mb}MB",
            status_code=413
        )


class AudioTooShortError(AudioError):
    """Fichier audio trop court."""
    
    def __init__(self, duration_s: float, min_duration_s: float = 0.1):
        super().__init__(
            message=f"Enregistrement trop court ({duration_s:.1f}s)",
            code=ErrorCode.AUDIO_TOO_SHORT,
            details=f"Durée minimum: {min_duration_s}s",
            status_code=400
        )


# ═══════════════════════════════════════════════════════════════
# 📚 EXCEPTIONS RAG ET DOCUMENTS
# ═══════════════════════════════════════════════════════════════

class RAGError(ChatbotException):
    """Erreur générale du système RAG."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[str] = None,
        status_code: int = 500
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            severity=ErrorSeverity.HIGH,
            status_code=status_code
        )


class DocumentNotFoundError(RAGError):
    """Document introuvable."""
    
    def __init__(self, document_path: str):
        super().__init__(
            message="Document éducatif introuvable",
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            details=f"Chemin: {document_path}",
            status_code=404
        )


class VectorstoreError(RAGError):
    """Erreur du vectorstore FAISS."""
    
    def __init__(
        self,
        message: str = "Erreur de la base de connaissances",
        details: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.VECTORSTORE_ERROR,
            details=details
        )


class EmbeddingError(RAGError):
    """Erreur lors de la vectorisation."""
    
    def __init__(
        self,
        message: str = "Erreur lors de l'analyse du contenu",
        details: Optional[str] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.EMBEDDING_ERROR,
            details=details
        )


class IngestionError(RAGError):
    """Erreur lors de l'ingestion de documents."""
    
    def __init__(
        self,
        document_path: str,
        details: Optional[str] = None
    ):
        super().__init__(
            message=f"Impossible d'ingérer le document {document_path}",
            code=ErrorCode.INGESTION_ERROR,
            details=details
        )


class SearchError(RAGError):
    """Erreur lors de la recherche dans les documents."""
    
    def __init__(
        self,
        query: str,
        details: Optional[str] = None
    ):
        super().__init__(
            message="Erreur lors de la recherche de contenu éducatif",
            code=ErrorCode.SEARCH_ERROR,
            details=details,
            context={"query": query}
        )


# ═══════════════════════════════════════════════════════════════
# 📝 EXCEPTIONS DE CONTENU
# ═══════════════════════════════════════════════════════════════

class ContentError(ChatbotException):
    """Erreur liée au contenu."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[str] = None,
        status_code: int = 400
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            severity=ErrorSeverity.MEDIUM,
            status_code=status_code
        )


class ContentInappropriateError(ContentError):
    """Contenu inapproprié détecté."""
    
    def __init__(self, content_type: str = "message"):
        super().__init__(
            message="Contenu inapproprié pour un contexte éducatif",
            code=ErrorCode.CONTENT_INAPPROPRIATE,
            details=f"Type de contenu: {content_type}",
            status_code=400
        )


class ContentTooLongError(ContentError):
    """Contenu trop long."""
    
    def __init__(self, length: int, max_length: int, content_type: str = "message"):
        super().__init__(
            message=f"{content_type.capitalize()} trop long ({length} caractères)",
            code=ErrorCode.CONTENT_TOO_LONG,
            details=f"Longueur maximum: {max_length} caractères",
            status_code=413
        )


class LanguageNotSupportedError(ContentError):
    """Langue non supportée."""
    
    def __init__(self, language: str, supported_languages: list):
        super().__init__(
            message=f"Langue '{language}' non supportée",
            code=ErrorCode.LANGUAGE_NOT_SUPPORTED,
            details=f"Langues supportées: {', '.join(supported_languages)}",
            status_code=400
        )


# ═══════════════════════════════════════════════════════════════
# 👤 EXCEPTIONS UTILISATEUR
# ═══════════════════════════════════════════════════════════════

class UserError(ChatbotException):
    """Erreur liée à l'utilisateur."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[str] = None,
        status_code: int = 400
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            severity=ErrorSeverity.LOW,
            status_code=status_code
        )


class RateLimitExceededError(UserError):
    """Limite de débit dépassée."""
    
    def __init__(
        self,
        limit: int,
        window_minutes: int = 60,
        resource: str = "requêtes"
    ):
        super().__init__(
            message=f"Trop de {resource}. Réessayez dans {window_minutes} minutes.",
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details=f"Limite: {limit} {resource} par {window_minutes} minutes",
            status_code=429
        )


class SessionExpiredError(UserError):
    """Session expirée."""
    
    def __init__(self):
        super().__init__(
            message="Session expirée. Veuillez vous reconnecter.",
            code=ErrorCode.SESSION_EXPIRED,
            status_code=401
        )


class UnauthorizedError(UserError):
    """Accès non autorisé."""
    
    def __init__(self, resource: str = "ressource"):
        super().__init__(
            message=f"Accès non autorisé à cette {resource}",
            code=ErrorCode.UNAUTHORIZED,
            details=f"Ressource: {resource}",
            status_code=403
        )


# ═══════════════════════════════════════════════════════════════
# 🔧 EXCEPTIONS DE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ConfigurationError(ChatbotException):
    """Erreur de configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[str] = None
    ):
        context = {"config_key": config_key} if config_key else {}
        
        super().__init__(
            message=message,
            code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            context=context
        )


class ValidationError(ChatbotException):
    """Erreur de validation des données."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        details: Optional[str] = None
    ):
        context = {}
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = str(field_value)
        
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            details=details,
            severity=ErrorSeverity.LOW,
            status_code=422,
            context=context
        )


# ═══════════════════════════════════════════════════════════════
# 🛠️ UTILITAIRES D'EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

def handle_external_api_error(
    api_name: str,
    status_code: int,
    response_data: Optional[Dict[str, Any]] = None,
    timeout: bool = False
) -> ChatbotException:
    """
    Convertit une erreur d'API externe en exception métier.
    
    Args:
        api_name: Nom de l'API (claude, elevenlabs, etc.)
        status_code: Code de statut HTTP
        response_data: Données de réponse de l'API
        timeout: True si c'est un timeout
        
    Returns:
        ChatbotException: Exception appropriée
    """
    
    if timeout:
        if api_name.lower() == "claude":
            return ClaudeTimeoutError(30.0)
        else:
            return ChatbotException(
                message=f"Timeout de l'API {api_name}",
                code=ErrorCode.UNKNOWN_ERROR,
                status_code=504
            )
    
    # Gestion par API
    if api_name.lower() == "claude":
        if status_code == 429:
            return ClaudeQuotaExceededError()
        elif status_code == 400:
            return ClaudeInvalidModelError("modèle_inconnu")
        else:
            return ClaudeAPIError(
                message="Erreur de l'assistant IA",
                api_response=response_data,
                status_code=status_code
            )
    
    elif api_name.lower() == "elevenlabs":
        if status_code == 429:
            return ElevenLabsQuotaError()
        else:
            return ElevenLabsError(
                message="Erreur de synthèse vocale",
                api_response=response_data
            )
    
    # Erreur générique
    return ChatbotException(
        message=f"Erreur API {api_name}",
        details=f"Status: {status_code}",
        context={"api_name": api_name, "response": response_data},
        status_code=status_code if 400 <= status_code < 600 else 500
    )


def wrap_with_context(
    func,
    context: Dict[str, Any],
    exception_class: type = ChatbotException
):
    """
    Décorateur pour enrichir les exceptions avec du contexte.
    
    Args:
        func: Fonction à décorer
        context: Contexte à ajouter aux exceptions
        exception_class: Classe d'exception à utiliser
        
    Example:
        @wrap_with_context({"service": "claude"}, ClaudeAPIError)
        def call_claude_api():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ChatbotException as e:
                # Enrichir le contexte existant
                e.context.update(context)
                raise
            except Exception as e:
                # Convertir en exception métier
                raise exception_class(
                    message=str(e),
                    details=f"Erreur dans {func.__name__}: {e}",
                    context=context
                )
        return wrapper
    return decorator


def create_error_response(exception: ChatbotException) -> Dict[str, Any]:
    """
    Crée une réponse d'erreur standardisée pour l'API.
    
    Args:
        exception: Exception métier
        
    Returns:
        Dict: Réponse formatée pour l'API
    """
    return {
        "success": False,
        "error": {
            "message": exception.message,
            "code": exception.code.value,
            "severity": exception.severity.value,
            "timestamp": exception.timestamp.isoformat(),
            "details": exception.details,
            "context": exception.context
        }
    }


# ═══════════════════════════════════════════════════════════════
# 🧪 FONCTIONS DE TEST
# ═══════════════════════════════════════════════════════════════

def test_exceptions():
    """Teste toutes les classes d'exceptions."""
    
    print("🧪 Test des exceptions métier...")
    
    # Test exception de base
    try:
        raise ChatbotException(
            "Test exception de base",
            code=ErrorCode.UNKNOWN_ERROR,
            details="Détails de test",
            context={"test": True}
        )
    except ChatbotException as e:
        print(f"✅ Exception de base: {e}")
        print(f"   Dict: {e.to_dict()}")
    
    # Test Claude API
    try:
        raise ClaudeTimeoutError(30.0)
    except ClaudeTimeoutError as e:
        print(f"✅ Claude timeout: {e}")
    
    # Test Audio
    try:
        raise AudioFormatError("mp4", ["wav", "mp3"])
    except AudioFormatError as e:
        print(f"✅ Audio format: {e}")
    
    # Test RAG
    try:
        raise DocumentNotFoundError("/data/documents/missing.md")
    except DocumentNotFoundError as e:
        print(f"✅ Document not found: {e}")
    
    # Test conversion API externe
    api_error = handle_external_api_error("claude", 429)
    print(f"✅ API externe: {api_error}")
    
    # Test réponse d'erreur
    error_response = create_error_response(api_error)
    print(f"✅ Réponse d'erreur: {error_response}")
    
    print("\n✅ Test des exceptions terminé!")


if __name__ == "__main__":
    test_exceptions()