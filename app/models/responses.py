"""
Modèles Pydantic pour les Réponses API
======================================

Définit les structures de données pour toutes les réponses sortantes
du chatbot éducatif vocal.

Utilisation:
    from app.models.responses import ChatResponse, TTSResponse
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        return ChatResponse(...)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# 🏷️ ÉNUMÉRATIONS POUR LES RÉPONSES
# ═══════════════════════════════════════════════════════════════

class ResponseStatus(str, Enum):
    """Statuts possibles des réponses."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class ServiceStatus(str, Enum):
    """Statuts des services."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════
# 📊 MODÈLE DE BASE POUR LES RÉPONSES
# ═══════════════════════════════════════════════════════════════

class BaseResponse(BaseModel):
    """Modèle de base pour toutes les réponses API."""
    
    success: bool = Field(
        description="Indique si la requête a réussi"
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Message explicatif"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Horodatage de la réponse"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Identifiant unique de la requête"
    )


# ═══════════════════════════════════════════════════════════════
# 💬 RÉPONSES DE CONVERSATION
# ═══════════════════════════════════════════════════════════════

class DocumentContext(BaseModel):
    """Contexte d'un document utilisé pour la réponse."""
    
    content: str = Field(
        description="Contenu du document"
    )
    
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de pertinence (0-1)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Métadonnées du document"
    )
    
    source: Optional[str] = Field(
        default=None,
        description="Source/nom du document"
    )


class ChatMetadata(BaseModel):
    """Métadonnées de la réponse de chat."""
    
    model_used: str = Field(
        description="Modèle OpenAI utilisé"
    )
    
    tokens_used: int = Field(
        ge=0,
        description="Nombre total de tokens utilisés"
    )
    
    prompt_tokens: int = Field(
        ge=0,
        description="Tokens utilisés pour le prompt"
    )
    
    completion_tokens: int = Field(
        ge=0,
        description="Tokens utilisés pour la réponse"
    )
    
    response_time_ms: float = Field(
        ge=0,
        description="Temps de réponse en millisecondes"
    )
    
    context_documents_count: int = Field(
        ge=0,
        description="Nombre de documents utilisés comme contexte"
    )
    
    education_level_detected: Optional[str] = Field(
        default=None,
        description="Niveau éducatif détecté automatiquement"
    )
    
    subject_detected: Optional[str] = Field(
        default=None,
        description="Matière détectée automatiquement"
    )


class ChatResponse(BaseResponse):
    """Réponse à une requête de chat."""
    
    response: str = Field(
        description="Réponse générée par l'IA"
    )
    
    context_documents: Optional[List[DocumentContext]] = Field(
        default=None,
        description="Documents utilisés pour générer la réponse"
    )
    
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions de questions de suivi"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session"
    )
    
    metadata: ChatMetadata = Field(
        description="Métadonnées techniques de la réponse"
    )
    
    audio_available: bool = Field(
        default=False,
        description="Indique si une version audio est disponible"
    )
    
    audio_url: Optional[str] = Field(
        default=None,
        description="URL de la version audio (si disponible)"
    )


# ═══════════════════════════════════════════════════════════════
# 🔊 RÉPONSES AUDIO
# ═══════════════════════════════════════════════════════════════

class AudioMetadata(BaseModel):
    """Métadonnées audio."""
    
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Durée en secondes"
    )
    
    format: str = Field(
        description="Format audio (mp3, wav, etc.)"
    )
    
    size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Taille du fichier en bytes"
    )
    
    sample_rate: Optional[int] = Field(
        default=None,
        description="Taux d'échantillonnage"
    )
    
    bitrate: Optional[int] = Field(
        default=None,
        description="Débit binaire"
    )


class TTSResponse(BaseResponse):
    """Réponse à une requête de synthèse vocale."""
    
    audio_data: Optional[bytes] = Field(
        default=None,
        description="Données audio encodées en base64"
    )
    
    audio_url: Optional[str] = Field(
        default=None,
        description="URL temporaire du fichier audio"
    )
    
    text_processed: str = Field(
        description="Texte qui a été traité"
    )
    
    voice_id: str = Field(
        description="ID de la voix utilisée"
    )
    
    metadata: AudioMetadata = Field(
        description="Métadonnées de l'audio généré"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Temps de traitement en millisecondes"
    )


class STTResponse(BaseResponse):
    """Réponse à une requête de transcription audio."""
    
    transcription: str = Field(
        description="Texte transcrit"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Niveau de confiance de la transcription (0-1)"
    )
    
    language_detected: Optional[str] = Field(
        default=None,
        description="Langue détectée automatiquement"
    )
    
    segments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Segments de transcription avec timestamps"
    )
    
    audio_metadata: AudioMetadata = Field(
        description="Métadonnées du fichier audio d'entrée"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Temps de traitement en millisecondes"
    )
    
    model_used: str = Field(
        description="Modèle Whisper utilisé"
    )


# ═══════════════════════════════════════════════════════════════
# 📚 RÉPONSES RAG ET DOCUMENTS
# ═══════════════════════════════════════════════════════════════

class DocumentInfo(BaseModel):
    """Informations sur un document ingéré."""
    
    filename: str = Field(
        description="Nom du fichier"
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Titre du document"
    )
    
    chunks_count: int = Field(
        ge=0,
        description="Nombre de chunks créés"
    )
    
    subject: Optional[str] = Field(
        default=None,
        description="Matière du document"
    )
    
    education_level: Optional[str] = Field(
        default=None,
        description="Niveau éducatif"
    )
    
    word_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre de mots"
    )
    
    language: Optional[str] = Field(
        default=None,
        description="Langue détectée"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Temps de traitement"
    )
    
    status: str = Field(
        description="Statut de l'ingestion (success, error, warning)"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Message d'erreur si échec"
    )


class IngestResponse(BaseResponse):
    """Réponse à une requête d'ingestion de documents."""
    
    documents_processed: List[DocumentInfo] = Field(
        description="Liste des documents traités"
    )
    
    total_documents: int = Field(
        ge=0,
        description="Nombre total de documents traités"
    )
    
    successful_ingestions: int = Field(
        ge=0,
        description="Nombre d'ingestions réussies"
    )
    
    failed_ingestions: int = Field(
        ge=0,
        description="Nombre d'ingestions échouées"
    )
    
    total_chunks_created: int = Field(
        ge=0,
        description="Nombre total de chunks créés"
    )
    
    vectorstore_size: int = Field(
        ge=0,
        description="Taille du vectorstore après ingestion"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Temps total de traitement"
    )
    
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Avertissements éventuels"
    )


class SearchResult(BaseModel):
    """Résultat de recherche dans les documents."""
    
    content: str = Field(
        description="Contenu du chunk trouvé"
    )
    
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score de similarité (0-1)"
    )
    
    metadata: Dict[str, Any] = Field(
        description="Métadonnées du document source"
    )
    
    document_title: Optional[str] = Field(
        default=None,
        description="Titre du document source"
    )
    
    document_path: Optional[str] = Field(
        default=None,
        description="Chemin du document source"
    )
    
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index du chunk dans le document"
    )
    
    context_preview: Optional[str] = Field(
        default=None,
        description="Aperçu du contexte environnant"
    )


class SearchResponse(BaseResponse):
    """Réponse à une requête de recherche."""
    
    results: List[SearchResult] = Field(
        description="Résultats de recherche"
    )
    
    query: str = Field(
        description="Requête de recherche originale"
    )
    
    total_results: int = Field(
        ge=0,
        description="Nombre total de résultats trouvés"
    )
    
    results_returned: int = Field(
        ge=0,
        description="Nombre de résultats retournés"
    )
    
    search_time_ms: float = Field(
        ge=0,
        description="Temps de recherche en millisecondes"
    )
    
    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres appliqués à la recherche"
    )


# ═══════════════════════════════════════════════════════════════
# 🏥 RÉPONSES DE SANTÉ ET MONITORING
# ═══════════════════════════════════════════════════════════════

class ServiceHealth(BaseModel):
    """État de santé d'un service."""
    
    name: str = Field(
        description="Nom du service"
    )
    
    status: ServiceStatus = Field(
        description="Statut du service"
    )
    
    response_time_ms: Optional[float] = Field(
        default=None,
        description="Temps de réponse en millisecondes"
    )
    
    last_check: datetime = Field(
        description="Dernière vérification"
    )
    
    version: Optional[str] = Field(
        default=None,
        description="Version du service"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Informations supplémentaires"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Message d'erreur si unhealthy"
    )


class HealthResponse(BaseResponse):
    """Réponse complète de santé du système."""
    
    overall_status: ServiceStatus = Field(
        description="Statut global du système"
    )
    
    services: List[ServiceHealth] = Field(
        description="État détaillé de chaque service"
    )
    
    system_info: Dict[str, Any] = Field(
        description="Informations système"
    )
    
    uptime_seconds: float = Field(
        ge=0,
        description="Temps de fonctionnement en secondes"
    )
    
    version: str = Field(
        description="Version de l'API"
    )
    
    environment: str = Field(
        description="Environnement (dev, staging, prod)"
    )


# ═══════════════════════════════════════════════════════════════
# 📊 RÉPONSES DE MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

class ServiceMetrics(BaseModel):
    """Métriques d'un service spécifique."""
    
    service_name: str = Field(
        description="Nom du service"
    )
    
    total_requests: int = Field(
        ge=0,
        description="Nombre total de requêtes"
    )
    
    successful_requests: int = Field(
        ge=0,
        description="Nombre de requêtes réussies"
    )
    
    failed_requests: int = Field(
        ge=0,
        description="Nombre de requêtes échouées"
    )
    
    average_response_time_ms: float = Field(
        ge=0,
        description="Temps de réponse moyen"
    )
    
    error_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Taux d'erreur en pourcentage"
    )
    
    last_request_time: Optional[datetime] = Field(
        default=None,
        description="Dernière requête"
    )


class MetricsResponse(BaseResponse):
    """Réponse avec les métriques système."""
    
    time_period: str = Field(
        description="Période couverte par les métriques"
    )
    
    global_metrics: Dict[str, Any] = Field(
        description="Métriques globales"
    )
    
    service_metrics: List[ServiceMetrics] = Field(
        description="Métriques par service"
    )
    
    performance_stats: Dict[str, float] = Field(
        description="Statistiques de performance"
    )
    
    resource_usage: Dict[str, Any] = Field(
        description="Utilisation des ressources"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Moment de génération des métriques"
    )


# ═══════════════════════════════════════════════════════════════
# ⚙️ RÉPONSES DE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ConfigInfo(BaseModel):
    """Informations de configuration actuelles."""
    
    openai_model: str = Field(
        description="Modèle OpenAI configuré"
    )
    
    max_tokens: int = Field(
        description="Nombre maximum de tokens"
    )
    
    temperature: float = Field(
        description="Température configurée"
    )
    
    vectorstore_enabled: bool = Field(
        description="RAG activé"
    )
    
    tts_enabled: bool = Field(
        description="Synthèse vocale activée"
    )
    
    stt_enabled: bool = Field(
        description="Transcription activée"
    )
    
    debug_mode: bool = Field(
        description="Mode debug activé"
    )
    
    environment: str = Field(
        description="Environnement actuel"
    )


class ConfigResponse(BaseResponse):
    """Réponse avec la configuration actuelle."""
    
    config: ConfigInfo = Field(
        description="Configuration actuelle"
    )
    
    modifiable_settings: List[str] = Field(
        description="Paramètres modifiables à chaud"
    )
    
    restart_required_settings: List[str] = Field(
        description="Paramètres nécessitant un redémarrage"
    )


# ═══════════════════════════════════════════════════════════════
# 🎪 RÉPONSES DE SESSION
# ═══════════════════════════════════════════════════════════════

class SessionInfo(BaseModel):
    """Informations de session utilisateur."""
    
    session_id: str = Field(
        description="Identifiant de session"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="Identifiant utilisateur"
    )
    
    created_at: datetime = Field(
        description="Date de création"
    )
    
    expires_at: datetime = Field(
        description="Date d'expiration"
    )
    
    conversation_count: int = Field(
        ge=0,
        description="Nombre d'échanges dans cette session"
    )
    
    preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Préférences utilisateur"
    )
    
    last_activity: datetime = Field(
        description="Dernière activité"
    )


class SessionResponse(BaseResponse):
    """Réponse de gestion de session."""
    
    session: SessionInfo = Field(
        description="Informations de session"
    )
    
    token: Optional[str] = Field(
        default=None,
        description="Token de session (si nouveau)"
    )


# ═══════════════════════════════════════════════════════════════
# 🔄 RÉPONSES DE PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════════

class PipelineStep(BaseModel):
    """Étape du pipeline de traitement."""
    
    step_name: str = Field(
        description="Nom de l'étape"
    )
    
    status: str = Field(
        description="Statut de l'étape (pending, running, completed, failed)"
    )
    
    start_time: Optional[datetime] = Field(
        default=None,
        description="Début de l'étape"
    )
    
    end_time: Optional[datetime] = Field(
        default=None,
        description="Fin de l'étape"
    )
    
    duration_ms: Optional[float] = Field(
        default=None,
        description="Durée en millisecondes"
    )
    
    output: Optional[Any] = Field(
        default=None,
        description="Sortie de l'étape"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Erreur éventuelle"
    )


class VoicePipelineResponse(BaseResponse):
    """Réponse du pipeline vocal complet (STT -> Chat -> TTS)."""
    
    final_response: str = Field(
        description="Réponse textuelle finale"
    )
    
    audio_response_url: Optional[str] = Field(
        default=None,
        description="URL de la réponse audio"
    )
    
    transcription: str = Field(
        description="Transcription de l'audio d'entrée"
    )
    
    pipeline_steps: List[PipelineStep] = Field(
        description="Détail des étapes du pipeline"
    )
    
    total_processing_time_ms: float = Field(
        ge=0,
        description="Temps total de traitement"
    )
    
    tokens_used: int = Field(
        ge=0,
        description="Tokens utilisés pour la génération"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="ID de session"
    )


# ═══════════════════════════════════════════════════════════════
# ❌ RÉPONSES D'ERREUR
# ═══════════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    """Détail d'une erreur."""
    
    code: str = Field(
        description="Code d'erreur"
    )
    
    message: str = Field(
        description="Message d'erreur lisible"
    )
    
    field: Optional[str] = Field(
        default=None,
        description="Champ concerné par l'erreur"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexte supplémentaire"
    )


class ErrorResponse(BaseResponse):
    """Réponse d'erreur standardisée."""
    
    error: ErrorDetail = Field(
        description="Détails de l'erreur"
    )
    
    suggestion: Optional[str] = Field(
        default=None,
        description="Suggestion pour résoudre l'erreur"
    )
    
    documentation_url: Optional[str] = Field(
        default=None,
        description="URL de documentation pertinente"
    )
    
    support_contact: Optional[str] = Field(
        default=None,
        description="Contact support"
    )


# ═══════════════════════════════════════════════════════════════
# 🧪 RÉPONSES DE TEST
# ═══════════════════════════════════════════════════════════════

class TestResult(BaseModel):
    """Résultat d'un test."""
    
    test_name: str = Field(
        description="Nom du test"
    )
    
    passed: bool = Field(
        description="Test réussi ou non"
    )
    
    duration_ms: float = Field(
        ge=0,
        description="Durée du test"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Détails du test"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Message d'erreur si échec"
    )
    
    expected_result: Optional[Any] = Field(
        default=None,
        description="Résultat attendu"
    )
    
    actual_result: Optional[Any] = Field(
        default=None,
        description="Résultat obtenu"
    )


class TestResponse(BaseResponse):
    """Réponse d'exécution de tests."""
    
    test_suite: str = Field(
        description="Suite de tests exécutée"
    )
    
    tests: List[TestResult] = Field(
        description="Résultats des tests individuels"
    )
    
    total_tests: int = Field(
        ge=0,
        description="Nombre total de tests"
    )
    
    passed_tests: int = Field(
        ge=0,
        description="Nombre de tests réussis"
    )
    
    failed_tests: int = Field(
        ge=0,
        description="Nombre de tests échoués"
    )
    
    success_rate_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Taux de réussite"
    )
    
    total_duration_ms: float = Field(
        ge=0,
        description="Durée totale des tests"
    )
    
    environment_info: Dict[str, Any] = Field(
        description="Informations sur l'environnement de test"
    )


# ═══════════════════════════════════════════════════════════════
# 📁 RÉPONSES DE GESTION DES FICHIERS
# ═══════════════════════════════════════════════════════════════

class FileInfo(BaseModel):
    """Informations sur un fichier."""
    
    filename: str = Field(
        description="Nom du fichier"
    )
    
    path: str = Field(
        description="Chemin du fichier"
    )
    
    size_bytes: int = Field(
        ge=0,
        description="Taille en bytes"
    )
    
    created_at: datetime = Field(
        description="Date de création"
    )
    
    modified_at: datetime = Field(
        description="Dernière modification"
    )
    
    file_type: str = Field(
        description="Type de fichier"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Métadonnées du fichier"
    )


class FileListResponse(BaseResponse):
    """Réponse avec liste de fichiers."""
    
    files: List[FileInfo] = Field(
        description="Liste des fichiers"
    )
    
    total_files: int = Field(
        ge=0,
        description="Nombre total de fichiers"
    )
    
    total_size_bytes: int = Field(
        ge=0,
        description="Taille totale"
    )
    
    directory: str = Field(
        description="Répertoire listé"
    )


# ═══════════════════════════════════════════════════════════════
# 🎯 RÉPONSES RAPIDES (HELPERS)
# ═══════════════════════════════════════════════════════════════

def create_success_response(
    message: str = "Opération réussie",
    data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> BaseResponse:
    """Crée une réponse de succès standardisée."""
    response = BaseResponse(
        success=True,
        message=message,
        request_id=request_id
    )
    
    if data:
        for key, value in data.items():
            setattr(response, key, value)
    
    return response


def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    details: Optional[str] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Crée une réponse d'erreur standardisée."""
    return ErrorResponse(
        success=False,
        message="Une erreur s'est produite",
        request_id=request_id,
        error=ErrorDetail(
            code=error_code,
            message=message,
            context={"details": details} if details else None
        )
    )


# ═══════════════════════════════════════════════════════════════
# 📋 TYPES UNION POUR FLEXIBILITÉ
# ═══════════════════════════════════════════════════════════════

# Union des réponses principales pour typage flexible
ApiResponse = Union[
    ChatResponse,
    TTSResponse, 
    STTResponse,
    IngestResponse,
    SearchResponse,
    HealthResponse,
    MetricsResponse,
    SessionResponse,
    VoicePipelineResponse,
    TestResponse,
    ErrorResponse
]

# Union des réponses avec données
DataResponse = Union[
    ChatResponse,
    SearchResponse,
    IngestResponse,
    MetricsResponse
]

# Union des réponses système  
SystemResponse = Union[
    HealthResponse,
    ConfigResponse,
    TestResponse
]