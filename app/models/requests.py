"""
Modèles Pydantic pour les Requêtes API
======================================

Définit toutes les structures de données pour les requêtes entrantes
du chatbot éducatif vocal.

Utilisation:
    from app.models.requests import ChatRequest, TTSRequest
    
    @app.post("/chat")
    async def chat(request: ChatRequest):
        pass
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator


# ═══════════════════════════════════════════════════════════════
# 🎯 ÉNUMÉRATIONS
# ═══════════════════════════════════════════════════════════════

class LanguageCode(str, Enum):
    """Codes de langue supportés."""
    FRENCH = "fr"
    ENGLISH = "en"
    SPANISH = "es"
    ARABIC = "ar"


class DifficultyLevel(str, Enum):
    """Niveaux de difficulté éducative."""
    BEGINNER = "beginner"        # Débutant
    ELEMENTARY = "elementary"    # Élémentaire  
    INTERMEDIATE = "intermediate" # Intermédiaire
    ADVANCED = "advanced"        # Avancé


class AudioFormat(str, Enum):
    """Formats audio supportés."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    M4A = "m4a"
    FLAC = "flac"


class ResponseStyle(str, Enum):
    """Styles de réponse."""
    CONVERSATIONAL = "conversational"  # Conversationnel
    EDUCATIONAL = "educational"        # Éducatif structuré
    STORYTELLING = "storytelling"      # Narratif
    PRACTICAL = "practical"            # Pratique/exercices


# ═══════════════════════════════════════════════════════════════
# 💬 MODÈLES CHAT ET CONVERSATION
# ═══════════════════════════════════════════════════════════════

class ConversationMessage(BaseModel):
    """Un message dans l'historique de conversation."""
    
    role: str = Field(
        ...,
        description="Rôle du message: 'user' ou 'assistant'"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Contenu du message"
    )
    
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Moment du message"
    )
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError('Le rôle doit être "user" ou "assistant"')
        return v


class ChatRequest(BaseModel):
    """Requête pour générer une réponse de chat."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Message de l'utilisateur"
    )
    
    language: LanguageCode = Field(
        default=LanguageCode.FRENCH,
        description="Langue de la conversation"
    )
    
    difficulty_level: DifficultyLevel = Field(
        default=DifficultyLevel.BEGINNER,
        description="Niveau de difficulté souhaité"
    )
    
    response_style: ResponseStyle = Field(
        default=ResponseStyle.CONVERSATIONAL,
        description="Style de réponse souhaité"
    )
    
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        max_items=20,
        description="Historique de la conversation (max 20 messages)"
    )
    
    use_rag: bool = Field(
        default=True,
        description="Utiliser la recherche dans les documents éducatifs"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=50,
        le=2000,
        description="Nombre maximum de tokens pour la réponse"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Température de génération (créativité)"
    )
    
    context_hint: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Indice de contexte pour améliorer la recherche"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="Identifiant utilisateur (optionnel)"
    )
    
    @validator('message')
    def clean_message(cls, v):
        """Nettoie le message utilisateur."""
        return v.strip()
    
    @validator('conversation_history')
    def validate_history(cls, v):
        """Valide l'historique de conversation."""
        if v is None:
            return v
        
        # Vérifier l'alternance user/assistant
        if len(v) > 1:
            for i in range(len(v) - 1):
                current_role = v[i].role
                next_role = v[i + 1].role
                if current_role == next_role:
                    raise ValueError('Les messages doivent alterner entre user et assistant')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Explique-moi comment faire une multiplication",
                "language": "fr",
                "difficulty_level": "beginner",
                "response_style": "conversational",
                "use_rag": True,
                "context_hint": "mathématiques de base"
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🔊 MODÈLES AUDIO
# ═══════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    """Requête pour la synthèse vocale (Text-to-Speech)."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Texte à synthétiser"
    )
    
    language: LanguageCode = Field(
        default=LanguageCode.FRENCH,
        description="Langue de synthèse"
    )
    
    voice_id: Optional[str] = Field(
        default=None,
        description="ID de voix spécifique (ElevenLabs)"
    )
    
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Vitesse de lecture (0.5x à 2x)"
    )
    
    stability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stabilité de la voix (ElevenLabs)"
    )
    
    similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarité à la voix originale (ElevenLabs)"
    )
    
    output_format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Format audio de sortie"
    )
    
    @validator('text')
    def clean_text(cls, v):
        """Nettoie le texte pour la synthèse."""
        # Supprime les caractères spéciaux problématiques
        import re
        cleaned = re.sub(r'[^\w\s\.,!?;:()\-\'\"àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]', '', v)
        return cleaned.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Bonjour ! Aujourd'hui nous allons apprendre les tables de multiplication.",
                "language": "fr",
                "speed": 1.0,
                "output_format": "mp3"
            }
        }


class STTRequest(BaseModel):
    """Requête pour la reconnaissance vocale (Speech-to-Text)."""
    
    language: LanguageCode = Field(
        default=LanguageCode.FRENCH,
        description="Langue attendue de l'audio"
    )
    
    enhance_audio: bool = Field(
        default=True,
        description="Améliorer la qualité audio avant transcription"
    )
    
    filter_profanity: bool = Field(
        default=True,
        description="Filtrer les gros mots"
    )
    
    context_hint: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Contexte pour améliorer la transcription"
    )
    
    max_duration_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Durée maximum autorisée (en secondes)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "language": "fr",
                "enhance_audio": True,
                "filter_profanity": True,
                "context_hint": "éducation mathématiques"
            }
        }


# ═══════════════════════════════════════════════════════════════
# 📚 MODÈLES RAG ET DOCUMENTS
# ═══════════════════════════════════════════════════════════════

class IngestionRequest(BaseModel):
    """Requête pour ingérer des documents éducatifs."""
    
    file_paths: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="Chemins des fichiers à ingérer"
    )
    
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=2000,
        description="Taille des chunks de texte"
    )
    
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        description="Chevauchement entre chunks"
    )
    
    subject_category: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Catégorie du sujet (math, français, etc.)"
    )
    
    difficulty_level: Optional[DifficultyLevel] = Field(
        default=None,
        description="Niveau de difficulté du contenu"
    )
    
    overwrite_existing: bool = Field(
        default=False,
        description="Remplacer l'index existant"
    )
    
    @validator('file_paths')
    def validate_paths(cls, v):
        """Valide les chemins de fichiers."""
        for path in v:
            if not path.endswith(('.md', '.txt')):
                raise ValueError(f"Format non supporté: {path}")
        return v
    
    @root_validator
    def validate_chunk_sizes(cls, values):
        """Valide la cohérence des tailles de chunks."""
        chunk_size = values.get('chunk_size')
        chunk_overlap = values.get('chunk_overlap')
        
        if chunk_size and chunk_overlap and chunk_overlap >= chunk_size:
            raise ValueError('chunk_overlap doit être inférieur à chunk_size')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "file_paths": [
                    "data/documents/math_basics.md",
                    "data/documents/fractions.md"
                ],
                "subject_category": "mathématiques",
                "difficulty_level": "beginner",
                "overwrite_existing": False
            }
        }


class SearchRequest(BaseModel):
    """Requête pour rechercher dans les documents."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Requête de recherche"
    )
    
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum de résultats"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Seuil de similarité minimum"
    )
    
    subject_filter: Optional[str] = Field(
        default=None,
        description="Filtrer par catégorie de sujet"
    )
    
    difficulty_filter: Optional[DifficultyLevel] = Field(
        default=None,
        description="Filtrer par niveau de difficulté"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "comment faire une division",
                "max_results": 3,
                "similarity_threshold": 0.8,
                "subject_filter": "mathématiques"
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🔧 MODÈLES UTILITAIRES
# ═══════════════════════════════════════════════════════════════

class HealthCheckRequest(BaseModel):
    """Requête pour vérifier la santé des services."""
    
    include_details: bool = Field(
        default=True,
        description="Inclure les détails de chaque service"
    )
    
    test_connections: bool = Field(
        default=False,
        description="Tester les connexions aux APIs externes"
    )


class ConfigUpdateRequest(BaseModel):
    """Requête pour mettre à jour la configuration."""
    
    openai_model: Optional[str] = Field(
        default=None,
        description="Nouveau modèle OpenAI"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=50,
        le=4000,
        description="Nouveau nombre maximum de tokens"
    )
    
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nouvelle température"
    )
    
    log_level: Optional[str] = Field(
        default=None,
        description="Nouveau niveau de log"
    )
    
    @validator('openai_model')
    def validate_model(cls, v):
        if v is not None:
            allowed = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']
            if v not in allowed:
                raise ValueError(f'Modèle doit être parmi: {allowed}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        if v is not None:
            allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v.upper() not in allowed:
                raise ValueError(f'Niveau de log doit être parmi: {allowed}')
            return v.upper()
        return v


# ═══════════════════════════════════════════════════════════════
# 🎪 MODÈLES CONVERSATION COMPLEXE
# ═══════════════════════════════════════════════════════════════

class VoiceConversationRequest(BaseModel):
    """Requête pour une conversation vocale complète (STT + Chat + TTS)."""
    
    # Paramètres STT
    stt_language: LanguageCode = Field(
        default=LanguageCode.FRENCH,
        description="Langue de transcription"
    )
    
    enhance_audio: bool = Field(
        default=True,
        description="Améliorer l'audio avant transcription"
    )
    
    # Paramètres Chat
    difficulty_level: DifficultyLevel = Field(
        default=DifficultyLevel.BEGINNER,
        description="Niveau éducatif"
    )
    
    response_style: ResponseStyle = Field(
        default=ResponseStyle.CONVERSATIONAL,
        description="Style de réponse"
    )
    
    use_rag: bool = Field(
        default=True,
        description="Utiliser la recherche documentaire"
    )
    
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        max_items=10,
        description="Historique récent"
    )
    
    # Paramètres TTS
    tts_language: Optional[LanguageCode] = Field(
        default=None,
        description="Langue de synthèse (défaut: même que STT)"
    )
    
    voice_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Vitesse de la voix"
    )
    
    audio_format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Format audio final"
    )
    
    # Métadonnées
    user_id: Optional[str] = Field(
        default=None,
        description="ID utilisateur pour personalisation"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="ID de session pour continuité"
    )
    
    @root_validator
    def set_tts_language_default(cls, values):
        """Met la même langue pour TTS si non spécifiée."""
        if values.get('tts_language') is None:
            values['tts_language'] = values.get('stt_language', LanguageCode.FRENCH)
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "stt_language": "fr",
                "difficulty_level": "beginner",
                "response_style": "conversational",
                "use_rag": True,
                "voice_speed": 0.9,
                "audio_format": "mp3"
            }
        }


# ═══════════════════════════════════════════════════════════════
# 📋 EXPORTS PRINCIPAUX
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "LanguageCode",
    "DifficultyLevel", 
    "AudioFormat",
    "ResponseStyle",
    
    # Models Chat
    "ConversationMessage",
    "ChatRequest",
    
    # Models Audio
    "TTSRequest",
    "STTRequest",
    
    # Models RAG
    "IngestionRequest",
    "SearchRequest",
    
    # Models Utilitaires
    "HealthCheckRequest",
    "ConfigUpdateRequest",
    
    # Models complexes
    "VoiceConversationRequest"
]