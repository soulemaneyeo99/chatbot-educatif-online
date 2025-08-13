"""
Configuration du Chatbot Éducatif Vocal
=======================================

Gestion centralisée de la configuration via Pydantic Settings.
Variables d'environnement, validation automatique, et valeurs par défaut.

Utilisation:
    from app.config import get_settings
    settings = get_settings()
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Configuration principale de l'application.
    
    Toutes les variables peuvent être surchargées via des variables d'environnement
    ou un fichier .env à la racine du projet.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # 🏗️ CONFIGURATION GÉNÉRALE
    # ═══════════════════════════════════════════════════════════════
    
    APP_NAME: str = Field(
        default="Chatbot Éducatif Vocal",
        description="Nom de l'application"
    )
    
    VERSION: str = Field(
        default="1.0.0",
        description="Version de l'API"
    )
    
    DEBUG: bool = Field(
        default=False,
        description="Mode debug - active les logs détaillés et la documentation"
    )
    
    ENVIRONMENT: str = Field(
        default="development",
        description="Environnement: development, staging, production"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 🌐 CONFIGURATION SERVEUR
    # ═══════════════════════════════════════════════════════════════
    
    HOST: str = Field(
        default="127.0.0.1",
        description="Adresse IP du serveur"
    )
    
    PORT: int = Field(
        default=8000,
        description="Port du serveur"
    )
    
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ],
        description="Origines autorisées pour CORS"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 🤖 CONFIGURATION OPENAI API
    # ═══════════════════════════════════════════════════════════════
    
    OPENAI_API_KEY: str = Field(
        ...,  # Obligatoire
        description="Clé API OpenAI"
    )
    
    OPENAI_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="Modèle OpenAI à utiliser"
    )
    
    OPENAI_MAX_TOKENS: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Nombre maximum de tokens pour les réponses OpenAI"
    )
    
    OPENAI_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Température pour la génération (créativité)"
    )
    
    OPENAI_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout en secondes pour les requêtes OpenAI"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 🔊 CONFIGURATION ELEVENLABS (TTS)
    # ═══════════════════════════════════════════════════════════════
    
    ELEVENLABS_API_KEY: Optional[str] = Field(
        default=None,
        description="Clé API ElevenLabs pour la synthèse vocale"
    )
    
    ELEVENLABS_VOICE_ID: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        description="ID de la voix ElevenLabs à utiliser"
    )
    
    ELEVENLABS_MODEL: str = Field(
        default="eleven_multilingual_v2",
        description="Modèle ElevenLabs à utiliser"
    )
    
    ELEVENLABS_STABILITY: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Stabilité de la voix (0 = plus variable, 1 = plus stable)"
    )
    
    ELEVENLABS_SIMILARITY: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarité à la voix originale"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 🎤 CONFIGURATION WHISPER (STT)
    # ═══════════════════════════════════════════════════════════════
    
    WHISPER_MODEL: str = Field(
        default="base",
        description="Modèle Whisper: tiny, base, small, medium, large"
    )
    
    WHISPER_LANGUAGE: str = Field(
        default="fr",
        description="Langue principale pour la transcription"
    )
    
    WHISPER_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout en secondes pour la transcription"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 📚 CONFIGURATION RAG ET VECTORSTORE
    # ═══════════════════════════════════════════════════════════════
    
    DOCUMENTS_DIR: Path = Field(
        default=Path("data/documents"),
        description="Répertoire contenant les documents Markdown"
    )
    
    VECTORSTORE_DIR: Path = Field(
        default=Path("data/vectorstore"),
        description="Répertoire de stockage de l'index FAISS"
    )
    
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="Taille des chunks pour la vectorisation"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Chevauchement entre les chunks"
    )
    
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Modèle d'embedding pour la vectorisation"
    )
    
    MAX_SEARCH_RESULTS: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum de documents à récupérer dans RAG"
    )
    
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Seuil de similarité pour filtrer les résultats RAG"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 📁 CONFIGURATION FICHIERS
    # ═══════════════════════════════════════════════════════════════
    
    MAX_UPLOAD_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Taille maximum des fichiers uploadés (en bytes)"
    )
    
    ALLOWED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "ogg", "m4a", "flac"],
        description="Formats audio acceptés pour STT"
    )
    
    ALLOWED_DOCUMENT_FORMATS: List[str] = Field(
        default=["md", "txt"],
        description="Formats de documents acceptés pour l'ingestion"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 📊 CONFIGURATION LOGS
    # ═══════════════════════════════════════════════════════════════
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Niveau de log: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Format des logs"
    )
    
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Fichier de log (None = sortie console uniquement)"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # ⚡ CONFIGURATION PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    
    REQUEST_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout global des requêtes HTTP en secondes"
    )
    
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Nombre maximum de requêtes simultanées"
    )
    
    CACHE_TTL: int = Field(
        default=3600,  # 1 heure
        description="Durée de vie du cache en secondes"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 🛡️ VALIDATEURS PERSONNALISÉS
    # ═══════════════════════════════════════════════════════════════
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Valide que l'environnement est reconnu."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'ENVIRONMENT doit être parmi: {allowed}')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Valide le niveau de log."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'LOG_LEVEL doit être parmi: {allowed}')
        return v.upper()
    
    @validator('OPENAI_MODEL')
    def validate_openai_model(cls, v):
        """Valide le modèle OpenAI."""
        allowed_models = [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-4-turbo-preview',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k',
            'gpt-4o',
            'gpt-4o-mini'
        ]
        if v not in allowed_models:
            raise ValueError(f'OPENAI_MODEL doit être parmi: {allowed_models}')
        return v
    
    @validator('WHISPER_MODEL')
    def validate_whisper_model(cls, v):
        """Valide le modèle Whisper."""
        allowed = ['tiny', 'base', 'small', 'medium', 'large']
        if v not in allowed:
            raise ValueError(f'WHISPER_MODEL doit être parmi: {allowed}')
        return v
    
    @validator('DOCUMENTS_DIR', 'VECTORSTORE_DIR')
    def validate_directories(cls, v):
        """S'assure que les répertoires existent ou peuvent être créés."""
        if isinstance(v, str):
            v = Path(v)
        
        # Créer le répertoire s'il n'existe pas
        try:
            v.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f'Impossible de créer le répertoire {v}: {e}')
        
        return v
    
    @validator('CHUNK_OVERLAP')
    def validate_chunk_overlap(cls, v, values):
        """S'assure que le chevauchement est inférieur à la taille des chunks."""
        # Convertir en int si c'est une string depuis l'environnement
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                raise ValueError('CHUNK_OVERLAP doit être un nombre entier')
        
        # Récupérer CHUNK_SIZE depuis values ou utiliser la valeur par défaut
        chunk_size = values.get('CHUNK_SIZE', 1000)
        if isinstance(chunk_size, str):
            try:
                chunk_size = int(chunk_size)
            except ValueError:
                chunk_size = 1000  # Valeur par défaut
                
        if v >= chunk_size:
            raise ValueError('CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE')
        return v
    
    # ═══════════════════════════════════════════════════════════════
    # ⚙️ PROPRIÉTÉS CALCULÉES
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def is_production(self) -> bool:
        """Retourne True si on est en production."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Retourne True si on est en développement."""
        return self.ENVIRONMENT == "development"
    
    @property
    def server_url(self) -> str:
        """URL complète du serveur."""
        return f"http://{self.HOST}:{self.PORT}"
    
    @property
    def has_elevenlabs(self) -> bool:
        """Retourne True si ElevenLabs est configuré."""
        return self.ELEVENLABS_API_KEY is not None
    
    @property
    def vectorstore_path(self) -> Path:
        """Chemin complet vers l'index FAISS."""
        return self.VECTORSTORE_DIR / "faiss_index"
    
    # ═══════════════════════════════════════════════════════════════
    # 📁 CONFIGURATION PYDANTIC
    # ═══════════════════════════════════════════════════════════════
    
    class Config:
        """Configuration Pydantic."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Documentation des champs
        schema_extra = {
            "example": {
                "DEBUG": True,
                "OPENAI_API_KEY": "sk-...",
                "ELEVENLABS_API_KEY": "your-elevenlabs-key",
                "HOST": "0.0.0.0",
                "PORT": 8000
            }
        }


# ═══════════════════════════════════════════════════════════════
# 🏭 SINGLETON PATTERN POUR LA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@lru_cache()
def get_settings() -> Settings:
    """
    Retourne l'instance singleton des paramètres.
    
    Utilise @lru_cache() pour éviter de recharger la config à chaque appel.
    La configuration est chargée une seule fois au démarrage.
    
    Returns:
        Settings: Instance configurée de l'application
    """
    return Settings()


# ═══════════════════════════════════════════════════════════════
# 🧪 FONCTION D'AIDE POUR LES TESTS
# ═══════════════════════════════════════════════════════════════

def get_test_settings(**overrides) -> Settings:
    """
    Crée une configuration de test avec des surcharges.
    
    Args:
        **overrides: Paramètres à surcharger
        
    Returns:
        Settings: Configuration de test
        
    Example:
        settings = get_test_settings(
            DEBUG=True,
            CLAUDE_API_KEY="test-key"
        )
    """
    # Efface le cache pour permettre une nouvelle instance
    get_settings.cache_clear()
    
    # Variables d'environnement temporaires
    original_env = {}
    
    try:
        # Sauvegarde et modification des variables d'environnement
        for key, value in overrides.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        # Création de la nouvelle configuration
        settings = Settings()
        
        return settings
        
    finally:
        # Restauration des variables d'environnement
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
                
                # Ajout dans app/config.py
@validator('ALLOWED_ORIGINS')
def validate_origins(cls, v):
    """Ajoute automatiquement les origines Render."""
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.0.2.2:8000",  # Android Emulator
        "https://*.onrender.com",  # Render domains
        "*"  # TEMPORAIRE pour développement mobile
    ]
    return list(set(v + default_origins))

# Ajout configuration production
@property
def is_render_deployment(self) -> bool:
    """Détecte si on est sur Render."""
    return "RENDER" in os.environ

@property
def render_external_url(self) -> Optional[str]:
    """URL externe Render."""
    return os.environ.get("RENDER_EXTERNAL_URL")