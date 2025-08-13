"""
Service TTS ElevenLabs - Synthèse Vocale
========================================

Service de synthèse vocale utilisant l'API ElevenLabs pour le chatbot
d'inclusion numérique. Optimisé pour l'accessibilité des femmes non-alphabétisées.

Fonctionnalités:
- Synthèse vocale avec voix naturelles
- Adaptation vitesse et intonation
- Support multi-langues (français prioritaire)
- Gestion des erreurs et fallbacks
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

import httpx
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from app.utils.exceptions import (
    ElevenLabsError,
    ElevenLabsQuotaError,
    TTSError,
    AudioFormatError
)
from app.utils.logger import log_metrics


class TTSService:
    """
    Service de synthèse vocale utilisant ElevenLabs.
    
    Optimisé pour l'accessibilité et l'inclusion numérique des femmes rurales.
    """
    
    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel - voix féminine claire
        model: str = "eleven_multilingual_v2",
        stability: float = 0.75,
        similarity: float = 0.75,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise le service TTS ElevenLabs.
        
        Args:
            api_key: Clé API ElevenLabs
            voice_id: ID de la voix (Rachel par défaut - idéale pour l'éducation)
            model: Modèle ElevenLabs à utiliser
            stability: Stabilité de la voix (0-1)
            similarity: Similarité à la voix originale (0-1)
            logger: Logger optionnel
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.stability = stability
        self.similarity = similarity
        self.logger = logger or logging.getLogger(__name__)
        
        # Client ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        
        # Configuration des voix pour l'éducation
        self.voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=0.0,  # Neutre pour l'éducation
            use_speaker_boost=True
        )
        
        # Statistiques d'usage
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_characters": 0,
            "total_audio_duration": 0.0,
            "average_processing_time": 0.0
        }
        
        # Cache des voix disponibles
        self._voices_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp = 0
        
        self.logger.info(f"🔊 Service TTS ElevenLabs initialisé - Voix: {voice_id}")
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stability: Optional[float] = None,
        similarity: Optional[float] = None,
        speed: float = 1.0,
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """
        Synthétise du texte en audio avec ElevenLabs.
        
        Args:
            text: Texte à synthétiser
            voice_id: ID de voix spécifique (optionnel)
            stability: Stabilité de la voix (optionnel)
            similarity: Similarité de la voix (optionnel)
            speed: Vitesse de lecture (0.5-2.0)
            output_format: Format audio (mp3, wav)
            
        Returns:
            Dict contenant les données audio et métadonnées
            
        Raises:
            TTSError: Si erreur lors de la synthèse
            ElevenLabsQuotaError: Si quota dépassé
        """
        
        start_time = time.time()
        
        try:
            # Validation des paramètres
            if not text or not text.strip():
                raise TTSError("Texte vide ou invalide")
            
            if len(text) > 5000:
                raise TTSError(f"Texte trop long ({len(text)} caractères, max: 5000)")
            
            # Préparation du texte pour l'éducation
            processed_text = self._prepare_text_for_education(text, speed)
            
            # Configuration de la voix
            current_voice_id = voice_id or self.voice_id
            voice_settings = VoiceSettings(
                stability=stability or self.stability,
                similarity_boost=similarity or self.similarity,
                style=0.0,  # Neutre pour l'éducation
                use_speaker_boost=True
            )
            
            self.logger.info(f"🔊 Synthèse TTS: {len(processed_text)} caractères")
            
            # Appel à l'API ElevenLabs
            try:
                audio_generator = self.client.generate(
                    text=processed_text,
                    voice=current_voice_id,
                    voice_settings=voice_settings,
                    model=self.model
                )
                
                # Collecte des données audio
                audio_data = b""
                for chunk in audio_generator:
                    audio_data += chunk
                
            except Exception as e:
                self._handle_elevenlabs_error(e)
            
            # Calcul des métriques
            processing_time = time.time() - start_time
            estimated_duration = self._estimate_audio_duration(processed_text, speed)
            
            # Mise à jour des statistiques
            self._update_stats(True, processing_time, len(processed_text), estimated_duration)
            
            # Sauvegarde temporaire du fichier audio
            audio_url = await self._save_temporary_audio(audio_data, output_format)
            
            # Métadonnées de la réponse
            result = {
                "success": True,
                "audio_data": audio_data,  # Données brutes
                "audio_url": audio_url,    # URL temporaire
                "text_processed": processed_text,
                "voice_id": current_voice_id,
                "metadata": {
                    "format": output_format,
                    "size_bytes": len(audio_data),
                    "estimated_duration_seconds": estimated_duration,
                    "processing_time_ms": processing_time * 1000,
                    "character_count": len(processed_text),
                    "voice_settings": {
                        "stability": voice_settings.stability,
                        "similarity": voice_settings.similarity_boost,
                        "speed_adjusted": speed
                    },
                    "educational_context": True,
                    "accessibility_optimized": True
                }
            }
            
            # Métriques globales
            log_metrics.log_tts_call()
            
            self.logger.info(f"✅ TTS réussi: {len(audio_data)} bytes, {estimated_duration:.1f}s")
            
            return result
            
        except ElevenLabsError:
            self._update_stats(False, time.time() - start_time, len(text), 0)
            raise
        except Exception as e:
            self._update_stats(False, time.time() - start_time, len(text), 0)
            self.logger.error(f"❌ Erreur TTS inattendue: {e}")
            raise TTSError(f"Erreur lors de la synthèse vocale: {str(e)}")
    
    def _prepare_text_for_education(self, text: str, speed: float) -> str:
        """
        Prépare le texte pour une synthèse vocale éducative optimale.
        
        Args:
            text: Texte original
            speed: Vitesse de lecture
            
        Returns:
            str: Texte optimisé pour l'éducation
        """
        
        # Nettoyage de base
        processed = text.strip()
        
        # Remplacement d'abréviations courantes en français
        replacements = {
            "etc.": "et cetera",
            "ex.": "exemple", 
            "c-à-d": "c'est-à-dire",
            "vs": "versus",
            "&": "et",
            "%": "pour cent",
            "€": "euros",
            "$": "dollars",
            "kg": "kilogrammes",
            "g": "grammes",
            "km": "kilomètres",
            "m": "mètres",
            "cm": "centimètres"
        }
        
        for abbrev, full in replacements.items():
            processed = processed.replace(abbrev, full)
        
        # Ajout de pauses pour la compréhension (important pour débutantes)
        if speed <= 0.8:  # Vitesse lente = plus de pauses
            processed = processed.replace(".", ". ... ")  # Pause longue après phrase
            processed = processed.replace(",", ", ... ")  # Pause courte après virgule
            processed = processed.replace(":", ": ... ")  # Pause après deux-points
        elif speed <= 1.2:  # Vitesse normale
            processed = processed.replace(".", ". .. ")   # Pause moyenne
            processed = processed.replace(",", ", .. ")   # Pause courte
        
        # Emphases pour l'éducation
        processed = processed.replace("IMPORTANT", "C'est très important")
        processed = processed.replace("ATTENTION", "Attention, c'est important")
        processed = processed.replace("NOTE", "Notez bien")
        
        return processed
    
    def _handle_elevenlabs_error(self, error: Exception):
        """
        Gère les erreurs spécifiques d'ElevenLabs.
        
        Args:
            error: Exception d'ElevenLabs
            
        Raises:
            ElevenLabsError: Exception métier appropriée
        """
        
        error_str = str(error).lower()
        
        # Quota dépassé
        if "quota" in error_str or "limit" in error_str or "429" in error_str:
            raise ElevenLabsQuotaError()
        
        # Clé API invalide
        elif "authentication" in error_str or "api key" in error_str or "401" in error_str:
            raise ElevenLabsError(
                "Clé API ElevenLabs invalide",
                details="Vérifiez votre clé API dans la configuration"
            )
        
        # Voix introuvable
        elif "voice" in error_str and "not found" in error_str:
            raise ElevenLabsError(
                "Voix ElevenLabs introuvable",
                details=f"Voix {self.voice_id} non disponible"
            )
        
        # Erreur générique
        else:
            raise ElevenLabsError(
                "Erreur du service ElevenLabs",
                details=str(error)
            )
    
    def _estimate_audio_duration(self, text: str, speed: float) -> float:
        """
        Estime la durée audio basée sur le texte et la vitesse.
        
        Args:
            text: Texte à synthétiser
            speed: Vitesse de lecture
            
        Returns:
            float: Durée estimée en secondes
        """
        
        # Estimation basée sur le nombre de mots
        # Vitesse moyenne: ~150 mots/minute en français
        # Ajusté pour l'éducation: plus lent pour la compréhension
        words = len(text.split())
        base_wpm = 120  # Mots par minute pour l'éducation
        
        # Ajustement selon la vitesse
        adjusted_wpm = base_wpm * speed
        
        duration_minutes = words / adjusted_wpm
        duration_seconds = duration_minutes * 60
        
        # Ajout de temps pour les pauses éducatives
        pause_time = text.count('.') * 0.5 + text.count(',') * 0.3
        
        return duration_seconds + pause_time
    
    async def _save_temporary_audio(self, audio_data: bytes, format: str) -> str:
        """
        Sauvegarde temporairement le fichier audio.
        
        Args:
            audio_data: Données audio
            format: Format du fichier
            
        Returns:
            str: URL d'accès temporaire
        """
        
        try:
            # Création d'un fichier temporaire
            timestamp = int(time.time())
            filename = f"tts_audio_{timestamp}.{format}"
            
            # Répertoire temporaire (à améliorer en production)
            temp_dir = Path("temp/audio")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            audio_path = temp_dir / filename
            
            # Sauvegarde
            audio_path.write_bytes(audio_data)
            
            # URL d'accès (relatif au serveur)
            audio_url = f"/temp/audio/{filename}"
            
            self.logger.debug(f"Audio sauvegardé: {audio_path} ({len(audio_data)} bytes)")
            
            return audio_url
            
        except Exception as e:
            self.logger.warning(f"Impossible de sauvegarder l'audio: {e}")
            return ""
    
    def _update_stats(self, success: bool, processing_time: float, 
                     character_count: int, duration: float):
        """Met à jour les statistiques d'usage."""
        
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            self.stats["total_characters"] += character_count
            self.stats["total_audio_duration"] += duration
            
            # Moyenne glissante du temps de traitement
            current_avg = self.stats["average_processing_time"]
            total_requests = self.stats["total_requests"]
            self.stats["average_processing_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des voix disponibles ElevenLabs.
        
        Returns:
            List[Dict]: Voix disponibles avec métadonnées
        """
        
        try:
            # Cache de 1 heure
            current_time = time.time()
            if (self._voices_cache and 
                current_time - self._cache_timestamp < 3600):
                return self._voices_cache
            
            # Récupération des voix depuis l'API
            voices_response = self.client.voices.get_all()
            
            # Formatage pour l'usage éducatif
            voices = []
            for voice in voices_response.voices:
                voice_info = {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category if hasattr(voice, 'category') else 'generated',
                    "description": getattr(voice, 'description', ''),
                    "gender": self._detect_voice_gender(voice.name),
                    "suitable_for_education": self._is_suitable_for_education(voice),
                    "language_support": ["en", "fr", "multilingual"],  # Approximation
                    "recommended_for_rural_education": voice.voice_id in [
                        "21m00Tcm4TlvDq8ikWAM",  # Rachel
                        "AZnzlk1XvdvUeBnXmlld",  # Domi
                        "EXAVITQu4vr4xnSDxMaL"   # Bella
                    ]
                }
                voices.append(voice_info)
            
            # Mise en cache
            self._voices_cache = voices
            self._cache_timestamp = current_time
            
            self.logger.info(f"📋 {len(voices)} voix ElevenLabs récupérées")
            
            return voices
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération voix: {e}")
            
            # Fallback avec voix par défaut
            return [
                {
                    "voice_id": self.voice_id,
                    "name": "Rachel (par défaut)",
                    "category": "premade",
                    "description": "Voix féminine claire et professionnelle",
                    "gender": "female",
                    "suitable_for_education": True,
                    "language_support": ["en", "fr", "multilingual"],
                    "recommended_for_rural_education": True
                }
            ]
    
    def _detect_voice_gender(self, voice_name: str) -> str:
        """Détecte le genre de la voix basé sur le nom."""
        female_names = [
            "rachel", "bella", "domi", "elli", "sarah", "charlotte", 
            "alice", "lily", "dorothy", "emily", "grace"
        ]
        
        if any(name in voice_name.lower() for name in female_names):
            return "female"
        else:
            return "male"
    
    def _is_suitable_for_education(self, voice) -> bool:
        """Détermine si la voix est adaptée à l'éducation."""
        # Critères pour l'éducation:
        # - Voix claire et professionnelle
        # - Pas trop expressive ou dramatique
        # - Adapté à un public rural
        
        voice_name = voice.name.lower()
        
        # Voix recommandées pour l'éducation
        educational_voices = [
            "rachel", "domi", "bella", "charlotte", "sarah",
            "daniel", "adam", "sam", "antoni"
        ]
        
        return any(name in voice_name for name in educational_voices)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'usage du service TTS.
        
        Returns:
            Dict: Statistiques détaillées
        """
        
        success_rate = 0
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_requests"] / self.stats["total_requests"]) * 100
        
        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "voice_id": self.voice_id,
            "model": self.model,
            "voice_settings": {
                "stability": self.stability,
                "similarity": self.similarity
            },
            "average_characters_per_request": (
                self.stats["total_characters"] / max(self.stats["total_requests"], 1)
            ),
            "estimated_cost_usd": self._estimate_usage_cost()
        }
    
    def _estimate_usage_cost(self) -> float:
        """Estime le coût d'usage basé sur les caractères."""
        # ElevenLabs: ~$0.30 per 1000 characters pour Multilingual v2
        characters = self.stats["total_characters"]
        estimated_cost = (characters / 1000) * 0.30
        return round(estimated_cost, 4)
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion au service ElevenLabs.
        
        Returns:
            Dict: Résultat du test
        """
        
        try:
            start_time = time.time()
            
            # Test simple avec texte court
            test_result = await self.synthesize(
                text="Test de connexion ElevenLabs",
                speed=1.0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "response_time_ms": response_time,
                "voice_id": self.voice_id,
                "service": "ElevenLabs TTS",
                "audio_generated": len(test_result["audio_data"]) > 0,
                "estimated_duration": test_result["metadata"]["estimated_duration_seconds"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "service": "ElevenLabs TTS",
                "voice_id": self.voice_id
            }
    
    async def close(self):
        """Nettoie les ressources du service."""
        try:
            # Nettoyage des fichiers temporaires (optionnel)
            temp_dir = Path("temp/audio")
            if temp_dir.exists():
                # Supprimer les fichiers de plus de 1 heure
                current_time = time.time()
                for audio_file in temp_dir.glob("tts_audio_*.mp3"):
                    if current_time - audio_file.stat().st_mtime > 3600:
                        audio_file.unlink()
            
            self.logger.info("🔒 Service TTS ElevenLabs fermé")
            
        except Exception as e:
            self.logger.warning(f"Erreur fermeture TTS: {e}")