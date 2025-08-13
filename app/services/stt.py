"""
Service STT Multi-Provider - Reconnaissance Vocale
=================================================

Service de reconnaissance vocale optimisé pour l'inclusion numérique
des femmes rurales. Support multi-providers avec fallbacks intelligents.

Fonctionnalités:
- Support OpenAI Whisper, Google Speech-to-Text, Azure Speech
- Optimisation pour accents régionaux et français parlé
- Détection automatique de langue
- Filtrage du bruit et amélioration audio
- Gestion intelligente des erreurs avec fallbacks
"""

import asyncio
import tempfile
import time
import io
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import logging
import json
from enum import Enum

import httpx
import openai
from google.cloud import speech
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter

from app.utils.exceptions import (
    STTError,
    STTProviderError,
    STTQuotaError,
    AudioFormatError,
    AudioProcessingError
)
from app.utils.logger import log_metrics


class STTProvider(Enum):
    """Providers STT disponibles."""
    OPENAI_WHISPER = "openai_whisper"
    GOOGLE_SPEECH = "google_speech"
    AZURE_SPEECH = "azure_speech"
    VOSK_OFFLINE = "vosk_offline"


class AudioFormat(Enum):
    """Formats audio supportés."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    WEBM = "webm"
    OGG = "ogg"
    M4A = "m4a"


class STTService:
    """
    Service de reconnaissance vocale multi-provider.
    
    Optimisé pour l'accessibilité des femmes rurales non-alphabétisées.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        google_credentials_path: Optional[str] = None,
        azure_speech_key: Optional[str] = None,
        azure_region: str = "francecentral",
        primary_provider: STTProvider = STTProvider.OPENAI_WHISPER,
        target_language: str = "fr-FR",
        enable_profanity_filter: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise le service STT multi-provider.
        
        Args:
            openai_api_key: Clé API OpenAI pour Whisper
            google_credentials_path: Chemin vers credentials Google Cloud
            azure_speech_key: Clé Azure Cognitive Services
            azure_region: Région Azure (francecentral recommandé)
            primary_provider: Provider principal à utiliser
            target_language: Langue cible (fr-FR par défaut)
            enable_profanity_filter: Activer le filtre de profanité
            logger: Logger optionnel
        """
        self.openai_api_key = openai_api_key
        self.google_credentials_path = google_credentials_path
        self.azure_speech_key = azure_speech_key
        self.azure_region = azure_region
        self.primary_provider = primary_provider
        self.target_language = target_language
        self.enable_profanity_filter = enable_profanity_filter
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration des providers
        self.providers_config = {
            STTProvider.OPENAI_WHISPER: {
                "enabled": bool(openai_api_key),
                "model": "whisper-1",
                "max_file_size_mb": 25,
                "supported_formats": [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.M4A],
                "cost_per_minute": 0.006  # $0.006/minute
            },
            STTProvider.GOOGLE_SPEECH: {
                "enabled": bool(google_credentials_path),
                "model": "latest_long",
                "max_file_size_mb": 10,
                "supported_formats": [AudioFormat.FLAC, AudioFormat.WAV],
                "cost_per_minute": 0.024  # $0.024/minute pour enhanced model
            },
            STTProvider.AZURE_SPEECH: {
                "enabled": bool(azure_speech_key),
                "model": "fr-FR-DeniseNeural",  # Voix française optimisée
                "max_file_size_mb": 100,
                "supported_formats": [AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OGG],
                "cost_per_minute": 0.02  # $0.02/minute
            }
        }
        
        # Ordre de fallback des providers
        self.fallback_order = [
            STTProvider.OPENAI_WHISPER,  # Le plus robuste pour les accents
            STTProvider.GOOGLE_SPEECH,   # Bon pour le français standard
            STTProvider.AZURE_SPEECH,    # Backup fiable
        ]
        
        # Statistiques d'usage
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "provider_usage": {provider.value: 0 for provider in STTProvider},
            "provider_success": {provider.value: 0 for provider in STTProvider},
            "average_confidence": 0.0,
            "language_detection_accuracy": 0.0
        }
        
        # Cache des résultats (pour éviter retraitement)
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialisation des clients
        self._init_providers()
        
        self.logger.info(f"🎤 Service STT initialisé - Provider principal: {primary_provider.value}")
    
    def _init_providers(self):
        """Initialise les clients des différents providers."""
        
        # OpenAI Whisper
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("✅ OpenAI Whisper configuré")
        
        # Google Speech-to-Text
        if self.google_credentials_path:
            try:
                self.google_client = speech.SpeechClient.from_service_account_file(
                    self.google_credentials_path
                )
                self.logger.info("✅ Google Speech-to-Text configuré")
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur config Google Speech: {e}")
                self.providers_config[STTProvider.GOOGLE_SPEECH]["enabled"] = False
        
        # Azure Speech Services
        if self.azure_speech_key:
            try:
                self.azure_config = SpeechConfig(
                    subscription=self.azure_speech_key,
                    region=self.azure_region
                )
                # Configuration optimisée pour le français rural
                self.azure_config.speech_recognition_language = self.target_language
                self.azure_config.enable_audio_logging()
                self.logger.info("✅ Azure Speech Services configuré")
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur config Azure Speech: {e}")
                self.providers_config[STTProvider.AZURE_SPEECH]["enabled"] = False
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, str, Path],
        provider: Optional[STTProvider] = None,
        language: Optional[str] = None,
        enable_word_timestamps: bool = False,
        enhance_for_rural_speech: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Transcrit un fichier audio en texte.
        
        Args:
            audio_data: Données audio (bytes, chemin fichier, ou Path)
            provider: Provider spécifique à utiliser
            language: Code langue (ex: fr-FR, wo-SN pour wolof)
            enable_word_timestamps: Activer les timestamps par mot
            enhance_for_rural_speech: Optimiser pour la parole rurale
            confidence_threshold: Seuil de confiance minimum
            
        Returns:
            Dict contenant la transcription et métadonnées
            
        Raises:
            STTError: Si erreur lors de la transcription
        """
        
        start_time = time.time()
        target_language = language or self.target_language
        
        try:
            # Préparation des données audio
            processed_audio = await self._prepare_audio_data(
                audio_data, 
                enhance_for_rural_speech
            )
            
            # Vérification du cache
            cache_key = self._generate_cache_key(processed_audio["data"], target_language)
            if cache_key in self._results_cache:
                self.logger.info("📋 Résultat trouvé en cache")
                return self._results_cache[cache_key]
            
            # Sélection du provider
            selected_provider = provider or self.primary_provider
            
            # Tentative avec le provider principal
            result = await self._attempt_transcription(
                processed_audio,
                selected_provider,
                target_language,
                enable_word_timestamps,
                confidence_threshold
            )
            
            # Si échec, essayer les fallbacks
            if not result["success"] and not provider:
                for fallback_provider in self.fallback_order:
                    if (fallback_provider != selected_provider and 
                        self.providers_config[fallback_provider]["enabled"]):
                        
                        self.logger.info(f"🔄 Fallback vers {fallback_provider.value}")
                        
                        result = await self._attempt_transcription(
                            processed_audio,
                            fallback_provider,
                            target_language,
                            enable_word_timestamps,
                            confidence_threshold
                        )
                        
                        if result["success"]:
                            break
            
            # Finalisation du résultat
            processing_time = time.time() - start_time
            
            if result["success"]:
                # Post-traitement du texte pour l'éducation rurale
                result["text"] = self._post_process_text_for_rural_education(
                    result["text"], 
                    target_language
                )
                
                # Mise en cache
                self._results_cache[cache_key] = result
                
                # Statistiques
                self._update_stats(True, processing_time, result)
                
                # Métriques globales
                log_metrics.log_stt_call()
                
                self.logger.info(
                    f"✅ Transcription réussie: {len(result['text'])} caractères "
                    f"(confiance: {result['confidence']:.2f})"
                )
            else:
                self._update_stats(False, processing_time, result)
                self.logger.error(f"❌ Échec transcription: {result.get('error', 'Erreur inconnue')}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time, {"provider": "error"})
            self.logger.error(f"❌ Erreur transcription: {e}")
            raise STTError(f"Erreur lors de la transcription: {str(e)}")
    
    async def _prepare_audio_data(
        self, 
        audio_input: Union[bytes, str, Path],
        enhance_for_rural: bool = True
    ) -> Dict[str, Any]:
        """
        Prépare et optimise les données audio pour la transcription.
        
        Args:
            audio_input: Données audio d'entrée
            enhance_for_rural: Activer les optimisations pour la parole rurale
            
        Returns:
            Dict avec données audio traitées et métadonnées
        """
        
        try:
            # Chargement des données audio
            if isinstance(audio_input, (str, Path)):
                audio_path = Path(audio_input)
                if not audio_path.exists():
                    raise AudioProcessingError(f"Fichier audio introuvable: {audio_path}")
                audio_data = audio_path.read_bytes()
                original_format = audio_path.suffix.lower().replace('.', '')
            else:
                audio_data = audio_input
                original_format = self._detect_audio_format(audio_data)
            
            # Chargement avec pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format=original_format
            )
            
            # Métadonnées originales
            original_metadata = {
                "duration_ms": len(audio_segment),
                "channels": audio_segment.channels,
                "frame_rate": audio_segment.frame_rate,
                "sample_width": audio_segment.sample_width,
                "format": original_format,
                "size_bytes": len(audio_data)
            }
            
            self.logger.info(
                f"🎵 Audio chargé: {original_metadata['duration_ms']/1000:.1f}s, "
                f"{original_metadata['channels']}ch, {original_metadata['frame_rate']}Hz"
            )
            
            # Optimisations pour la parole rurale
            if enhance_for_rural:
                audio_segment = await self._enhance_audio_for_rural_speech(audio_segment)
            
            # Conversion au format optimal (16kHz, mono, WAV)
            optimized_audio = (
                audio_segment
                .set_frame_rate(16000)  # 16kHz optimal pour STT
                .set_channels(1)        # Mono
                .set_sample_width(2)    # 16-bit
            )
            
            # Export en WAV pour compatibilité maximale
            wav_buffer = io.BytesIO()
            optimized_audio.export(wav_buffer, format="wav")
            wav_data = wav_buffer.getvalue()
            
            processed_metadata = {
                "duration_ms": len(optimized_audio),
                "channels": 1,
                "frame_rate": 16000,
                "sample_width": 2,
                "format": "wav",
                "size_bytes": len(wav_data),
                "enhancement_applied": enhance_for_rural
            }
            
            return {
                "data": wav_data,
                "original_metadata": original_metadata,
                "processed_metadata": processed_metadata,
                "duration_seconds": len(optimized_audio) / 1000.0
            }
            
        except Exception as e:
            raise AudioProcessingError(f"Erreur traitement audio: {str(e)}")
    
    async def _enhance_audio_for_rural_speech(self, audio: AudioSegment) -> AudioSegment:
        """
        Applique des optimisations spécifiques pour la parole rurale.
        
        Args:
            audio: Segment audio à optimiser
            
        Returns:
            AudioSegment: Audio optimisé
        """
        
        try:
            # Normalisation du volume
            normalized_audio = normalize(audio)
            
            # Filtre passe-haut pour réduire le bruit de fond
            # (élimine les basses fréquences souvent présentes en milieu rural)
            filtered_audio = high_pass_filter(normalized_audio, cutoff=100)
            
            # Réduction du bruit avancée avec librosa
            if len(filtered_audio) > 1000:  # Éviter les erreurs sur audio très court
                # Conversion pour librosa
                audio_array = np.array(filtered_audio.get_array_of_samples())
                if filtered_audio.channels == 2:
                    audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
                
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Réduction du bruit stationnaire
                # (utile pour éliminer le bruit constant des générateurs, ventilateurs, etc.)
                noise_reduced = librosa.effects.preemphasis(audio_float, coef=0.97)
                
                # Reconversion
                audio_int = (noise_reduced * 32767).astype(np.int16)
                enhanced_audio = AudioSegment(
                    audio_int.tobytes(),
                    frame_rate=filtered_audio.frame_rate,
                    sample_width=filtered_audio.sample_width,
                    channels=1
                )
            else:
                enhanced_audio = filtered_audio
            
            self.logger.debug("🔧 Optimisations audio rurales appliquées")
            return enhanced_audio
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur optimisation audio: {e}")
            # Retour au audio original en cas d'erreur
            return audio
    
    async def _attempt_transcription(
        self,
        audio_data: Dict[str, Any],
        provider: STTProvider,
        language: str,
        word_timestamps: bool,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """
        Tente une transcription avec un provider spécifique.
        
        Args:
            audio_data: Données audio préparées
            provider: Provider STT à utiliser
            language: Code langue
            word_timestamps: Activer timestamps
            confidence_threshold: Seuil de confiance
            
        Returns:
            Dict: Résultat de transcription
        """
        
        provider_start = time.time()
        
        try:
            # Vérification de la disponibilité du provider
            if not self.providers_config[provider]["enabled"]:
                return {
                    "success": False,
                    "error": f"Provider {provider.value} non configuré",
                    "provider": provider.value
                }
            
            # Vérification de la taille du fichier
            max_size_mb = self.providers_config[provider]["max_file_size_mb"]
            size_mb = len(audio_data["data"]) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                return {
                    "success": False,
                    "error": f"Fichier trop volumineux ({size_mb:.1f}MB > {max_size_mb}MB)",
                    "provider": provider.value
                }
            
            # Transcription selon le provider
            if provider == STTProvider.OPENAI_WHISPER:
                result = await self._transcribe_with_whisper(
                    audio_data, language, word_timestamps
                )
            elif provider == STTProvider.GOOGLE_SPEECH:
                result = await self._transcribe_with_google(
                    audio_data, language, word_timestamps
                )
            elif provider == STTProvider.AZURE_SPEECH:
                result = await self._transcribe_with_azure(
                    audio_data, language, word_timestamps
                )
            else:
                return {
                    "success": False,
                    "error": f"Provider {provider.value} non implémenté",
                    "provider": provider.value
                }
            
            # Vérification du seuil de confiance
            if result["success"] and result["confidence"] < confidence_threshold:
                result["success"] = False
                result["error"] = f"Confiance trop faible ({result['confidence']:.2f} < {confidence_threshold})"
            
            # Ajout des métadonnées
            result.update({
                "provider": provider.value,
                "processing_time_ms": (time.time() - provider_start) * 1000,
                "audio_duration_seconds": audio_data["duration_seconds"],
                "language_detected": result.get("language_detected", language)
            })
            
            # Mise à jour stats provider
            self.stats["provider_usage"][provider.value] += 1
            if result["success"]:
                self.stats["provider_success"][provider.value] += 1
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur {provider.value}: {str(e)}",
                "provider": provider.value,
                "processing_time_ms": (time.time() - provider_start) * 1000
            }
    
    async def _transcribe_with_whisper(
        self, 
        audio_data: Dict[str, Any],
        language: str,
        word_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcription avec OpenAI Whisper."""
        
        try:
            # Préparation du fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data["data"])
                temp_file_path = temp_file.name
            
            try:
                # Configuration Whisper optimisée pour le français rural
                whisper_language = language.split('-')[0] if '-' in language else language
                
                # Appel API Whisper
                with open(temp_file_path, "rb") as audio_file:
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: openai.Audio.transcribe(
                            model="whisper-1",
                            file=audio_file,
                            language=whisper_language,
                            response_format="verbose_json" if word_timestamps else "json",
                            temperature=0.2,  # Faible température pour plus de précision
                            prompt="Transcription en français standard avec ponctuation correcte."
                        )
                    )
                
                # Traitement de la réponse
                text = transcript.get("text", "").strip()
                confidence = 0.9  # Whisper ne fournit pas de score de confiance direct
                
                # Analyse de la qualité pour estimer la confiance
                if len(text) > 0:
                    # Heuristiques pour estimer la confiance
                    word_count = len(text.split())
                    duration = audio_data["duration_seconds"]
                    
                    # Ratio mots/seconde normal : 2-4 mots/sec
                    words_per_second = word_count / max(duration, 1)
                    if words_per_second < 0.5 or words_per_second > 6:
                        confidence *= 0.8  # Réduction si ratio anormal
                    
                    # Pénalité pour texte très court
                    if len(text) < 10:
                        confidence *= 0.7
                
                result = {
                    "success": True,
                    "text": text,
                    "confidence": confidence,
                    "language_detected": transcript.get("language", whisper_language),
                    "word_count": len(text.split()) if text else 0
                }
                
                # Ajout des timestamps si demandé
                if word_timestamps and "segments" in transcript:
                    result["word_timestamps"] = [
                        {
                            "start": segment.get("start", 0),
                            "end": segment.get("end", 0),
                            "text": segment.get("text", "")
                        }
                        for segment in transcript["segments"]
                    ]
                
                return result
                
            finally:
                # Nettoyage du fichier temporaire
                Path(temp_file_path).unlink(missing_ok=True)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur Whisper: {str(e)}",
                "text": "",
                "confidence": 0.0
            }
    
    async def _transcribe_with_google(
        self,
        audio_data: Dict[str, Any],
        language: str,
        word_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcription avec Google Speech-to-Text."""
        
        try:
            # Configuration Google Speech
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
                model="latest_long",  # Modèle optimisé pour audio long
                use_enhanced=True,    # Modèle amélioré (coût plus élevé mais meilleure qualité)
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=word_timestamps,
                # Optimisations pour français parlé
                speech_contexts=[
                    speech.SpeechContext(
                        phrases=["bonjour", "merci", "s'il vous plaît", "au revoir", "oui", "non"],
                        boost=10.0
                    )
                ],
                profanity_filter=self.enable_profanity_filter,
                max_alternatives=1
            )
            
            # Préparation de l'audio
            audio = speech.RecognitionAudio(content=audio_data["data"])
            
            # Reconnaissance
            operation = self.google_client.long_running_recognize(
                config=config,
                audio=audio
            )
            
            # Attente du résultat (avec timeout)
            response = operation.result(timeout=90)
            
            # Traitement de la réponse
            if response.results:
                # Meilleur résultat
                best_result = response.results[0]
                alternative = best_result.alternatives[0] if best_result.alternatives else None
                
                if alternative:
                    text = alternative.transcript.strip()
                    confidence = alternative.confidence
                    
                    result = {
                        "success": True,
                        "text": text,
                        "confidence": confidence,
                        "word_count": len(text.split()) if text else 0
                    }
                    
                    # Timestamps des mots si demandé
                    if word_timestamps and hasattr(alternative, 'words'):
                        result["word_timestamps"] = [
                            {
                                "start": word.start_time.total_seconds(),
                                "end": word.end_time.total_seconds(),
                                "text": word.word,
                                "confidence": word.confidence if hasattr(word, 'confidence') else confidence
                            }
                            for word in alternative.words
                        ]
                    
                    return result
            
            # Aucun résultat trouvé
            return {
                "success": False,
                "error": "Aucune transcription trouvée",
                "text": "",
                "confidence": 0.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur Google Speech: {str(e)}",
                "text": "",
                "confidence": 0.0
            }
    
    async def _transcribe_with_azure(
        self,
        audio_data: Dict[str, Any],
        language: str,
        word_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcription avec Azure Speech Services."""
        
        try:
            # Configuration Azure spécifique à la langue
            self.azure_config.speech_recognition_language = language
            
            # Sauvegarde temporaire de l'audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data["data"])
                temp_file_path = temp_file.name
            
            try:
                # Configuration audio
                audio_config = AudioConfig(filename=temp_file_path)
                
                # Création du recognizer
                recognizer = SpeechRecognizer(
                    speech_config=self.azure_config,
                    audio_config=audio_config
                )
                
                # Reconnaissance (synchrone pour simplicité)
                result_future = asyncio.Future()
                
                def result_callback(evt):
                    if not result_future.done():
                        result_future.set_result(evt.result)
                
                def error_callback(evt):
                    if not result_future.done():
                        result_future.set_exception(Exception(f"Azure Speech Error: {evt.result.reason}"))
                
                recognizer.recognized.connect(result_callback)
                recognizer.session_stopped.connect(lambda evt: None)
                recognizer.canceled.connect(error_callback)
                
                # Démarrage de la reconnaissance
                recognizer.start_continuous_recognition()
                
                try:
                    # Attente du résultat avec timeout
                    azure_result = await asyncio.wait_for(result_future, timeout=60)
                    
                    if azure_result.reason == speech.ResultReason.RecognizedSpeech:
                        text = azure_result.text.strip()
                        # Azure ne fournit pas de score de confiance direct
                        confidence = 0.85  # Score estimé
                        
                        return {
                            "success": True,
                            "text": text,
                            "confidence": confidence,
                            "word_count": len(text.split()) if text else 0,
                            "azure_result_id": azure_result.result_id
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Azure recognition failed: {azure_result.reason}",
                            "text": "",
                            "confidence": 0.0
                        }
                        
                finally:
                    recognizer.stop_continuous_recognition()
                    
            finally:
                # Nettoyage
                Path(temp_file_path).unlink(missing_ok=True)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Erreur Azure Speech: {str(e)}",
                "text": "",
                "confidence": 0.0
            }
    
    def _post_process_text_for_rural_education(self, text: str, language: str) -> str:
        """
        Post-traitement du texte transcrit pour l'éducation rurale.
        
        Args:
            text: Texte transcrit
            language: Code langue
            
        Returns:
            str: Texte optimisé
        """
        
        if not text:
            return text
        
        processed = text.strip()
        
        # Corrections spécifiques au français parlé rural
        if language.startswith("fr"):
            # Corrections d'expressions courantes mal transcrites
            corrections = {
                # Expressions courantes en français rural
                "sa va": "ça va",
                "sa marche": "ça marche", 
                "j'veux": "je veux",
                "j'sais": "je sais",
                "j'peux": "je peux",
                "t'es": "tu es",
                "qu'est-ce qu'il": "qu'est-ce qui",
                "y'a": "il y a",
                "j'ai dit": "j'ai dit",
                "comment ça s'appelle": "comment ça s'appelle",
                
                # Corrections numériques
                "1": "un", "2": "deux", "3": "trois", "4": "quatre", "5": "cinq",
                "6": "six", "7": "sept", "8": "huit", "9": "neuf", "10": "dix",
                
                # Mots souvent mal transcrits
                "internet": "internet", "ordinateur": "ordinateur", "téléphone": "téléphone",
                "whatsapp": "WhatsApp", "facebook": "Facebook", "google": "Google",
                
                # Expressions de politesse
                "stp": "s'il te plaît", "svp": "s'il vous plaît",
                "merci bcp": "merci beaucoup", "de rien": "de rien"
            }
            
            # Application des corrections
            for incorrect, correct in corrections.items():
                processed = processed.replace(incorrect, correct)
            
            # Normalisation de la ponctuation
            processed = processed.replace("...", ".")
            processed = processed.replace("??", "?")
            processed = processed.replace("!!", "!")
            
            # Suppression des mots de remplissage excessifs
            filler_words = ["euh", "heu", "ben", "bah", "alors", "donc", "voilà"]
            words = processed.split()
            
            # Garder maximum 1 mot de remplissage consécutif
            filtered_words = []
            prev_was_filler = False
            
            for word in words:
                word_lower = word.lower().strip(".,!?")
                is_filler = word_lower in filler_words
                
                if is_filler and prev_was_filler:
                    continue  # Skip consecutive filler words
                
                filtered_words.append(word)
                prev_was_filler = is_filler
            
            processed = " ".join(filtered_words)
        
        # Nettoyage final
        processed = " ".join(processed.split())  # Multiple spaces -> single space
        processed = processed.capitalize()       # Première lettre majuscule
        
        return processed
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """Détecte le format audio à partir des données binaires."""
        
        # Magic numbers pour détecter les formats
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
            return 'wav'
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            return 'mp3'
        elif audio_data.startswith(b'fLaC'):
            return 'flac'
        elif audio_data.startswith(b'OggS'):
            return 'ogg'
        elif b'ftypM4A' in audio_data[:20] or b'ftymp4' in audio_data[:20]:
            return 'm4a'
        else:
            # Default fallback
            return 'wav'
    
    def _generate_cache_key(self, audio_data: bytes, language: str) -> str:
        """Génère une clé de cache pour les données audio."""
        import hashlib
        
        # Hash des données audio + langue
        audio_hash = hashlib.md5(audio_data).hexdigest()[:16]
        return f"stt_{audio_hash}_{language}"
    
    def _update_stats(self, success: bool, processing_time: float, result: Dict[str, Any]):
        """Met à jour les statistiques d'usage."""
        
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            
            # Mise à jour de la confiance moyenne
            confidence = result.get("confidence", 0.0)
            current_avg = self.stats["average_confidence"]
            total_success = self.stats["successful_requests"]
            self.stats["average_confidence"] = (
                (current_avg * (total_success - 1) + confidence) / total_success
            )
        
        # Temps de traitement moyen
        current_avg_time = self.stats["total_processing_time"]
        total_requests = self.stats["total_requests"]
        self.stats["total_processing_time"] = (
            (current_avg_time * (total_requests - 1) + processing_time) / total_requests
        )
    
    async def detect_language(self, audio_data: Union[bytes, str, Path]) -> Dict[str, Any]:
        """
        Détecte la langue parlée dans un fichier audio.
        
        Args:
            audio_data: Données audio à analyser
            
        Returns:
            Dict: Langues détectées avec scores de confiance
        """
        
        try:
            # Utilisation de Whisper pour la détection de langue (le plus fiable)
            if self.providers_config[STTProvider.OPENAI_WHISPER]["enabled"]:
                
                # Préparation audio (version courte pour détection rapide)
                processed_audio = await self._prepare_audio_data(audio_data, False)
                
                # Truncate à 30 secondes pour économiser les crédits
                duration_ms = min(30000, processed_audio["processed_metadata"]["duration_ms"])
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(processed_audio["data"][:duration_ms * 32])  # Approximation
                    temp_file_path = temp_file.name
                
                try:
                    with open(temp_file_path, "rb") as audio_file:
                        # Utilisation du modèle Whisper pour détection
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: openai.Audio.transcribe(
                                model="whisper-1",
                                file=audio_file,
                                response_format="verbose_json"
                            )
                        )
                    
                    detected_language = response.get("language", "unknown")
                    
                    # Mapping des codes langue Whisper vers codes standards
                    language_mapping = {
                        "fr": "fr-FR", "en": "en-US", "ar": "ar-MA", 
                        "wo": "wo-SN", "es": "es-ES", "pt": "pt-PT"
                    }
                    
                    standard_code = language_mapping.get(detected_language, f"{detected_language}-XX")
                    
                    return {
                        "success": True,
                        "primary_language": standard_code,
                        "confidence": 0.9,  # Whisper est très fiable pour la détection
                        "detected_languages": [
                            {
                                "language": standard_code,
                                "confidence": 0.9,
                                "name": self._get_language_name(standard_code)
                            }
                        ],
                        "provider": "whisper"
                    }
                    
                finally:
                    Path(temp_file_path).unlink(missing_ok=True)
            
            # Fallback si Whisper indisponible
            return {
                "success": False,
                "error": "Aucun provider de détection de langue disponible",
                "primary_language": self.target_language,
                "confidence": 0.5
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur détection langue: {e}")
            return {
                "success": False,
                "error": str(e),
                "primary_language": self.target_language,
                "confidence": 0.0
            }
    
    def _get_language_name(self, language_code: str) -> str:
        """Retourne le nom de la langue à partir du code."""
        
        language_names = {
            "fr-FR": "Français",
            "en-US": "Anglais",
            "ar-MA": "Arabe",
            "wo-SN": "Wolof",
            "es-ES": "Espagnol",
            "pt-PT": "Portugais",
            "de-DE": "Allemand",
            "it-IT": "Italien"
        }
        
        return language_names.get(language_code, language_code)
    
    async def transcribe_realtime_stream(
        self,
        audio_stream,
        provider: STTProvider = STTProvider.OPENAI_WHISPER,
        language: Optional[str] = None,
        chunk_duration_ms: int = 3000
    ):
        """
        Transcription en temps réel d'un flux audio.
        
        Args:
            audio_stream: Flux audio en temps réel
            provider: Provider à utiliser
            language: Langue cible
            chunk_duration_ms: Durée des chunks en millisecondes
            
        Yields:
            Dict: Résultats de transcription par chunks
        """
        
        target_language = language or self.target_language
        
        try:
            self.logger.info(f"🎙️ Démarrage transcription temps réel ({provider.value})")
            
            audio_buffer = AudioSegment.empty()
            chunk_counter = 0
            
            async for audio_chunk in audio_stream:
                # Accumulation des chunks audio
                chunk_segment = AudioSegment.from_raw(
                    audio_chunk,
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                
                audio_buffer += chunk_segment
                
                # Traitement par chunks de durée fixe
                while len(audio_buffer) >= chunk_duration_ms:
                    # Extraction du chunk à traiter
                    current_chunk = audio_buffer[:chunk_duration_ms]
                    audio_buffer = audio_buffer[chunk_duration_ms:]
                    
                    # Conversion en bytes
                    chunk_wav = io.BytesIO()
                    current_chunk.export(chunk_wav, format="wav")
                    chunk_data = {
                        "data": chunk_wav.getvalue(),
                        "duration_seconds": chunk_duration_ms / 1000.0,
                        "processed_metadata": {"format": "wav"}
                    }
                    
                    # Transcription du chunk
                    try:
                        result = await self._attempt_transcription(
                            chunk_data,
                            provider,
                            target_language,
                            False,  # Pas de timestamps en temps réel
                            0.6     # Seuil de confiance plus bas pour temps réel
                        )
                        
                        if result["success"] and result["text"].strip():
                            yield {
                                "chunk_id": chunk_counter,
                                "timestamp": time.time(),
                                "text": result["text"],
                                "confidence": result["confidence"],
                                "is_final": False,  # Résultat intermédiaire
                                "provider": provider.value
                            }
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erreur chunk {chunk_counter}: {e}")
                        
                    chunk_counter += 1
            
            # Traitement du dernier chunk restant
            if len(audio_buffer) > 1000:  # Au moins 1 seconde
                final_chunk_wav = io.BytesIO()
                audio_buffer.export(final_chunk_wav, format="wav")
                
                final_chunk_data = {
                    "data": final_chunk_wav.getvalue(),
                    "duration_seconds": len(audio_buffer) / 1000.0,
                    "processed_metadata": {"format": "wav"}
                }
                
                try:
                    final_result = await self._attempt_transcription(
                        final_chunk_data,
                        provider,
                        target_language,
                        False,
                        0.6
                    )
                    
                    if final_result["success"]:
                        yield {
                            "chunk_id": chunk_counter,
                            "timestamp": time.time(),
                            "text": final_result["text"],
                            "confidence": final_result["confidence"],
                            "is_final": True,  # Résultat final
                            "provider": provider.value
                        }
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Erreur chunk final: {e}")
            
            self.logger.info(f"✅ Transcription temps réel terminée ({chunk_counter} chunks)")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur transcription temps réel: {e}")
            yield {
                "error": str(e),
                "timestamp": time.time(),
                "is_final": True
            }
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des langues supportées.
        
        Returns:
            List[Dict]: Langues supportées avec métadonnées
        """
        
        languages = [
            {
                "code": "fr-FR",
                "name": "Français (France)",
                "native_name": "Français",
                "optimal_for_rural": True,
                "providers": ["whisper", "google", "azure"],
                "priority": 1
            },
            {
                "code": "wo-SN", 
                "name": "Wolof (Sénégal)",
                "native_name": "Wolof",
                "optimal_for_rural": True,
                "providers": ["whisper"],  # Seul Whisper supporte bien le wolof
                "priority": 2
            },
            {
                "code": "ar-MA",
                "name": "Arabe (Maghreb)",
                "native_name": "العربية",
                "optimal_for_rural": True,
                "providers": ["whisper", "google", "azure"],
                "priority": 3
            },
            {
                "code": "en-US",
                "name": "Anglais",
                "native_name": "English",
                "optimal_for_rural": False,
                "providers": ["whisper", "google", "azure"],
                "priority": 4
            }
        ]
        
        # Filtrer selon les providers disponibles
        available_languages = []
        for lang in languages:
            available_providers = []
            for provider_name in lang["providers"]:
                if provider_name == "whisper" and self.providers_config[STTProvider.OPENAI_WHISPER]["enabled"]:
                    available_providers.append("whisper")
                elif provider_name == "google" and self.providers_config[STTProvider.GOOGLE_SPEECH]["enabled"]:
                    available_providers.append("google")
                elif provider_name == "azure" and self.providers_config[STTProvider.AZURE_SPEECH]["enabled"]:
                    available_providers.append("azure")
            
            if available_providers:
                lang["available_providers"] = available_providers
                available_languages.append(lang)
        
        return available_languages
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées du service STT.
        
        Returns:
            Dict: Statistiques complètes
        """
        
        success_rate = 0
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_requests"] / self.stats["total_requests"]) * 100
        
        # Calcul de la fiabilité par provider
        provider_reliability = {}
        for provider, usage_count in self.stats["provider_usage"].items():
            if usage_count > 0:
                success_count = self.stats["provider_success"][provider]
                reliability = (success_count / usage_count) * 100
                provider_reliability[provider] = reliability
            else:
                provider_reliability[provider] = 0
        
        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "provider_reliability": provider_reliability,
            "primary_provider": self.primary_provider.value,
            "target_language": self.target_language,
            "cache_size": len(self._results_cache),
            "average_processing_time_ms": self.stats["total_processing_time"] * 1000,
            "estimated_cost_usd": self._estimate_usage_cost(),
            "providers_status": {
                provider.value: self.providers_config[provider]["enabled"]
                for provider in STTProvider
            }
        }
    
    def _estimate_usage_cost(self) -> float:
        """Estime le coût d'usage basé sur la durée audio traitée."""
        
        total_minutes = self.stats["total_audio_duration"] / 60
        estimated_cost = 0.0
        
        # Estimation basée sur l'usage des providers
        for provider, usage_count in self.stats["provider_usage"].items():
            if usage_count > 0:
                provider_enum = STTProvider(provider)
                cost_per_minute = self.providers_config[provider_enum]["cost_per_minute"]
                
                # Estimation proportionnelle
                provider_proportion = usage_count / self.stats["total_requests"]
                provider_minutes = total_minutes * provider_proportion
                estimated_cost += provider_minutes * cost_per_minute
        
        return round(estimated_cost, 4)
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion aux différents providers STT.
        
        Returns:
            Dict: Résultats des tests
        """
        
        test_results = {
            "overall_success": False,
            "providers": {},
            "timestamp": time.time()
        }
        
        # Audio de test (silence de 1 seconde)
        test_audio = AudioSegment.silent(duration=1000, frame_rate=16000).set_channels(1)
        test_wav = io.BytesIO()
        test_audio.export(test_wav, format="wav")
        
        test_audio_data = {
            "data": test_wav.getvalue(),
            "duration_seconds": 1.0,
            "processed_metadata": {"format": "wav"}
        }
        
        # Test de chaque provider disponible
        at_least_one_success = False
        
        for provider in STTProvider:
            if self.providers_config[provider]["enabled"]:
                try:
                    start_time = time.time()
                    
                    result = await self._attempt_transcription(
                        test_audio_data,
                        provider,
                        self.target_language,
                        False,
                        0.1  # Seuil très bas pour test
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    test_results["providers"][provider.value] = {
                        "success": result["success"],
                        "response_time_ms": response_time,
                        "error": result.get("error") if not result["success"] else None
                    }
                    
                    if result["success"]:
                        at_least_one_success = True
                        
                except Exception as e:
                    test_results["providers"][provider.value] = {
                        "success": False,
                        "error": f"Test error: {str(e)}",
                        "response_time_ms": 0
                    }
            else:
                test_results["providers"][provider.value] = {
                    "success": False,
                    "error": "Provider not configured",
                    "response_time_ms": 0
                }
        
        test_results["overall_success"] = at_least_one_success
        
        return test_results
    
    async def close(self):
        """Nettoie les ressources du service STT."""
        
        try:
            # Nettoyage du cache
            self._results_cache.clear()
            
            # Log des statistiques finales
            stats = self.get_stats()
            self.logger.info(f"📊 Statistiques STT finales:")
            self.logger.info(f"   - Requêtes: {stats['total_requests']}")
            self.logger.info(f"   - Taux de succès: {stats['success_rate_percent']:.1f}%")
            self.logger.info(f"   - Durée audio: {stats['total_audio_duration']:.1f}s")
            self.logger.info(f"   - Coût estimé: ${stats['estimated_cost_usd']}")
            
            self.logger.info("🔒 Service STT fermé")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur fermeture STT: {e}")