"""
Endpoints Audio (TTS et STT)
============================

Routes pour la synthèse vocale (Text-to-Speech) et la reconnaissance 
vocale (Speech-to-Text) du chatbot éducatif.

Routes:
- POST /tts    → Synthèse vocale (texte vers audio)
- POST /stt    → Reconnaissance vocale (audio vers texte)
- GET /voices  → Liste des voix disponibles
"""

import time
import tempfile
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io

from app.config import get_settings
from app.dependencies import (
    get_tts_service_async,
    get_stt_service_async,
    validate_audio_file,
    validate_text_input
)
from app.models.requests import TTSRequest, STTRequest
from app.models.responses import (
    TTSResponse,
    STTResponse,
    AudioMetadata,
    create_success_response,
    create_error_response
)
from app.utils.logger import setup_logger, log_metrics
from app.utils.exceptions import (
    AudioError,
    ElevenLabsError,
    STTError,
    AudioFormatError,
    AudioTooLargeError
)

# ═══════════════════════════════════════════════════════════════
# 🏗️ CONFIGURATION DU ROUTER
# ═══════════════════════════════════════════════════════════════

router = APIRouter()
logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════
# 🔊 ENDPOINT DE SYNTHÈSE VOCALE (TTS)
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/tts",
    summary="Synthèse vocale",
    description="Convertit du texte en audio avec ElevenLabs",
    response_model=TTSResponse,
    responses={
        200: {"description": "Audio généré avec succès"},
        400: {"description": "Texte invalide"},
        413: {"description": "Texte trop long"},
        503: {"description": "Service TTS indisponible"},
        429: {"description": "Quota ElevenLabs dépassé"}
    }
)
async def text_to_speech(
    request: TTSRequest,
    tts_service = Depends(get_tts_service_async)
) -> TTSResponse:
    """
    Convertit du texte en audio avec synthèse vocale.
    
    Utilise ElevenLabs pour générer un audio de haute qualité
    adapté à l'éducation des personnes peu alphabétisées.
    
    Args:
        request: Paramètres de synthèse vocale
        tts_service: Service TTS injecté
        
    Returns:
        TTSResponse: Audio généré avec métadonnées
        
    Raises:
        HTTPException: Si service indisponible ou erreur de synthèse
    """
    
    if tts_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de synthèse vocale non configuré. Vérifiez la clé ElevenLabs."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"🔊 TTS: Synthèse de {len(request.text)} caractères")
        
        # Validation du texte
        cleaned_text = await validate_text_input(request.text)
        
        # Nettoyage spécifique pour TTS
        processed_text = prepare_text_for_tts(cleaned_text)
        
        # Synthèse vocale
        # audio_result = await tts_service.synthesize(
        #     text=processed_text,
        #     voice_id=request.voice_id,
        #     stability=request.stability,
        #     similarity=request.similarity,
        #     speed=request.speed
        # )
        
        # Simulation de synthèse pour l'instant
        audio_data = simulate_audio_generation(processed_text, request.format)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Métadonnées audio
        metadata = AudioMetadata(
            duration_seconds=estimate_audio_duration(processed_text, request.speed),
            format=request.format.value,
            size_bytes=len(audio_data),
            sample_rate=22050,  # Valeur par défaut ElevenLabs
            bitrate=128  # kbps
        )
        
        # Génération d'URL temporaire (simulation)
        audio_url = f"/temp/audio/tts_{int(time.time())}.{request.format.value}"
        
        response = TTSResponse(
            success=True,
            message="Audio généré avec succès",
            audio_data=None,  # Ne pas retourner les données dans l'API pour éviter les gros payloads
            audio_url=audio_url,
            text_processed=processed_text,
            voice_id=request.voice_id or "default",
            metadata=metadata,
            processing_time_ms=processing_time
        )
        
        # Métriques
        log_metrics.log_tts_call()
        log_metrics.log_request(processing_time)
        
        logger.info(f"✅ TTS: Audio généré en {processing_time:.0f}ms ({metadata.size_bytes} bytes)")
        
        return response
        
    except ElevenLabsError as e:
        logger.error(f"❌ Erreur ElevenLabs: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur TTS inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la synthèse vocale"
        )


# ═══════════════════════════════════════════════════════════════
# 🎤 ENDPOINT DE RECONNAISSANCE VOCALE (STT)
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/stt",
    summary="Reconnaissance vocale",
    description="Convertit un fichier audio en texte avec Whisper",
    response_model=STTResponse,
    responses={
        200: {"description": "Transcription réussie"},
        413: {"description": "Fichier audio trop volumineux"},
        415: {"description": "Format audio non supporté"},
        422: {"description": "Audio trop court ou de mauvaise qualité"},
        503: {"description": "Service STT indisponible"}
    }
)
async def speech_to_text(
    audio_file: UploadFile = File(..., description="Fichier audio à transcrire"),
    language: str = Form(default="fr", description="Code langue (ex: fr, en)"),
    model: str = Form(default="base", description="Modèle Whisper"),
    prompt: Optional[str] = Form(default=None, description="Prompt pour guider la transcription"),
    stt_service = Depends(get_stt_service_async)
) -> STTResponse:
    """
    Transcrit un fichier audio en texte.
    
    Utilise OpenAI Whisper pour une transcription précise,
    particulièrement adaptée aux voix d'apprenants.
    
    Args:
        audio_file: Fichier audio à transcrire
        language: Langue de l'audio (fr, en, etc.)
        model: Modèle Whisper à utiliser
        prompt: Prompt optionnel pour améliorer la transcription
        stt_service: Service STT injecté
        
    Returns:
        STTResponse: Transcription avec métadonnées
        
    Raises:
        HTTPException: Si fichier invalide ou service indisponible
    """
    
    if stt_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de reconnaissance vocale indisponible"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"🎤 STT: Transcription {audio_file.filename} ({language})")
        
        # Validation du fichier audio
        await validate_audio_file(
            content_type=audio_file.content_type,
            content_length=audio_file.size if hasattr(audio_file, 'size') else 0
        )
        
        # Lecture du fichier
        audio_content = await audio_file.read()
        
        # Vérification de la taille minimale
        if len(audio_content) < 1000:  # Moins de 1KB
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Fichier audio trop court ou corrompu"
            )
        
        # Création d'un fichier temporaire pour Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Transcription avec Whisper
            # transcription_result = await stt_service.transcribe(
            #     audio_path=temp_file_path,
            #     language=language,
            #     model=model,
            #     prompt=prompt
            # )
            
            # Simulation de transcription
            transcription_result = simulate_transcription(audio_content, language)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Métadonnées audio d'entrée
            audio_metadata = AudioMetadata(
                duration_seconds=estimate_audio_duration_from_size(len(audio_content)),
                format=audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'unknown',
                size_bytes=len(audio_content),
                sample_rate=None,  # Non détectable sans analyse approfondie
                bitrate=None
            )
            
            response = STTResponse(
                success=True,
                message="Transcription réussie",
                transcription=transcription_result["text"],
                confidence=transcription_result.get("confidence", 0.95),
                language_detected=transcription_result.get("language", language),
                segments=transcription_result.get("segments"),
                audio_metadata=audio_metadata,
                processing_time_ms=processing_time,
                model_used=model
            )
            
            # Métriques
            log_metrics.log_stt_call()
            log_metrics.log_request(processing_time)
            
            logger.info(f"✅ STT: '{transcription_result['text'][:50]}...' en {processing_time:.0f}ms")
            
            return response
            
        finally:
            # Nettoyage du fichier temporaire
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
        
    except HTTPException:
        # Re-lever les HTTPException
        raise
    except Exception as e:
        logger.error(f"❌ Erreur STT inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la transcription audio"
        )


# ═══════════════════════════════════════════════════════════════
# 🎭 ENDPOINT LISTE DES VOIX DISPONIBLES
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/voices",
    summary="Voix disponibles",
    description="Liste des voix disponibles pour la synthèse vocale",
    response_model=dict
)
async def list_available_voices(
    tts_service = Depends(get_tts_service_async)
):
    """
    Retourne la liste des voix disponibles pour TTS.
    
    Inclut les informations sur chaque voix : nom, langue,
    genre, et caractéristiques adaptées à l'éducation.
    
    Returns:
        dict: Liste des voix avec leurs caractéristiques
        
    Raises:
        HTTPException: Si service TTS indisponible
    """
    
    try:
        logger.debug("🎭 Récupération des voix disponibles")
        
        if tts_service is None:
            # Liste de voix par défaut si service non configuré
            default_voices = get_default_voices()
            return {
                "success": True,
                "message": "Voix par défaut (service TTS non configuré)",
                "voices": default_voices,
                "total_voices": len(default_voices)
            }
        
        # Récupération des voix ElevenLabs
        # voices = await tts_service.get_available_voices()
        
        # Simulation de voix ElevenLabs
        voices = simulate_elevenlabs_voices()
        
        # Filtrage des voix adaptées à l'éducation
        educational_voices = [
            voice for voice in voices
            if voice.get("suitable_for_education", True)
        ]
        
        return {
            "success": True,
            "message": f"{len(educational_voices)} voix disponibles",
            "voices": educational_voices,
            "total_voices": len(educational_voices),
            "recommended_voice": get_recommended_voice(educational_voices),
            "categories": {
                "french_female": [v for v in educational_voices if v["language"] == "fr" and v["gender"] == "female"],
                "french_male": [v for v in educational_voices if v["language"] == "fr" and v["gender"] == "male"],
                "multilingual": [v for v in educational_voices if v.get("multilingual", False)]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération des voix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de récupérer les voix disponibles"
        )


# ═══════════════════════════════════════════════════════════════
# 🎵 ENDPOINT DE TÉLÉCHARGEMENT D'AUDIO GÉNÉRÉ
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/download/{audio_filename}",
    summary="Télécharge un fichier audio généré",
    description="Télécharge un fichier audio créé par TTS",
    response_class=FileResponse
)
async def download_generated_audio(audio_filename: str):
    """
    Télécharge un fichier audio généré par le service TTS.
    
    Args:
        audio_filename: Nom du fichier audio
        
    Returns:
        FileResponse: Fichier audio à télécharger
        
    Raises:
        HTTPException: Si fichier introuvable
    """
    
    try:
        logger.debug(f"📁 Téléchargement audio: {audio_filename}")
        
        # Vérification de sécurité du nom de fichier
        if not audio_filename.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nom de fichier invalide"
            )
        
        # Chemin vers le fichier
        from pathlib import Path
        audio_path = Path("temp/audio") / audio_filename
        
        # Vérification de l'existence
        if not audio_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fichier audio introuvable ou expiré"
            )
        
        # Retour du fichier
        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename=audio_filename,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename={audio_filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors du téléchargement"
        )


# ═══════════════════════════════════════════════════════════════
# 🌊 ENDPOINT DE STREAMING AUDIO
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/stream/{audio_id}",
    summary="Stream audio",
    description="Diffuse un fichier audio en streaming",
    response_class=StreamingResponse
)
async def stream_audio(audio_id: str):
    """
    Diffuse un fichier audio en streaming.
    
    Permet l'écoute immédiate sans téléchargement complet,
    idéal pour l'expérience utilisateur du chatbot vocal.
    
    Args:
        audio_id: Identifiant du fichier audio
        
    Returns:
        StreamingResponse: Stream audio
        
    Raises:
        HTTPException: Si fichier introuvable
    """
    
    try:
        logger.debug(f"🌊 Streaming audio: {audio_id}")
        
        # Vérification de l'ID
        if not audio_id.replace('_', '').replace('-', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID audio invalide"
            )
        
        # Simulation de streaming
        audio_data = simulate_audio_streaming(audio_id)
        
        def generate_audio():
            """Générateur pour le streaming."""
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=audio_{audio_id}.mp3",
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur streaming audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors du streaming"
        )


# ═══════════════════════════════════════════════════════════════
# 🛠️ FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def prepare_text_for_tts(text: str) -> str:
    """
    Prépare le texte pour la synthèse vocale.
    
    Args:
        text: Texte brut
        
    Returns:
        str: Texte optimisé pour TTS
    """
    
    # Nettoyage de base
    cleaned = text.strip()
    
    # Remplacement d'abréviations courantes
    replacements = {
        "etc.": "et cetera",
        "ex.": "exemple",
        "c-à-d": "c'est-à-dire",
        "vs": "versus",
        "&": "et",
        "%": "pour cent",
        "€": "euros",
        "$": "dollars"
    }
    
    for abbrev, full in replacements.items():
        cleaned = cleaned.replace(abbrev, full)
    
    # Ajout de pauses pour améliorer la lisibilité
    import re
    
    # Pauses après les points
    cleaned = re.sub(r'\.(\s+)([A-Z])', r'.\1\2', cleaned)
    
    # Pauses après les virgules importantes
    cleaned = re.sub(r',(\s+)(mais|donc|alors|ensuite)', r',\1\2', cleaned)
    
    return cleaned


def estimate_audio_duration(text: str, speed: float = 1.0) -> float:
    """
    Estime la durée audio basée sur le texte.
    
    Args:
        text: Texte à synthétiser
        speed: Vitesse de lecture
        
    Returns:
        float: Durée estimée en secondes
    """
    
    # Estimation: ~150 mots par minute en français
    # Ajusté pour la vitesse de lecture éducative (plus lent)
    words = len(text.split())
    base_wpm = 120  # Mots par minute pour éducation
    
    duration_minutes = words / base_wpm
    duration_seconds = duration_minutes * 60
    
    # Ajustement selon la vitesse
    return duration_seconds / speed


def estimate_audio_duration_from_size(size_bytes: int) -> float:
    """
    Estime la durée audio basée sur la taille du fichier.
    
    Args:
        size_bytes: Taille en bytes
        
    Returns:
        float: Durée estimée en secondes
    """
    
    # Estimation grossière : 1 MB ≈ 60 secondes pour du MP3 128kbps
    mb_size = size_bytes / (1024 * 1024)
    return mb_size * 60


def simulate_audio_generation(text: str, format_type) -> bytes:
    """
    Simule la génération audio pour les tests.
    
    Args:
        text: Texte à synthétiser
        format_type: Format audio
        
    Returns:
        bytes: Données audio simulées
    """
    
    # Simulation : taille proportionnelle au texte
    estimated_size = len(text) * 100  # 100 bytes par caractère
    return b"SIMULATED_AUDIO_DATA" * (estimated_size // 20)


def simulate_transcription(audio_content: bytes, language: str) -> dict:
    """
    Simule la transcription pour les tests.
    
    Args:
        audio_content: Données audio
        language: Langue attendue
        
    Returns:
        dict: Résultat de transcription simulé
    """
    
    # Simulation basée sur la taille
    size_mb = len(audio_content) / (1024 * 1024)
    
    if size_mb < 0.1:
        transcription = "Bonjour"
    elif size_mb < 0.5:
        transcription = "Bonjour, pouvez-vous m'expliquer comment faire une addition ?"
    else:
        transcription = "Bonjour, je suis un apprenant et j'aimerais que vous m'expliquiez comment faire une addition simple avec des exemples concrets."
    
    return {
        "text": transcription,
        "confidence": 0.95,
        "language": language,
        "segments": [
            {
                "start": 0.0,
                "end": len(transcription) * 0.1,  # ~10 caractères par seconde
                "text": transcription
            }
        ]
    }


def get_default_voices() -> list:
    """
    Retourne des voix par défaut si ElevenLabs n'est pas configuré.
    
    Returns:
        list: Liste de voix par défaut
    """
    
    return [
        {
            "voice_id": "default_fr_female",
            "name": "Marie (Française)",
            "language": "fr",
            "gender": "female",
            "age": "adult",
            "description": "Voix féminine française claire et pédagogique",
            "suitable_for_education": True,
            "recommended": True
        },
        {
            "voice_id": "default_fr_male",
            "name": "Pierre (Français)",
            "language": "fr",
            "gender": "male",
            "age": "adult",
            "description": "Voix masculine française patiente et encourageante",
            "suitable_for_education": True,
            "recommended": False
        }
    ]


def simulate_elevenlabs_voices() -> list:
    """
    Simule la réponse de l'API ElevenLabs pour les voix.
    
    Returns:
        list: Liste de voix simulées
    """
    
    return [
        {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "name": "Rachel",
            "language": "en",
            "gender": "female", 
            "age": "young_adult",
            "description": "Voix anglaise claire et expressive",
            "suitable_for_education": True,
            "multilingual": True,
            "recommended": False
        },
        {
            "voice_id": "AZnzlk1XvdvUeBnXmlld",
            "name": "Domi",
            "language": "en",
            "gender": "female",
            "age": "adult",
            "description": "Voix anglaise professionnelle et rassurante",
            "suitable_for_education": True,
            "multilingual": False,
            "recommended": True
        },
        {
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "name": "Sarah",
            "language": "fr",
            "gender": "female",
            "age": "adult",
            "description": "Voix française éducative spécialement adaptée à l'apprentissage",
            "suitable_for_education": True,
            "multilingual": False,
            "recommended": True
        }
    ]


def get_recommended_voice(voices: list) -> dict:
    """
    Retourne la voix recommandée pour l'éducation.
    
    Args:
        voices: Liste des voix disponibles
        
    Returns:
        dict: Voix recommandée
    """
    
    # Priorité aux voix françaises éducatives
    french_educational = [
        v for v in voices 
        if v["language"] == "fr" and v.get("suitable_for_education", False)
    ]
    
    if french_educational:
        return french_educational[0]
    
    # Sinon, première voix recommandée
    recommended = [v for v in voices if v.get("recommended", False)]
    if recommended:
        return recommended[0]
    
    # Sinon, première voix de la liste
    return voices[0] if voices else {}


def simulate_file_exists(audio_id: str) -> bool:
    """
    Simule l'existence d'un fichier audio.
    
    Args:
        audio_id: ID du fichier
        
    Returns:
        bool: True si le fichier "existe"
    """
    
    # Simulation : les IDs avec des chiffres pairs "existent"
    try:
        return sum(int(c) for c in audio_id if c.isdigit()) % 2 == 0
    except:
        return False


def simulate_audio_streaming(audio_id: str) -> bytes:
    """
    Simule les données pour le streaming audio.
    
    Args:
        audio_id: ID du fichier
        
    Returns:
        bytes: Données audio simulées
    """
    
    # Génération de données MP3 simulées
    base_data = b"SIMULATED_MP3_HEADER" + b"\x00" * 1000
    return base_data * (len(audio_id) + 1)  # Taille variable selon l'ID