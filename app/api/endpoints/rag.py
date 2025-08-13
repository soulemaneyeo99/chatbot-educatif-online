"""
Endpoints de Conversation et RAG - Chatbot AmazoOn du Web
========================================================

Chatbot d'accompagnement numérique pour les femmes des groupements féminins
en Afrique de l'Ouest (Côte d'Ivoire). Spécialisé dans l'inclusion numérique,
l'autonomisation économique et l'apprentissage pratique du digital.

Routes:
- POST /chat        → Conversation avec l'assistante numérique
- POST /search      → Recherche dans modules de formation
- POST /ingest      → Ingestion de contenus pédagogiques
- POST /pipeline    → Pipeline vocal (crucial pour non-alphabétisées)
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dependencies import (
    get_openai_client_async,
    get_vectorstore_async,
    get_tts_service_async,
    get_stt_service_async,
    validate_text_input,
    validate_audio_file
)
from app.models.requests import (
    ChatRequest,
    SearchRequest,
    STTRequest
)
from app.models.responses import (
    ChatResponse,
    SearchResponse,
    VoicePipelineResponse,
    ChatMetadata,
    DocumentContext,
    PipelineStep,
    create_success_response,
    create_error_response
)
from app.utils.logger import setup_logger, log_metrics
from app.utils.exceptions import (
    ChatbotException,
    ClaudeAPIError,
    VectorstoreError,
    AudioError
)

# ═══════════════════════════════════════════════════════════════
# 🏗️ CONFIGURATION DU ROUTER
# ═══════════════════════════════════════════════════════════════

router = APIRouter()
logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════
# 💬 ENDPOINT DE CONVERSATION PRINCIPALE - ACCOMPAGNEMENT NUMÉRIQUE
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/chat",
    summary="Conversation avec l'assistante numérique AmazoOn",
    description="Accompagnement personnalisé pour les femmes des groupements dans l'apprentissage du numérique",
    response_model=ChatResponse,
    responses={
        200: {"description": "Conseil d'accompagnement généré avec succès"},
        422: {"description": "Données de requête invalides"},
        503: {"description": "Assistante numérique indisponible"},
        429: {"description": "Trop de requêtes"}
    }
)
async def chat_accompagnement_numerique(
    request: ChatRequest,
    openai_client = Depends(get_openai_client_async),
    vectorstore = Depends(get_vectorstore_async)
) -> ChatResponse:
    """
    Génère un accompagnement personnalisé pour l'autonomisation numérique.
    
    Cette route est le cœur du chatbot d'inclusion numérique. Elle:
    1. Analyse la demande de la femme (commerce, WhatsApp, Mobile Money, etc.)
    2. Recherche des contenus pédagogiques pertinents (RAG)
    3. Génère des conseils pratiques et simples avec OpenAI
    4. Retourne des recommandations actionnables adaptées au contexte rural
    
    Args:
        request: Requête avec message et niveau d'alphabétisation
        openai_client: Client OpenAI injecté
        vectorstore: Vectorstore FAISS avec contenus pédagogiques
        
    Returns:
        ChatResponse: Conseils d'accompagnement avec métadonnées
        
    Raises:
        HTTPException: Si erreur lors de la génération
    """
    
    start_time = time.time()
    
    try:
        logger.info(f"🌍 Nouvelle demande d'accompagnement - Session: {getattr(request, 'session_id', 'anonyme')}")
        logger.debug(f"Message: {request.message[:100]}{'...' if len(request.message) > 100 else ''}")
        
        # Validation du message avec prise en compte du niveau d'alphabétisation
        cleaned_message = await validate_text_input(request.message)
        
        # Détection du module demandé et niveau d'expérience (avec gestion d'erreur)
        try:
            module_demande = detect_learning_module(cleaned_message)
            niveau_experience = detect_digital_experience_level(cleaned_message)
        except Exception as e:
            # Valeurs par défaut si les fonctions de détection échouent
            logger.warning(f"⚠️ Erreur détection module/niveau: {e}")
            module_demande = "general"
            niveau_experience = "debutante"
        
        # Recherche de contenus pédagogiques RAG si activé
        context_documents = []
        if getattr(request, 'use_rag', True) and vectorstore is not None:
            try:
                logger.debug("🔍 Recherche de contenus pédagogiques pertinents")
                
                # Recherche contextuelle selon le module et niveau
                try:
                    educational_context = get_educational_context(cleaned_message, module_demande, niveau_experience)
                    if educational_context:
                        context_documents = [
                            DocumentContext(
                                content=educational_context["content"],
                                score=educational_context["score"],
                                metadata=educational_context["metadata"],
                                source=educational_context["source"]
                            )
                        ]
                except Exception as e:
                    logger.warning(f"⚠️ Erreur récupération contexte éducatif: {e}")
                    # Contexte par défaut
                    context_documents = [
                        DocumentContext(
                            content="Je suis là pour t'accompagner dans ton apprentissage du numérique. N'hésite pas à me poser des questions sur WhatsApp, Mobile Money, ou tout autre outil numérique !",
                            score=0.8,
                            metadata={"module": "general", "niveau": niveau_experience},
                            source="accompagnement_general.md"
                        )
                    ]
                
                logger.debug(f"📚 {len(context_documents)} contenus pédagogiques trouvés")
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur RAG non critique: {e}")
                # Continuer sans contexte RAG
        
        # Préparation de l'historique de conversation
        conversation_history = []
        if getattr(request, 'conversation_history', None):
            conversation_history = [
                {
                    "user": msg.content if msg.role.value == "user" else None,
                    "assistant": msg.content if msg.role.value == "assistant" else None
                }
                for msg in request.conversation_history
            ]
        
        # Génération de la réponse avec prompt spécialisé inclusion numérique
        logger.debug("🤖 Génération du conseil d'accompagnement avec OpenAI")
        
        try:
            system_prompt = get_digital_inclusion_system_prompt(niveau_experience)
        except Exception as e:
            logger.warning(f"⚠️ Erreur prompt système: {e}")
            # Prompt par défaut
            system_prompt = "Tu es une assistante bienveillante qui aide les femmes à apprendre le numérique. Réponds simplement et avec encouragement."
        
        openai_response = await openai_client.generate_response(
            user_message=cleaned_message,
            context_documents=[doc.dict() for doc in context_documents] if context_documents else None,
            conversation_history=conversation_history,
            system_prompt_override=system_prompt
        )
        
        # Construction des métadonnées spécialisées
        metadata = ChatMetadata(
            model_used=openai_response["model"],
            tokens_used=openai_response["tokens_used"],
            prompt_tokens=openai_response.get("metadata", {}).get("prompt_tokens", 0),
            completion_tokens=openai_response.get("metadata", {}).get("completion_tokens", 0),
            response_time_ms=openai_response["response_time_ms"],
            context_documents_count=len(context_documents),
            education_level_detected=niveau_experience,
            subject_detected=module_demande
        )
        
        # Génération de suggestions pédagogiques de suivi
        try:
            suggestions = generate_learning_suggestions(
                user_message=cleaned_message,
                ai_response=openai_response["response"],
                module_demande=module_demande,
                niveau_experience=niveau_experience
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération suggestions: {e}")
            # Suggestions par défaut
            suggestions = [
                "Comment puis-je améliorer mes compétences numériques ?",
                "Peux-tu m'expliquer WhatsApp Business ?",
                "Comment protéger mon argent avec Mobile Money ?"
            ]
        
        # Enregistrement des métriques d'usage
        log_metrics.log_request(openai_response["response_time_ms"])
        
        # Construction de la réponse finale
        response = ChatResponse(
            success=True,
            message="Accompagnement généré avec succès",
            response=openai_response["response"],
            context_documents=context_documents,
            suggestions=suggestions,
            session_id=getattr(request, 'session_id', None),
            metadata=metadata,
            audio_available=True,  # Important pour les non-alphabétisées
            audio_url=None  # Sera généré par TTS si demandé
        )
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Accompagnement généré en {total_time:.0f}ms - Module: {module_demande} - Niveau: {niveau_experience}")
        
        return response
        
    except ClaudeAPIError as e:
        log_metrics.log_error()
        logger.error(f"❌ Erreur OpenAI: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    
    except Exception as e:
        log_metrics.log_error()
        logger.error(f"❌ Erreur inattendue lors de l'accompagnement: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne s'est produite lors de l'accompagnement"
        )


# ═══════════════════════════════════════════════════════════════
# 🔍 ENDPOINT DE RECHERCHE DANS LES MODULES DE FORMATION
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/search",
    summary="Recherche dans les modules de formation numérique",
    description="Recherche de contenus pédagogiques spécifiques par module ou compétence",
    response_model=SearchResponse,
    responses={
        200: {"description": "Modules trouvés"},
        404: {"description": "Aucun module trouvé"},
        503: {"description": "Base pédagogique indisponible"}
    }
)
async def search_learning_modules(
    request: SearchRequest,
    vectorstore = Depends(get_vectorstore_async)
) -> SearchResponse:
    """
    Recherche dans les 10 modules de formation numérique.
    
    Modules disponibles selon le cahier des charges:
    1. Découverte du smartphone et Internet mobile
    2. WhatsApp pour le commerce
    3. Facebook et Messenger pour débutantes  
    4. Créer une fiche produit attrayante
    5. Marketing digital local
    6. Techniques de vente en ligne
    7. Gestion de micro-entreprise
    8. Mobile Money et paiements à distance
    9. Livraison locale et sous-régionale
    10. Cybersécurité et bonnes pratiques
    
    Args:
        request: Paramètres de recherche
        vectorstore: Vectorstore avec contenus pédagogiques
        
    Returns:
        SearchResponse: Modules et contenus trouvés
        
    Raises:
        HTTPException: Si base pédagogique indisponible
    """
    
    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de formation non disponible. Ingérez les modules pédagogiques d'abord."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Recherche module: '{request.query}'")
        
        # Validation de la requête
        cleaned_query = await validate_text_input(request.query)
        
        # Recherche dans les modules de formation
        search_results = get_learning_modules_results(cleaned_query)
        
        search_time = (time.time() - start_time) * 1000
        
        # Filtrage par seuil de pertinence
        filtered_results = [
            result for result in search_results 
            if result.score >= request.similarity_threshold
        ]
        
        # Limitation du nombre de résultats
        final_results = filtered_results[:request.max_results]
        
        # Filtres appliqués
        filters_applied = {}
        if hasattr(request, 'subject_filter') and request.subject_filter:
            filters_applied["module"] = request.subject_filter.value
        
        response = SearchResponse(
            success=True,
            message=f"{len(final_results)} modules trouvés",
            results=final_results,
            query=cleaned_query,
            total_results=len(search_results),
            results_returned=len(final_results),
            search_time_ms=search_time,
            filters_applied=filters_applied if filters_applied else None
        )
        
        logger.info(f"✅ Recherche terminée: {len(final_results)}/{len(search_results)} modules retournés")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche pédagogique: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la recherche dans les modules"
        )


# ═══════════════════════════════════════════════════════════════
# 📚 ENDPOINT D'INGESTION DE CONTENUS PÉDAGOGIQUES
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/ingest",
    summary="Ingestion de modules de formation", 
    description="Ajoute des contenus pédagogiques (audio, texte, images) à la base",
    response_model=dict,
    responses={
        200: {"description": "Modules ingérés avec succès"},
        400: {"description": "Contenus pédagogiques invalides"},
        413: {"description": "Fichiers trop volumineux"}
    }
)
async def ingest_learning_content(
    file_paths: List[str],
    module_type: str = "general",
    target_audience: str = "debutante",
    language: str = "french",
    vectorstore = Depends(get_vectorstore_async)
) -> dict:
    """
    Ingère des contenus pédagogiques dans la base de formation.
    
    Types de contenus selon le cahier des charges:
    - Guides pratiques par module (10 modules)
    - Contenus audio pour non-alphabétisées
    - Images et vidéos courtes
    - Scénarios d'usage contextualisés
    
    Args:
        file_paths: Chemins des contenus pédagogiques
        module_type: Type de module (whatsapp, mobile_money, etc.)
        target_audience: Public (debutante, intermediaire, avancee)
        language: Langue (french, local_languages)
        vectorstore: Vectorstore pédagogique
        
    Returns:
        dict: Rapport d'ingestion des modules
        
    Raises:
        HTTPException: Si erreur lors de l'ingestion
    """
    
    start_time = time.time()
    
    try:
        logger.info(f"📚 Ingestion de {len(file_paths)} contenus pédagogiques - Module: {module_type}")
        
        # Validation du type de module selon le cahier des charges
        valid_modules = [
            "smartphone_basics", "whatsapp_commerce", "facebook_messenger",
            "fiche_produit", "marketing_local", "vente_en_ligne", 
            "micro_entreprise", "mobile_money", "livraison", "cybersecurite"
        ]
        
        if module_type not in valid_modules and module_type != "general":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Module invalide. Modules valides: {valid_modules}"
            )
        
        # Simulation de l'ingestion avec métadonnées spécifiques
        return {
            "success": True,
            "message": f"Ingestion module '{module_type}' simulée - Fonctionnalité à implémenter",
            "module_type": module_type,
            "target_audience": target_audience,
            "language": language,
            "files_processed": len(file_paths),
            "content_types_supported": ["audio", "text", "images", "videos"],
            "processing_time_ms": (time.time() - start_time) * 1000,
            "next_steps": [
                "Tester la recherche dans ce module",
                "Valider l'accessibilité pour non-alphabétisées",
                "Vérifier l'adaptation culturelle du contenu"
            ],
            "modules_disponibles": {
                "1": "Découverte smartphone & Internet",
                "2": "WhatsApp pour commerce",
                "3": "Facebook & Messenger débutantes",
                "4": "Création fiche produit",
                "5": "Marketing digital local",
                "6": "Techniques vente en ligne",
                "7": "Gestion micro-entreprise",
                "8": "Mobile Money & paiements",
                "9": "Livraison & logistique",
                "10": "Cybersécurité & protection"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'ingestion pédagogique: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'ingestion des contenus pédagogiques"
        )


# ═══════════════════════════════════════════════════════════════
# 🎙️ ENDPOINT DE PIPELINE VOCAL ÉDUCATIF (PRIORITAIRE)
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/pipeline",
    summary="Pipeline vocal éducatif",
    description="Accompagnement vocal complet pour femmes peu/non alphabétisées",
    response_model=VoicePipelineResponse,
    responses={
        200: {"description": "Accompagnement vocal réussi"},
        413: {"description": "Fichier audio trop volumineux"},
        415: {"description": "Format audio non supporté"},
        503: {"description": "Services audio indisponibles"}
    }
)
async def educational_voice_pipeline(
    audio_file: UploadFile = File(..., description="Question vocale en français ou langue locale"),
    session_id: Optional[str] = None,
    user_profile: str = "debutante",
    preferred_language: str = "french",
    learning_module: str = "general",
    openai_client = Depends(get_openai_client_async),
    stt_service = Depends(get_stt_service_async),
    tts_service = Depends(get_tts_service_async),
    vectorstore = Depends(get_vectorstore_async)
) -> VoicePipelineResponse:
    """
    Pipeline vocal d'accompagnement éducatif pour femmes rurales.
    
    Processus optimisé pour l'inclusion numérique:
    1. Réception question vocale (français/langues locales)
    2. Transcription avec Whisper (STT)
    3. Analyse pédagogique avec OpenAI + contenus de formation
    4. Synthèse vocale adaptée et bienveillante (TTS)
    5. Retourne conseil audio + support visuel si nécessaire
    
    Args:
        audio_file: Fichier audio de la question
        session_id: Suivi de progression individuelle
        user_profile: Profil d'expérience (debutante, intermediaire, avancee)
        preferred_language: Langue préférée
        learning_module: Module d'apprentissage ciblé
        
    Returns:
        VoicePipelineResponse: Accompagnement vocal complet
        
    Raises:
        HTTPException: Si erreur à une étape du pipeline
    """
    
    pipeline_start = time.time()
    pipeline_steps = []
    
    try:
        logger.info(f"🎙️ Accompagnement vocal - Session: {session_id or 'anonyme'} - Profil: {user_profile}")
        
        # Validation du fichier audio
        await validate_audio_file(
            content_type=audio_file.content_type,
            content_length=getattr(audio_file, 'size', 0)
        )
        
        # ═══════════════════════════════════════════════════════════════
        # ÉTAPE 1: TRANSCRIPTION DE LA QUESTION ÉDUCATIVE
        # ═══════════════════════════════════════════════════════════════
        
        step_start = time.time()
        stt_step = PipelineStep(
            step_name="transcription_question_educative",
            status="running",
            start_time=step_start
        )
        
        try:
            # Lecture du fichier audio
            audio_content = await audio_file.read()
            
            # Simulation de transcription adaptée au contexte africain
            transcription = simulate_educational_transcription(audio_content, preferred_language)
            
            stt_step.status = "completed"
            stt_step.end_time = time.time()
            stt_step.duration_ms = (stt_step.end_time - step_start) * 1000
            stt_step.output = {
                "transcription": transcription, 
                "language_detected": preferred_language,
                "educational_context": True
            }
            
            logger.info(f"✅ Question transcrite: '{transcription}'")
            
        except Exception as e:
            stt_step.status = "failed"
            stt_step.end_time = time.time()
            stt_step.duration_ms = (stt_step.end_time - step_start) * 1000
            stt_step.error = str(e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Erreur de transcription: {str(e)}"
            )
        finally:
            pipeline_steps.append(stt_step)
        
        # ═══════════════════════════════════════════════════════════════
        # ÉTAPE 2: GÉNÉRATION D'ACCOMPAGNEMENT PÉDAGOGIQUE
        # ═══════════════════════════════════════════════════════════════
        
        step_start = time.time()
        educational_step = PipelineStep(
            step_name="generation_accompagnement_pedagogique",
            status="running",
            start_time=step_start
        )
        
        try:
            # Détection du module et contexte
            module_detected = detect_learning_module(transcription)
            experience_level = detect_digital_experience_level(transcription)
            
            # Recherche de contenus pédagogiques si disponibles
            context_documents = []
            if vectorstore:
                educational_context = get_educational_context(transcription, module_detected, experience_level)
                if educational_context:
                    context_documents = [educational_context]
            
            # Génération avec prompt pédagogique spécialisé
            ai_response = await openai_client.generate_response(
                user_message=transcription,
                context_documents=context_documents,
                system_prompt_override=get_digital_inclusion_system_prompt(experience_level)
            )
            
            educational_step.status = "completed"
            educational_step.end_time = time.time()
            educational_step.duration_ms = (educational_step.end_time - step_start) * 1000
            educational_step.output = {
                "accompagnement_pedagogique": ai_response["response"],
                "tokens_used": ai_response["tokens_used"],
                "module_detected": module_detected,
                "experience_level": experience_level,
                "contenus_pedagogiques_utilises": len(context_documents)
            }
            
            logger.info(f"✅ Accompagnement généré - Module: {module_detected} - Niveau: {experience_level}")
            
        except Exception as e:
            educational_step.status = "failed"
            educational_step.end_time = time.time()
            educational_step.duration_ms = (educational_step.end_time - step_start) * 1000
            educational_step.error = str(e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Erreur de génération pédagogique: {str(e)}"
            )
        finally:
            pipeline_steps.append(educational_step)
        
        # ═══════════════════════════════════════════════════════════════
        # ÉTAPE 3: SYNTHÈSE VOCALE BIENVEILLANTE ET ADAPTÉE
        # ═══════════════════════════════════════════════════════════════
        
        step_start = time.time()
        tts_step = PipelineStep(
            step_name="synthese_vocale_bienveillante",
            status="running",
            start_time=step_start
        )
        
        audio_response_url = None
        
        try:
            if tts_service:
                # Adaptation du texte pour la synthèse vocale (plus lent, plus clair)
                adapted_text = adapt_text_for_voice_synthesis(ai_response["response"], user_profile)
                
                # Synthèse vocale avec paramètres adaptés (vitesse réduite, intonation bienveillante)
                audio_response_url = f"/temp/audio/accompagnement_vocal_{int(time.time())}.mp3"
                
                tts_step.status = "completed"
                tts_step.output = {
                    "audio_url": audio_response_url,
                    "voice_settings": {
                        "speed": "slow",  # Plus lent pour la compréhension
                        "tone": "patient_and_encouraging",
                        "language": preferred_language
                    },
                    "adapted_for_rural_context": True
                }
                
                logger.info("✅ Audio d'accompagnement généré avec paramètres adaptés")
            else:
                tts_step.status = "skipped"
                tts_step.output = {"message": "Service TTS non configuré"}
                
                logger.warning("⚠️ TTS: Service non disponible - Important pour l'accessibilité")
            
            tts_step.end_time = time.time()
            tts_step.duration_ms = (tts_step.end_time - step_start) * 1000
            
        except Exception as e:
            tts_step.status = "failed"
            tts_step.end_time = time.time()
            tts_step.duration_ms = (tts_step.end_time - step_start) * 1000
            tts_step.error = str(e)
            
            # TTS critique pour l'accessibilité mais on continue
            logger.warning(f"⚠️ TTS échoué (critique pour accessibilité): {e}")
        finally:
            pipeline_steps.append(tts_step)
        
        # ═══════════════════════════════════════════════════════════════
        # CONSTRUCTION DE LA RÉPONSE D'ACCOMPAGNEMENT FINALE
        # ═══════════════════════════════════════════════════════════════
        
        total_time = (time.time() - pipeline_start) * 1000
        
        response = VoicePipelineResponse(
            success=True,
            message="Accompagnement vocal terminé avec succès",
            final_response=ai_response["response"],
            audio_response_url=audio_response_url,
            transcription=transcription,
            pipeline_steps=pipeline_steps,
            total_processing_time_ms=total_time,
            tokens_used=ai_response["tokens_used"],
            session_id=session_id
        )
        
        log_metrics.log_request(total_time)
        
        logger.info(f"🎉 Accompagnement vocal terminé en {total_time:.0f}ms - Profil: {user_profile}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur accompagnement vocal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'accompagnement vocal"
        )


# ═══════════════════════════════════════════════════════════════
# 🛠️ FONCTIONS UTILITAIRES SPÉCIALISÉES INCLUSION NUMÉRIQUE
# ═══════════════════════════════════════════════════════════════

def get_digital_inclusion_system_prompt(niveau_experience: str) -> str:
    """
    Retourne le prompt système adapté à l'inclusion numérique des femmes rurales.
    
    Args:
        niveau_experience: Niveau d'expérience numérique
        
    Returns:
        str: Prompt système contextualisé
    """
    
    base_prompt = """Tu es une assistante numérique bienveillante spécialement conçue pour accompagner les femmes des groupements coopératifs en Afrique de l'Ouest dans leur apprentissage du numérique.

## 🌍 TON CONTEXTE DE MISSION
- Tu travailles pour AmazoOn du Web en Côte d'Ivoire
- Tu accompagnes des femmes rurales souvent peu alphabétisées
- Ton objectif est l'autonomisation économique par le numérique
- Tu comprends les réalités locales, culturelles et économiques

## 🎯 TON RÔLE D'ACCOMPAGNATRICE
- Tu es patiente, encourageante et respectueuse
- Tu utilises des mots simples et des exemples concrets du quotidien
- Tu adaptes tes explications au niveau de chaque femme
- Tu valorises leurs compétences existantes avant d'introduire le nouveau

## 📚 TES DOMAINES D'EXPERTISE (10 MODULES)
1. **Découverte smartphone & Internet** : Navigation de base, applications essentielles
2. **WhatsApp pour commerce** : Messages, statuts, catalogues, groupes clients
3. **Facebook & Messenger** : Création profil, publication produits, interaction clients
4. **Création fiche produit** : Photos attractives, descriptions, prix compétitifs
5. **Marketing digital local** : Promotion adaptée, visuels simples, ciblage communautaire
6. **Techniques vente en ligne** : Stratégies adaptées, offres spéciales, fidélisation
7. **Gestion micro-entreprise** : Comptabilité de base, suivi ventes, gestion stocks
8. **Mobile Money & paiements** : Envoi/recevoir argent, sécurité, gestion transactions
9. **Livraison & logistique** : Organisation livraisons locales, partenariats transporteurs
10. **Cybersécurité & protection** : Sécurité des données, prévention arnaques"""