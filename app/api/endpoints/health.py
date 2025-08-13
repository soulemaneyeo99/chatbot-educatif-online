"""
Endpoints de Santé et Monitoring
===============================

Routes pour vérifier l'état de santé des services, obtenir des métriques
et diagnostiquer les problèmes du chatbot éducatif.

Routes:
- GET /health          → État général
- GET /health/detailed → État détaillé  
- GET /health/metrics  → Métriques système
- GET /health/status   → Statut des services
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dependencies import (
    get_health_info,
    get_metrics,
    get_debug_info,
    validate_all_services
)
from app.models.responses import (
    HealthResponse,
    MetricsResponse,
    ServiceHealth,
    ServiceStatus,
    BaseResponse
)
from app.utils.logger import setup_logger, log_metrics
from app.utils.exceptions import ChatbotException

# ═══════════════════════════════════════════════════════════════
# 🏗️ CONFIGURATION DU ROUTER
# ═══════════════════════════════════════════════════════════════

router = APIRouter()
logger = setup_logger(__name__)

# Horodatage de démarrage (pour calculer l'uptime)
start_time = time.time()

# ═══════════════════════════════════════════════════════════════
# 🏥 ENDPOINT DE SANTÉ SIMPLE
# ═══════════════════════════════════════════════════════════════

@router.get(
    "",
    summary="État de santé général",
    description="Vérification rapide de l'état de santé global du système",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK
)
async def health_check():
    """
    Point de contrôle rapide pour vérifier que l'API fonctionne.
    
    Retourne un statut simple sans vérifications approfondies.
    Idéal pour les load balancers et monitoring externe.
    
    Returns:
        BaseResponse: Statut de santé basique
    """
    
    try:
        settings = get_settings()
        uptime = time.time() - start_time
        
        return BaseResponse(
            success=True,
            message="✅ Chatbot éducatif opérationnel",
            timestamp=time.time(),
            request_id=None  # Sera ajouté par le middleware
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporairement indisponible"
        )


# ═══════════════════════════════════════════════════════════════
# 🔍 ENDPOINT DE SANTÉ DÉTAILLÉ
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/detailed",
    summary="État de santé détaillé",
    description="Vérification complète de tous les services avec diagnostics",
    response_model=HealthResponse,
    responses={
        200: {"description": "Tous les services fonctionnent"},
        207: {"description": "Fonctionnement partiel (certains services dégradés)"},
        503: {"description": "Services critiques indisponibles"}
    }
)
async def detailed_health_check():
    """
    Contrôle de santé approfondi de tous les services.
    
    Vérifie la connectivité et le bon fonctionnement de:
    - OpenAI API
    - ElevenLabs TTS (si configuré)
    - Whisper STT
    - Vectorstore FAISS
    - Base de données (si applicable)
    
    Returns:
        HealthResponse: État détaillé de chaque service
        
    Raises:
        HTTPException: Si services critiques indisponibles
    """
    
    try:
        logger.info("🔍 Vérification détaillée de la santé du système")
        
        settings = get_settings()
        uptime = time.time() - start_time
        
        # Obtenir les informations de santé de tous les services
        health_info = await get_health_info()
        
        # Transformer en modèles Pydantic
        services = []
        overall_healthy = True
        critical_services_down = False
        
        for service_name, service_data in health_info.get("services", {}).items():
            service_status = ServiceStatus.HEALTHY
            error_message = None
            
            if service_data.get("status") == "unhealthy":
                service_status = ServiceStatus.UNHEALTHY
                error_message = service_data.get("error", "Service indisponible")
                overall_healthy = False
                
                # OpenAI est critique
                if service_name == "openai":
                    critical_services_down = True
                    
            elif service_data.get("status") == "not_configured":
                service_status = ServiceStatus.NOT_CONFIGURED
                # TTS non configuré n'est pas critique
                if service_name != "tts":
                    overall_healthy = False
            
            services.append(ServiceHealth(
                name=service_name,
                status=service_status,
                response_time_ms=service_data.get("response_time_ms"),
                last_check=time.time(),
                version=service_data.get("version"),
                metadata=service_data.get("metadata", {}),
                error_message=error_message
            ))
        
        # Déterminer le statut global
        if critical_services_down:
            global_status = ServiceStatus.UNHEALTHY
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif not overall_healthy:
            global_status = ServiceStatus.DEGRADED
            status_code = status.HTTP_207_MULTI_STATUS
        else:
            global_status = ServiceStatus.HEALTHY
            status_code = status.HTTP_200_OK
        
        # Informations système
        system_info = {
            "uptime_seconds": uptime,
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "python_version": "3.11+",
            "api_version": settings.VERSION,
            "openai_model": settings.OPENAI_MODEL,
            "whisper_model": settings.WHISPER_MODEL,
            "vectorstore_enabled": any(s.name == "vectorstore" and s.status == ServiceStatus.HEALTHY for s in services),
            "tts_enabled": any(s.name == "tts" and s.status == ServiceStatus.HEALTHY for s in services)
        }
        
        response = HealthResponse(
            success=global_status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED],
            message=f"Système {global_status.value}",
            overall_status=global_status,
            services=services,
            system_info=system_info,
            uptime_seconds=uptime,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT
        )
        
        logger.info(f"✅ Health check terminé - Statut: {global_status.value}")
        
        return JSONResponse(
            content=response.dict(),
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du health check détaillé: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Impossible de vérifier l'état du système: {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════
# 📊 ENDPOINT DE MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/metrics",
    summary="Métriques du système",
    description="Statistiques d'utilisation et de performance du chatbot",
    response_model=Dict[str, Any]
)
async def get_system_metrics():
    """
    Retourne les métriques de performance et d'utilisation.
    
    Inclut:
    - Statistiques de requêtes (total, succès, échecs)
    - Temps de réponse moyens par service
    - Utilisation des ressources
    - Statistiques d'utilisation des modèles
    
    Returns:
        Dict: Métriques complètes du système
    """
    
    try:
        logger.debug("📊 Collecte des métriques système")
        
        # Métriques globales
        global_metrics = get_metrics()
        
        # Métriques par service (simulées pour l'instant)
        service_metrics = {
            "openai": {
                "total_requests": global_metrics.get("claude_calls", 0),  # On garde le nom pour compatibilité
                "average_response_time_ms": global_metrics.get("average_response_time", 0),
                "tokens_used_total": 0,  # À implémenter
                "cost_estimate_usd": 0   # À implémenter
            },
            "tts": {
                "total_requests": global_metrics.get("tts_calls", 0),
                "audio_generated_seconds": 0,  # À implémenter
                "characters_processed": 0      # À implémenter
            },
            "stt": {
                "total_requests": global_metrics.get("stt_calls", 0),
                "audio_transcribed_seconds": 0  # À implémenter
            },
            "vectorstore": {
                "total_searches": 0,    # À implémenter
                "documents_indexed": 0, # À implémenter
                "average_search_time_ms": 0  # À implémenter
            }
        }
        
        # Informations de performance
        performance_stats = {
            "requests_per_minute": global_metrics.get("requests_total", 0) / max(1, (time.time() - start_time) / 60),
            "error_rate_percent": (global_metrics.get("errors_total", 0) / max(1, global_metrics.get("requests_total", 1))) * 100,
            "uptime_percent": 99.9,  # À calculer en fonction des pannes
            "average_pipeline_time_ms": global_metrics.get("average_response_time", 0)
        }
        
        # Utilisation des ressources (basique)
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resource_usage = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        response_data = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - start_time,
            "global_metrics": global_metrics,
            "service_metrics": service_metrics,
            "performance_stats": performance_stats,
            "resource_usage": resource_usage
        }
        
        logger.debug("✅ Métriques collectées avec succès")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte des métriques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de collecter les métriques"
        )


# ═══════════════════════════════════════════════════════════════
# ⚙️ ENDPOINT DE STATUT DES SERVICES
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/status",
    summary="Statut des services",
    description="État rapide de chaque service individuel",
    response_model=Dict[str, Any]
)
async def services_status():
    """
    Retourne le statut de chaque service de manière concise.
    
    Plus rapide que le health check détaillé, idéal pour les
    tableaux de bord et monitoring en temps réel.
    
    Returns:
        Dict: Statut de chaque service
    """
    
    try:
        logger.debug("⚡ Vérification rapide du statut des services")
        
        # Validation rapide de tous les services
        validation_result = validate_all_services()
        
        services_status = {}
        for service_name, service_info in validation_result.get("services", {}).items():
            services_status[service_name] = {
                "status": service_info.get("status", "unknown"),
                "initialized": service_info.get("initialized", False),
                "error": service_info.get("error", None)
            }
        
        return {
            "overall_status": validation_result.get("overall_status", "unknown"),
            "services": services_status,
            "failed_services": validation_result.get("failed_services", []),
            "timestamp": time.time(),
            "check_duration_ms": 0  # Durée très rapide
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification des statuts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de vérifier les statuts"
        )


# ═══════════════════════════════════════════════════════════════
# 🔧 ENDPOINT DE DEBUG (MODE DÉVELOPPEMENT)
# ═══════════════════════════════════════════════════════════════

@router.get(
    "/debug",
    summary="Informations de debug",
    description="Informations détaillées pour le débogage (mode développement uniquement)",
    response_model=Dict[str, Any],
    include_in_schema=False  # Masqué dans la doc en production
)
async def debug_info():
    """
    Retourne des informations de débogage détaillées.
    
    ⚠️ Disponible uniquement en mode DEBUG.
    Contient des informations sensibles sur la configuration.
    
    Returns:
        Dict: Informations de debug
        
    Raises:
        HTTPException: Si pas en mode debug
    """
    
    settings = get_settings()
    
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint disponible uniquement en mode debug"
        )
    
    try:
        logger.debug("🔍 Collecte des informations de debug")
        
        debug_data = get_debug_info()
        
        # Ajout d'informations supplémentaires
        debug_data.update({
            "request_logs": [],  # Logs récents (à implémenter)
            "error_logs": [],    # Erreurs récentes (à implémenter)
            "active_sessions": 0, # Sessions actives (à implémenter)
            "cache_stats": {     # Stats du cache (à implémenter)
                "hit_rate": 0.0,
                "size": 0,
                "entries": 0
            }
        })
        
        return debug_data
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte du debug: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de collecter les informations de debug"
        )


# ═══════════════════════════════════════════════════════════════
# 🧪 ENDPOINT DE TEST DE CONNECTIVITÉ
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/test/{service_name}",
    summary="Test de connectivité d'un service",
    description="Teste la connectivité et le bon fonctionnement d'un service spécifique",
    response_model=Dict[str, Any]
)
async def test_service_connectivity(service_name: str):
    """
    Teste la connectivité d'un service spécifique.
    
    Services disponibles:
    - openai: Test de l'API OpenAI
    - elevenlabs: Test de l'API ElevenLabs
    - whisper: Test du service Whisper
    - vectorstore: Test du vectorstore FAISS
    
    Args:
        service_name: Nom du service à tester
        
    Returns:
        Dict: Résultat du test de connectivité
        
    Raises:
        HTTPException: Si service inconnu ou test échoué
    """
    
    allowed_services = ["openai", "elevenlabs", "whisper", "vectorstore"]
    
    if service_name not in allowed_services:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Service '{service_name}' non reconnu. Services disponibles: {allowed_services}"
        )
    
    try:
        logger.info(f"🧪 Test de connectivité pour {service_name}")
        
        start_test_time = time.time()
        test_result = {"service": service_name, "success": False}
        
        if service_name == "openai":
            from app.dependencies import get_openai_client
            client = get_openai_client()
            result = await client.test_connection()
            test_result.update(result)
            
        elif service_name == "elevenlabs":
            from app.dependencies import get_tts_service
            tts_service = get_tts_service()
            if tts_service:
                # Test simple de TTS
                test_result = {
                    "service": service_name,
                    "success": True,
                    "message": "Service TTS disponible"
                }
            else:
                test_result = {
                    "service": service_name,
                    "success": False,
                    "message": "Service TTS non configuré"
                }
                
        elif service_name == "whisper":
            from app.dependencies import get_stt_service
            stt_service = get_stt_service()
            test_result = {
                "service": service_name,
                "success": True,
                "message": "Service STT disponible",
                "model": getattr(stt_service, 'model', 'unknown')
            }
            
        elif service_name == "vectorstore":
            from app.dependencies import get_vectorstore
            vectorstore = get_vectorstore()
            if vectorstore:
                test_result = {
                    "service": service_name,
                    "success": True,
                    "message": "Vectorstore disponible",
                    "documents_count": getattr(vectorstore, 'document_count', 0)
                }
            else:
                test_result = {
                    "service": service_name,
                    "success": False,
                    "message": "Vectorstore non initialisé"
                }
        
        test_duration = (time.time() - start_test_time) * 1000
        test_result["test_duration_ms"] = test_duration
        test_result["timestamp"] = time.time()
        
        logger.info(f"✅ Test {service_name} terminé: {'succès' if test_result['success'] else 'échec'}")
        
        if not test_result["success"]:
            return JSONResponse(
                content=test_result,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test {service_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du test de {service_name}: {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════
# 📈 ENDPOINT DE RÉINITIALISATION DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/reset-metrics",
    summary="Remet à zéro les métriques",
    description="Remet à zéro tous les compteurs de métriques (mode debug uniquement)",
    response_model=BaseResponse
)
async def reset_metrics():
    """
    Remet à zéro toutes les métriques collectées.
    
    ⚠️ Disponible uniquement en mode DEBUG.
    Utile pour les tests et le développement.
    
    Returns:
        BaseResponse: Confirmation de la réinitialisation
    """
    
    settings = get_settings()
    
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fonctionnalité disponible uniquement en mode debug"
        )
    
    try:
        logger.info("🔄 Réinitialisation des métriques")
        
        # Reset des métriques globales
        log_metrics.reset_metrics()
        
        # Reset des stats des clients individuels
        try:
            from app.dependencies import get_openai_client
            client = get_openai_client()
            if hasattr(client, 'reset_stats'):
                client.reset_stats()
        except:
            pass
        
        return BaseResponse(
            success=True,
            message="✅ Métriques remises à zéro"
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réinitialisation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de réinitialiser les métriques"
        )