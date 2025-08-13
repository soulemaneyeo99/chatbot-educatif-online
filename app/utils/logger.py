"""
Système de Logs du Chatbot Éducatif Vocal
=========================================

Module de logging avancé avec formatage coloré, rotation des fichiers,
et configuration adaptée pour le développement et la production.

Utilisation:
    from app.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Message d'information")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback

from app.config import get_settings


# ═══════════════════════════════════════════════════════════════
# 🎨 COULEURS POUR LES LOGS CONSOLE
# ═══════════════════════════════════════════════════════════════

class LogColors:
    """Codes de couleurs ANSI pour les logs."""
    
    # Couleurs de base
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Couleurs de texte
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Couleurs vives
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Couleurs de fond
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


# ═══════════════════════════════════════════════════════════════
# 🎯 FORMATEURS PERSONNALISÉS
# ═══════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    """Formateur avec couleurs pour la console."""
    
    # Mappage niveau -> couleur
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BRIGHT_RED + LogColors.BOLD,
    }
    
    def __init__(self, fmt: Optional[str] = None):
        super().__init__()
        self.fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    def format(self, record: logging.LogRecord) -> str:
        """Formate le log avec des couleurs."""
        
        # Couleur selon le niveau
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
        
        # Formatage de base
        log_message = super().format(record)
        
        # Application des couleurs si on est dans un terminal
        if sys.stdout.isatty():
            # Coloration du niveau
            colored_level = f"{level_color}{record.levelname:<8}{LogColors.RESET}"
            log_message = log_message.replace(f"{record.levelname:<8}", colored_level)
            
            # Coloration du nom du module
            if record.name:
                colored_name = f"{LogColors.DIM}{record.name}{LogColors.RESET}"
                log_message = log_message.replace(record.name, colored_name)
            
            # Coloration spéciale pour les erreurs critiques
            if record.levelno >= logging.CRITICAL:
                log_message = f"{LogColors.BG_RED}{LogColors.WHITE}{log_message}{LogColors.RESET}"
        
        return log_message


class JSONFormatter(logging.Formatter):
    """Formateur JSON pour les logs de production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formate le log en JSON."""
        
        # Données de base
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajout de l'exception si présente
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Ajout des données extra si présentes
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        if hasattr(record, 'duration'):
            log_data["duration_ms"] = record.duration
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Filtre pour ajouter du contexte aux logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Ajoute des informations contextuelles au log."""
        
        # Ajout du nom de l'application
        record.app_name = "chatbot-educatif"
        
        # Ajout de l'environnement
        settings = get_settings()
        record.environment = settings.ENVIRONMENT
        
        return True


# ═══════════════════════════════════════════════════════════════
# 🏗️ CONFIGURATION DU SYSTÈME DE LOGS
# ═══════════════════════════════════════════════════════════════

def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure et retourne un logger personnalisé.
    
    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format personnalisé des logs
        log_file: Fichier de log optionnel
        
    Returns:
        logging.Logger: Logger configuré
        
    Example:
        logger = setup_logger(__name__)
        logger.info("Application démarrée")
    """
    
    settings = get_settings()
    
    # Configuration du logger
    logger = logging.getLogger(name)
    
    # Éviter la duplication des handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Niveau de log
    log_level = level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Filtre contextuel
    context_filter = ContextFilter()
    logger.addFilter(context_filter)
    
    # ═══════════════════════════════════════════════════════════════
    # 🖥️ HANDLER CONSOLE
    # ═══════════════════════════════════════════════════════════════
    
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.is_production:
        # Format JSON en production
        console_formatter = JSONFormatter()
    else:
        # Format coloré en développement
        console_formatter = ColoredFormatter(
            format_string or settings.LOG_FORMAT
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # ═══════════════════════════════════════════════════════════════
    # 📁 HANDLER FICHIER (si configuré)
    # ═══════════════════════════════════════════════════════════════
    
    log_file_path = log_file or settings.LOG_FILE
    if log_file_path:
        
        # Création du répertoire de logs
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handler avec rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Format JSON pour les fichiers
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


# ═══════════════════════════════════════════════════════════════
# 🎛️ LOGGER SPÉCIALISÉS
# ═══════════════════════════════════════════════════════════════

def get_api_logger() -> logging.Logger:
    """Logger spécialisé pour les appels API."""
    logger = setup_logger("chatbot.api")
    return logger


def get_claude_logger() -> logging.Logger:
    """Logger spécialisé pour Claude API."""
    logger = setup_logger("chatbot.claude")
    return logger


def get_audio_logger() -> logging.Logger:
    """Logger spécialisé pour l'audio (STT/TTS)."""
    logger = setup_logger("chatbot.audio")
    return logger


def get_rag_logger() -> logging.Logger:
    """Logger spécialisé pour le système RAG."""
    logger = setup_logger("chatbot.rag")
    return logger


# ═══════════════════════════════════════════════════════════════
# 📊 MÉTRIQUES ET MONITORING
# ═══════════════════════════════════════════════════════════════

class LogMetrics:
    """Collecteur de métriques basé sur les logs."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "requests_total": 0,
            "errors_total": 0,
            "claude_calls": 0,
            "tts_calls": 0,
            "stt_calls": 0,
            "average_response_time": 0.0
        }
        self.response_times = []
    
    def log_request(self, duration_ms: float):
        """Enregistre une requête."""
        self.metrics["requests_total"] += 1
        self.response_times.append(duration_ms)
        
        # Calcul de la moyenne glissante (100 dernières requêtes)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        self.metrics["average_response_time"] = sum(self.response_times) / len(self.response_times)
    
    def log_error(self):
        """Enregistre une erreur."""
        self.metrics["errors_total"] += 1
    
    def log_claude_call(self):
        """Enregistre un appel Claude."""
        self.metrics["claude_calls"] += 1
    
    def log_tts_call(self):
        """Enregistre un appel TTS."""
        self.metrics["tts_calls"] += 1
    
    def log_stt_call(self):
        """Enregistre un appel STT."""
        self.metrics["stt_calls"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques actuelles."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Remet à zéro les métriques."""
        self.__init__()


# Instance globale des métriques
log_metrics = LogMetrics()


# ═══════════════════════════════════════════════════════════════
# 🔧 FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """
    Décorateur pour logger automatiquement les appels de fonction.
    
    Args:
        func_name: Nom de la fonction
        args: Arguments positionnels
        kwargs: Arguments par mots-clés
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = setup_logger(func.__module__)
            
            # Log de l'entrée
            logger.debug(f"🔵 Appel de {func_name}(args={args}, kwargs={kwargs})")
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Calcul du temps d'exécution
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.debug(f"✅ {func_name} terminé en {duration:.2f}ms")
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"❌ {func_name} échoué en {duration:.2f}ms: {e}")
                raise
        
        return wrapper
    return decorator


def log_with_context(logger: logging.Logger, level: str, message: str, **context):
    """
    Log un message avec du contexte supplémentaire.
    
    Args:
        logger: Instance du logger
        level: Niveau de log (info, warning, error, etc.)
        message: Message à logger
        **context: Contexte supplémentaire
        
    Example:
        log_with_context(
            logger, "info", "Utilisateur connecté",
            user_id="123", ip="192.168.1.1"
        )
    """
    
    # Création d'un LogRecord avec contexte
    record = logger.makeRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    
    # Ajout du contexte
    for key, value in context.items():
        setattr(record, key, value)
    
    logger.handle(record)


def setup_structured_logging():
    """
    Configure le logging structuré pour toute l'application.
    À appeler au démarrage de l'application.
    """
    
    settings = get_settings()
    
    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Silencer les libs externes
    
    # Logger principal de l'application
    app_logger = setup_logger("chatbot")
    
    # Loggers spécialisés
    setup_logger("chatbot.api")
    setup_logger("chatbot.claude") 
    setup_logger("chatbot.audio")
    setup_logger("chatbot.rag")
    
    app_logger.info(f"🎯 Logging configuré - Niveau: {settings.LOG_LEVEL}")
    
    return app_logger


# ═══════════════════════════════════════════════════════════════
# 🧪 FONCTION DE TEST
# ═══════════════════════════════════════════════════════════════

def test_logging():
    """Teste tous les niveaux de log avec des exemples."""
    
    logger = setup_logger("test_logger")
    
    logger.debug("🔍 Message de débogage")
    logger.info("ℹ️ Information générale")
    logger.warning("⚠️ Avertissement important")
    logger.error("❌ Erreur survenue")
    logger.critical("🚨 Erreur critique!")
    
    # Test avec contexte
    log_with_context(
        logger, "info", "Test avec contexte",
        request_id="req-123",
        user_id="user-456",
        duration=125.5
    )
    
    # Test d'exception
    try:
        raise ValueError("Erreur de test")
    except Exception as e:
        logger.exception("Exception capturée")
    
    print("\n✅ Test du système de logs terminé!")


if __name__ == "__main__":
    test_logging()