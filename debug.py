#!/usr/bin/env python3
"""
Script de Debug Professionnel - Chatbot AmazoOn du Web
====================================================

Script de diagnostic complet pour identifier et résoudre tous les problèmes
potentiels du chatbot d'inclusion numérique pour femmes rurales.

Usage:
    python debug.py [--fix] [--verbose] [--test-api] [--check-all]

Fonctionnalités:
- Vérification complète de l'environnement
- Test de toutes les dépendances
- Validation de la configuration
- Test des APIs externes
- Diagnostic des services
- Suggestions de correction automatiques
"""

import sys
import os
import subprocess
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from datetime import datetime
import traceback

# Ajout du chemin du projet pour les imports
sys.path.insert(0, str(Path(__file__).parent))

# Couleurs pour l'affichage
class Colors:
    """Codes couleurs ANSI pour un affichage professionnel."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Emojis pour les statuts
    SUCCESS = f"{GREEN}✅{RESET}"
    ERROR = f"{RED}❌{RESET}"
    WARNING = f"{YELLOW}⚠️{RESET}"
    INFO = f"{BLUE}ℹ️{RESET}"
    LOADING = f"{CYAN}🔄{RESET}"
    ROCKET = f"{MAGENTA}🚀{RESET}"


class DebugResult:
    """Classe pour stocker les résultats de diagnostic."""
    
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.start_time = time.time()
    
    def add_check(self, name: str, status: str, message: str, details: Optional[Dict] = None):
        """Ajoute un résultat de vérification."""
        self.checks.append({
            "name": name,
            "status": status,  # success, error, warning
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def add_error(self, error: str):
        """Ajoute une erreur."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Ajoute un avertissement."""
        self.warnings.append(warning)
    
    def add_suggestion(self, suggestion: str):
        """Ajoute une suggestion de correction."""
        self.suggestions.append(suggestion)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé du diagnostic."""
        success_count = len([c for c in self.checks if c["status"] == "success"])
        error_count = len([c for c in self.checks if c["status"] == "error"])
        warning_count = len([c for c in self.checks if c["status"] == "warning"])
        
        return {
            "total_checks": len(self.checks),
            "success_count": success_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "success_rate": (success_count / max(len(self.checks), 1)) * 100,
            "duration_seconds": time.time() - self.start_time,
            "overall_status": "success" if error_count == 0 else "error" if error_count > 3 else "warning"
        }


class ChatbotDebugger:
    """Classe principale de diagnostic du chatbot."""
    
    def __init__(self, verbose: bool = False, auto_fix: bool = False):
        self.verbose = verbose
        self.auto_fix = auto_fix
        self.result = DebugResult()
        self.project_root = Path(__file__).parent
        
        print(f"{Colors.ROCKET} {Colors.BOLD}Diagnostic Chatbot AmazoOn du Web{Colors.RESET}")
        print(f"{Colors.CYAN}Chatbot d'inclusion numérique pour femmes rurales{Colors.RESET}")
        print(f"{Colors.BLUE}Projet: {self.project_root}{Colors.RESET}")
        print("=" * 60)
    
    def log(self, message: str, level: str = "info"):
        """Log avec couleurs selon le niveau."""
        if level == "success":
            print(f"{Colors.SUCCESS} {message}")
        elif level == "error":
            print(f"{Colors.ERROR} {message}")
        elif level == "warning":
            print(f"{Colors.WARNING} {message}")
        elif level == "info":
            print(f"{Colors.INFO} {message}")
        else:
            print(f"{Colors.LOADING} {message}")
    
    def run_full_diagnostic(self) -> DebugResult:
        """Lance le diagnostic complet."""
        try:
            print(f"\n{Colors.BOLD}🔍 PHASE 1: ENVIRONNEMENT SYSTÈME{Colors.RESET}")
            self.check_python_environment()
            self.check_project_structure()
            self.check_required_files()
            
            print(f"\n{Colors.BOLD}📦 PHASE 2: DÉPENDANCES{Colors.RESET}")
            self.check_dependencies()
            self.check_imports()
            
            print(f"\n{Colors.BOLD}⚙️ PHASE 3: CONFIGURATION{Colors.RESET}")
            self.check_configuration()
            self.check_environment_variables()
            
            print(f"\n{Colors.BOLD}🔌 PHASE 4: SERVICES EXTERNES{Colors.RESET}")
            self.check_external_apis()
            
            print(f"\n{Colors.BOLD}🏗️ PHASE 5: ARCHITECTURE INTERNE{Colors.RESET}")
            self.check_internal_services()
            self.check_database_connections()
            
            print(f"\n{Colors.BOLD}🧪 PHASE 6: TESTS FONCTIONNELS{Colors.RESET}")
            self.run_functional_tests()
            
            print(f"\n{Colors.BOLD}📊 PHASE 7: PERFORMANCE{Colors.RESET}")
            self.check_performance()
            
        except Exception as e:
            self.log(f"Erreur critique pendant le diagnostic: {e}", "error")
            self.result.add_error(f"Diagnostic interrompu: {e}")
            if self.verbose:
                traceback.print_exc()
        
        return self.result
    
    def check_python_environment(self):
        """Vérifie l'environnement Python."""
        try:
            # Version Python
            python_version = sys.version_info
            if python_version >= (3, 9):
                self.result.add_check(
                    "python_version", "success", 
                    f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓",
                    {"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
                )
                self.log(f"Python {python_version.major}.{python_version.minor} détecté", "success")
            else:
                self.result.add_check(
                    "python_version", "error",
                    f"Python {python_version.major}.{python_version.minor} trop ancien (requis: 3.9+)"
                )
                self.log("Version Python trop ancienne", "error")
                self.result.add_suggestion("Installer Python 3.9+ : https://www.python.org/downloads/")
            
            # Environnement virtuel
            in_venv = sys.prefix != sys.base_prefix
            if in_venv:
                self.result.add_check("virtual_env", "success", "Environnement virtuel actif ✓")
                self.log("Environnement virtuel détecté", "success")
            else:
                self.result.add_check("virtual_env", "warning", "Pas d'environnement virtuel détecté")
                self.log("Recommandation: utiliser un environnement virtuel", "warning")
                self.result.add_suggestion("Créer un venv: python -m venv venv && source venv/bin/activate")
            
            # Pip disponible
            try:
                import pip
                self.result.add_check("pip", "success", "Pip disponible ✓")
            except ImportError:
                self.result.add_check("pip", "error", "Pip non disponible")
                self.result.add_suggestion("Installer pip: python -m ensurepip --upgrade")
                
        except Exception as e:
            self.result.add_check("python_env", "error", f"Erreur environnement Python: {e}")
    
    def check_project_structure(self):
        """Vérifie la structure du projet."""
        required_structure = {
            "main.py": "Point d'entrée principal",
            "app/": "Package principal",
            "app/__init__.py": "Init package app",
            "app/config.py": "Configuration",
            "app/dependencies.py": "Injection dépendances",
            "app/api/": "Package API",
            "app/api/__init__.py": "Init package API",
            "app/api/routes.py": "Routes principales",
            "app/api/endpoints/": "Endpoints spécialisés",
            "app/api/endpoints/__init__.py": "Init endpoints",
            "app/api/endpoints/health.py": "Endpoints santé",
            "app/api/endpoints/rag.py": "Endpoints conversation",
            "app/api/endpoints/audio.py": "Endpoints audio",
            "app/models/": "Modèles Pydantic",
            "app/models/__init__.py": "Init modèles",
            "app/models/requests.py": "Modèles requêtes",
            "app/models/responses.py": "Modèles réponses",
            "app/services/": "Services métier",
            "app/services/__init__.py": "Init services",
            "app/utils/": "Utilitaires",
            "app/utils/__init__.py": "Init utils",
            "app/utils/logger.py": "Système logs",
            "app/utils/exceptions.py": "Exceptions métier",
            "data/": "Données",
            "data/documents/": "Documents pédagogiques",
            "data/vectorstore/": "Index FAISS"
        }
        
        missing_files = []
        present_files = []
        
        for path, description in required_structure.items():
            full_path = self.project_root / path
            if full_path.exists():
                present_files.append(path)
                if self.verbose:
                    self.log(f"✓ {path} - {description}", "success")
            else:
                missing_files.append(path)
                self.log(f"✗ Manquant: {path} - {description}", "error")
        
        if missing_files:
            self.result.add_check(
                "project_structure", "error",
                f"{len(missing_files)} fichiers manquants sur {len(required_structure)}",
                {"missing": missing_files, "present": present_files}
            )
            
            if self.auto_fix:
                self.log("🔧 Création automatique des fichiers manquants...", "info")
                self.create_missing_files(missing_files)
        else:
            self.result.add_check(
                "project_structure", "success",
                f"Structure projet complète ({len(required_structure)} fichiers) ✓"
            )
    
    def create_missing_files(self, missing_files: List[str]):
        """Crée automatiquement les fichiers manquants."""
        templates = {
            "__init__.py": "",
            "app/services/tts.py": '''"""Stub TTS Service"""
class TTSService:
    def __init__(self, *args, **kwargs):
        pass
    
    async def synthesize(self, text: str, **kwargs):
        return {"message": "TTS non implémenté"}
''',
            "app/services/stt.py": '''"""Stub STT Service"""
class STTService:
    def __init__(self, *args, **kwargs):
        self.model = "stub"
    
    async def transcribe(self, **kwargs):
        return {"text": "STT non implémenté"}
''',
            "app/services/vectorstore.py": '''"""Stub Vectorstore Service"""
class VectorStore:
    def __init__(self, *args, **kwargs):
        self.document_count = 0
    
    async def search(self, query: str, **kwargs):
        return []
'''
        }
        
        for file_path in missing_files:
            full_path = self.project_root / file_path
            
            # Créer les répertoires parent si nécessaire
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Déterminer le contenu du fichier
            if file_path.endswith("__init__.py"):
                content = templates["__init__.py"]
            elif file_path in templates:
                content = templates[file_path]
            elif file_path.endswith("/"):
                # C'est un répertoire
                full_path.mkdir(exist_ok=True)
                continue
            else:
                content = f'"""Fichier auto-généré - À implémenter"""\n'
            
            # Créer le fichier
            try:
                full_path.write_text(content, encoding="utf-8")
                self.log(f"✓ Créé: {file_path}", "success")
            except Exception as e:
                self.log(f"✗ Erreur création {file_path}: {e}", "error")
    
    def check_required_files(self):
        """Vérifie les fichiers de configuration requis."""
        config_files = {
            ".env": "Variables d'environnement",
            "requirements.txt": "Dépendances Python"
        }
        
        for file_name, description in config_files.items():
            file_path = self.project_root / file_name
            if file_path.exists():
                self.result.add_check(f"config_file_{file_name}", "success", f"{description} présent ✓")
                
                # Vérifications spécifiques
                if file_name == ".env":
                    self.check_env_file_content(file_path)
                elif file_name == "requirements.txt":
                    self.check_requirements_file(file_path)
            else:
                self.result.add_check(f"config_file_{file_name}", "error", f"{description} manquant")
                self.result.add_suggestion(f"Créer {file_name}")
                
                if self.auto_fix and file_name == ".env":
                    self.create_default_env_file()
    
    def check_env_file_content(self, env_path: Path):
        """Vérifie le contenu du fichier .env."""
        try:
            content = env_path.read_text()
            lines = content.strip().split('\n')
            
            # Vérification des doublons
            env_vars = {}
            duplicates = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=')[0].strip()
                    if key in env_vars:
                        duplicates.append(f"Ligne {line_num}: {key} (première occurrence ligne {env_vars[key]})")
                    else:
                        env_vars[key] = line_num
            
            if duplicates:
                self.result.add_check(
                    "env_duplicates", "error",
                    f"Variables dupliquées dans .env: {duplicates}",
                    {"duplicates": duplicates}
                )
                self.log("❌ Variables dupliquées détectées dans .env", "error")
                for dup in duplicates:
                    self.log(f"   • {dup}", "error")
                self.result.add_suggestion("Supprimer les variables dupliquées dans .env")
            else:
                self.result.add_check("env_duplicates", "success", "Aucun doublon dans .env ✓")
            
            # Vérification des variables requises
            required_vars = {
                "OPENAI_API_KEY": "Clé API OpenAI (obligatoire)",
                "DEBUG": "Mode debug",
                "ENVIRONMENT": "Environnement (dev/staging/prod)"
            }
            
            missing_vars = []
            invalid_vars = []
            
            for var, description in required_vars.items():
                if var not in env_vars:
                    missing_vars.append(f"{var} - {description}")
                else:
                    # Vérification de la valeur
                    var_line = None
                    for line in lines:
                        if line.strip().startswith(f"{var}="):
                            var_line = line.strip()
                            break
                    
                    if var_line:
                        value = var_line.split('=', 1)[1].strip()
                        
                        # Vérifications spécifiques
                        if var == "OPENAI_API_KEY":
                            if not value or value in ["sk-VOTRE_CLE_OPENAI_ICI", "your-openai-key"]:
                                invalid_vars.append(f"{var} - Clé placeholder, remplacer par vraie clé")
                            elif not value.startswith("sk-"):
                                invalid_vars.append(f"{var} - Format invalide (doit commencer par 'sk-')")
                        
                        elif var == "CHUNK_SIZE" and value:
                            try:
                                int(value)
                            except ValueError:
                                invalid_vars.append(f"{var} - Doit être un nombre entier")
                        
                        elif var == "CHUNK_OVERLAP" and value:
                            try:
                                overlap = int(value)
                                chunk_size = 1000  # Valeur par défaut
                                for line in lines:
                                    if line.strip().startswith("CHUNK_SIZE="):
                                        try:
                                            chunk_size = int(line.split('=')[1].strip())
                                        except:
                                            pass
                                if overlap >= chunk_size:
                                    invalid_vars.append(f"{var} - Doit être inférieur à CHUNK_SIZE ({chunk_size})")
                            except ValueError:
                                invalid_vars.append(f"{var} - Doit être un nombre entier")
            
            # Rapport des variables manquantes
            if missing_vars:
                self.result.add_check(
                    "env_variables", "error",
                    f"{len(missing_vars)} variables manquantes dans .env",
                    {"missing": missing_vars}
                )
                for var in missing_vars:
                    self.log(f"❌ Variable manquante: {var}", "error")
            else:
                self.result.add_check("env_variables", "success", "Variables .env complètes ✓")
            
            # Rapport des variables invalides
            if invalid_vars:
                self.result.add_check(
                    "env_validation", "error",
                    f"{len(invalid_vars)} variables invalides dans .env",
                    {"invalid": invalid_vars}
                )
                for var in invalid_vars:
                    self.log(f"❌ Variable invalide: {var}", "error")
                self.result.add_suggestion("Corriger les valeurs invalides dans .env")
            else:
                self.result.add_check("env_validation", "success", "Variables .env valides ✓")
                
            # Vérification de la clé OpenAI
            openai_key = env_vars.get("OPENAI_API_KEY")
            if openai_key:
                for line in lines:
                    if line.strip().startswith("OPENAI_API_KEY="):
                        key_value = line.split('=', 1)[1].strip()
                        if len(key_value) > 50 and key_value.startswith("sk-"):
                            self.result.add_check("openai_key_format", "success", "Clé OpenAI format valide ✓")
                            self.log("✅ Clé OpenAI détectée avec format valide", "success")
                        break
                
        except Exception as e:
            self.result.add_check("env_file_read", "error", f"Erreur lecture .env: {e}")
            self.log(f"❌ Impossible de lire .env: {e}", "error")
    
    def create_default_env_file(self):
        """Crée un fichier .env par défaut."""
        default_env = """# Configuration Chatbot AmazoOn du Web
# =====================================

# 🔑 API KEYS (À COMPLÉTER OBLIGATOIREMENT)
OPENAI_API_KEY=sk-VOTRE_CLE_OPENAI_ICI
ELEVENLABS_API_KEY=VOTRE_CLE_ELEVENLABS_OPTIONNELLE

# 🏗️ CONFIGURATION GÉNÉRALE
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# 🌐 SERVEUR
HOST=127.0.0.1
PORT=8000

# 🤖 OPENAI
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# 🔊 ELEVENLABS (TTS)
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_STABILITY=0.75
ELEVENLABS_SIMILARITY=0.75

# 📚 RAG ET VECTORSTORE
MAX_SEARCH_RESULTS=5
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""
        try:
            env_path = self.project_root / ".env"
            env_path.write_text(default_env)
            self.log("✓ Fichier .env créé avec configuration par défaut", "success")
            self.result.add_suggestion("⚠️ IMPORTANT: Ajoutez votre vraie clé OpenAI dans .env")
        except Exception as e:
            self.log(f"Erreur création .env: {e}", "error")
    
    def check_requirements_file(self, req_path: Path):
        """Vérifie le fichier requirements.txt."""
        try:
            content = req_path.read_text()
            critical_deps = [
                "fastapi", "uvicorn", "openai", "pydantic", "python-dotenv"
            ]
            
            missing_deps = []
            for dep in critical_deps:
                if dep not in content:
                    missing_deps.append(dep)
            
            if missing_deps:
                self.result.add_check(
                    "critical_dependencies", "error",
                    f"Dépendances critiques manquantes: {missing_deps}"
                )
                self.result.add_suggestion(f"Ajouter à requirements.txt: {' '.join(missing_deps)}")
            else:
                self.result.add_check("critical_dependencies", "success", "Dépendances critiques présentes ✓")
                
        except Exception as e:
            self.result.add_check("requirements_read", "error", f"Erreur lecture requirements.txt: {e}")
    
    def check_dependencies(self):
        """Vérifie l'installation des dépendances."""
        critical_packages = {
            "fastapi": "Framework API",
            "uvicorn": "Serveur ASGI",
            "openai": "Client OpenAI",
            "pydantic": "Validation données",
            "python-dotenv": "Variables environnement"
        }
        
        optional_packages = {
            "elevenlabs": "Synthèse vocale",
            "whisper": "Reconnaissance vocale",  
            "langchain": "Framework RAG",
            "faiss-cpu": "Vectorstore",
            "sentence-transformers": "Embeddings"
        }
        
        installed_packages = self.get_installed_packages()
        
        # Vérification packages critiques
        missing_critical = []
        for package, description in critical_packages.items():
            if package in installed_packages:
                version = installed_packages[package]
                self.result.add_check(
                    f"package_{package}", "success", 
                    f"{package} {version} installé ✓",
                    {"version": version}
                )
                if self.verbose:
                    self.log(f"✓ {package} {version} - {description}", "success")
            else:
                missing_critical.append(package)
                self.result.add_check(f"package_{package}", "error", f"{package} manquant - {description}")
                self.log(f"✗ {package} manquant - {description}", "error")
        
        # Vérification packages optionnels
        missing_optional = []
        for package, description in optional_packages.items():
            if package in installed_packages:
                version = installed_packages[package]
                self.result.add_check(
                    f"optional_{package}", "success",
                    f"{package} {version} disponible ✓"
                )
                if self.verbose:
                    self.log(f"✓ {package} {version} - {description}", "success")
            else:
                missing_optional.append(package)
                self.result.add_check(f"optional_{package}", "warning", f"{package} optionnel manquant")
                if self.verbose:
                    self.log(f"⚠ {package} manquant - {description} (optionnel)", "warning")
        
        # Suggestions d'installation
        if missing_critical:
            self.result.add_suggestion(f"Installer packages critiques: pip install {' '.join(missing_critical)}")
        if missing_optional:
            self.result.add_suggestion(f"Installer packages optionnels: pip install {' '.join(missing_optional)}")
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Retourne la liste des packages installés avec versions."""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return {pkg["name"].lower(): pkg["version"] for pkg in packages}
            else:
                self.log("Impossible de lister les packages installés", "warning")
                return {}
        except Exception as e:
            self.log(f"Erreur liste packages: {e}", "warning")
            return {}
    
    def check_imports(self):
        """Teste les imports critiques."""
        critical_imports = {
            "fastapi": "FastAPI framework",
            "uvicorn": "Serveur ASGI",  
            "openai": "Client OpenAI",
            "pydantic": "Validation données",
            "dotenv": "Variables environnement"
        }
        
        for module, description in critical_imports.items():
            try:
                if module == "dotenv":
                    __import__("dotenv")
                else:
                    __import__(module)
                self.result.add_check(f"import_{module}", "success", f"Import {module} ✓")
                if self.verbose:
                    self.log(f"✓ Import {module} réussi", "success")
            except ImportError as e:
                self.result.add_check(f"import_{module}", "error", f"Import {module} échoué: {e}")
                self.log(f"✗ Import {module} échoué - {description}", "error")
        
        # Test imports du projet
        project_imports = [
            "app.config",
            "app.dependencies", 
            "app.api.routes",
            "app.utils.logger",
            "app.utils.exceptions"
        ]
        
        for module in project_imports:
            try:
                __import__(module)
                self.result.add_check(f"project_import_{module}", "success", f"Import projet {module} ✓")
                if self.verbose:
                    self.log(f"✓ Import projet {module} réussi", "success")
            except Exception as e:
                self.result.add_check(f"project_import_{module}", "error", f"Import projet {module} échoué: {e}")
                self.log(f"✗ Import projet {module} échoué", "error")
                if self.verbose:
                    self.log(f"   Détail: {e}", "error")
    
    def check_configuration(self):
        """Vérifie la configuration de l'application."""
        try:
            # Test chargement configuration
            from app.config import get_settings
            settings = get_settings()
            
            self.result.add_check("config_load", "success", "Configuration chargée ✓")
            
            # Vérification des paramètres critiques
            critical_settings = {
                "OPENAI_API_KEY": "Clé API OpenAI",
                "OPENAI_MODEL": "Modèle OpenAI",
                "DEBUG": "Mode debug",
                "ENVIRONMENT": "Environnement"
            }
            
            for setting, description in critical_settings.items():
                if hasattr(settings, setting):
                    value = getattr(settings, setting)
                    if value:
                        self.result.add_check(
                            f"setting_{setting}", "success", 
                            f"{description} configuré ✓"
                        )
                    else:
                        self.result.add_check(
                            f"setting_{setting}", "error",
                            f"{description} vide ou manquant"
                        )
                        if setting == "OPENAI_API_KEY":
                            self.result.add_suggestion("Ajouter votre clé OpenAI dans .env")
                else:
                    self.result.add_check(
                        f"setting_{setting}", "error",
                        f"{description} non défini"
                    )
            
            # Validation clé OpenAI
            if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY:
                api_key = settings.OPENAI_API_KEY
                if api_key.startswith("sk-"):
                    if api_key == "sk-VOTRE_CLE_OPENAI_ICI" or len(api_key) < 20:
                        self.result.add_check("openai_key_valid", "warning", "Clé OpenAI semble être un placeholder")
                        self.result.add_suggestion("Remplacer par votre vraie clé OpenAI")
                    else:
                        self.result.add_check("openai_key_valid", "success", "Format clé OpenAI valide ✓")
                else:
                    self.result.add_check("openai_key_valid", "error", "Format clé OpenAI invalide")
                    self.result.add_suggestion("La clé OpenAI doit commencer par 'sk-'")
            
        except Exception as e:
            self.result.add_check("config_load", "error", f"Erreur chargement configuration: {e}")
            if self.verbose:
                traceback.print_exc()
    
    def check_environment_variables(self):
        """Vérifie les variables d'environnement."""
        required_env_vars = {
            "OPENAI_API_KEY": "Clé API OpenAI (critique)",
        }
        
        optional_env_vars = {
            "ELEVENLABS_API_KEY": "Clé ElevenLabs pour TTS",
            "DEBUG": "Mode debug",
            "ENVIRONMENT": "Environnement déploiement"
        }
        
        # Variables critiques
        for var, description in required_env_vars.items():
            value = os.getenv(var)
            if value:
                self.result.add_check(f"env_{var}", "success", f"{description} définie ✓")
            else:
                self.result.add_check(f"env_{var}", "error", f"{description} manquante")
                self.result.add_suggestion(f"Définir {var} dans .env")
        
        # Variables optionnelles
        for var, description in optional_env_vars.items():
            value = os.getenv(var)
            if value:
                self.result.add_check(f"env_optional_{var}", "success", f"{description} définie ✓")
            else:
                self.result.add_check(f"env_optional_{var}", "warning", f"{description} optionnelle manquante")
    
    def check_external_apis(self):
        """Teste la connectivité aux APIs externes."""
        self.log("Test des APIs externes...", "info")
        
        # Test OpenAI API
        self.test_openai_api()
        
        # Test ElevenLabs API (optionnel)
        self.test_elevenlabs_api()
    
    def test_openai_api(self):
        """Teste l'API OpenAI."""
        try:
            from app.config import get_settings
            settings = get_settings()
            
            if not hasattr(settings, "OPENAI_API_KEY") or not settings.OPENAI_API_KEY:
                self.result.add_check("openai_api", "error", "Clé OpenAI manquante - test impossible")
                return
            
            if settings.OPENAI_API_KEY in ["sk-VOTRE_CLE_OPENAI_ICI", "sk-demo-key-for-testing"]:
                self.result.add_check("openai_api", "warning", "Clé OpenAI placeholder - test simulé")
                self.result.add_suggestion("Remplacer par votre vraie clé OpenAI pour tester l'API")
                return
            
            # Test réel de l'API
            import openai
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Test simple
            start_time = time.time()
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": "Test de connexion"}],
                max_tokens=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.choices:
                self.result.add_check(
                    "openai_api", "success", 
                    f"API OpenAI fonctionnelle ✓ ({response_time:.0f}ms)",
                    {
                        "model": settings.OPENAI_MODEL,
                        "response_time_ms": response_time,
                        "tokens_used": response.usage.total_tokens if response.usage else 0
                    }
                )
                self.log(f"✓ API OpenAI répond en {response_time:.0f}ms", "success")
            else:
                self.result.add_check("openai_api", "error", "API OpenAI répond mais sans contenu")
                
        except Exception as e:
            self.result.add_check("openai_api", "error", f"Erreur API OpenAI: {e}")
            self.log(f"✗ Test API OpenAI échoué: {e}", "error")
            
            # Suggestions selon le type d'erreur
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str:
                self.result.add_suggestion("Vérifiez votre clé API OpenAI")
            elif "quota" in error_str or "billing" in error_str:
                self.result.add_suggestion("Vérifiez votre crédit OpenAI sur platform.openai.com")
            elif "network" in error_str or "connection" in error_str:
                self.result.add_suggestion("Vérifiez votre connexion internet")
    
    def test_elevenlabs_api(self):
        """Teste l'API ElevenLabs (optionnel)."""
        try:
            from app.config import get_settings
            settings = get_settings()
            
            if not hasattr(settings, "ELEVENLABS_API_KEY") or not settings.ELEVENLABS_API_KEY:
                self.result.add_check("elevenlabs_api", "warning", "ElevenLabs non configuré (optionnel)")
                return
            
            # Test simulé pour ElevenLabs
            self.result.add_check("elevenlabs_api", "success", "ElevenLabs configuré ✓ (test simulé)")
            self.log("ElevenLabs configuré (synthèse vocale disponible)", "success")
            
        except Exception as e:
            self.result.add_check("elevenlabs_api", "warning", f"Erreur test ElevenLabs: {e}")
    
    def check_internal_services(self):
        """Vérifie les services internes du chatbot."""
        self.log("Test des services internes...", "info")
        
        # Test des dépendances
        try:
            from app.dependencies import (
                get_validated_settings,
                get_openai_client,
                get_vectorstore,
                get_tts_service,
                get_stt_service
            )
            
            # Test settings
            try:
                settings = get_validated_settings()
                self.result.add_check("internal_settings", "success", "Settings internes validées ✓")
            except Exception as e:
                self.result.add_check("internal_settings", "error", f"Erreur settings: {e}")
            
            # Test client OpenAI
            try:
                client = get_openai_client()
                if client:
                    self.result.add_check("internal_openai_client", "success", "Client OpenAI interne initialisé ✓")
                else:
                    self.result.add_check("internal_openai_client", "error", "Client OpenAI interne None")
            except Exception as e:
                self.result.add_check("internal_openai_client", "error", f"Erreur client OpenAI: {e}")
            
            # Test vectorstore (optionnel)
            try:
                vectorstore = get_vectorstore()
                if vectorstore:
                    self.result.add_check("internal_vectorstore", "success", "Vectorstore initialisé ✓")
                else:
                    self.result.add_check("internal_vectorstore", "warning", "Vectorstore non initialisé (normal)")
            except Exception as e:
                self.result.add_check("internal_vectorstore", "warning", f"Vectorstore indisponible: {e}")
            
            # Test services audio
            try:
                tts = get_tts_service()
                if tts:
                    self.result.add_check("internal_tts", "success", "Service TTS disponible ✓")
                else:
                    self.result.add_check("internal_tts", "warning", "Service TTS non configuré")
            except Exception as e:
                self.result.add_check("internal_tts", "warning", f"Service TTS: {e}")
            
            try:
                stt = get_stt_service()
                if stt:
                    self.result.add_check("internal_stt", "success", "Service STT disponible ✓")
                else:
                    self.result.add_check("internal_stt", "warning", "Service STT non configuré")
            except Exception as e:
                self.result.add_check("internal_stt", "warning", f"Service STT: {e}")
                
        except ImportError as e:
            self.result.add_check("internal_imports", "error", f"Erreur import services: {e}")
    
    def check_database_connections(self):
        """Vérifie les connexions base de données (vectorstore)."""
        try:
            data_dir = self.project_root / "data"
            docs_dir = data_dir / "documents"
            vector_dir = data_dir / "vectorstore"
            
            # Vérification répertoires data
            if data_dir.exists():
                self.result.add_check("data_directory", "success", "Répertoire data présent ✓")
            else:
                self.result.add_check("data_directory", "warning", "Répertoire data manquant")
                if self.auto_fix:
                    data_dir.mkdir(exist_ok=True)
                    self.log("✓ Répertoire data créé", "success")
            
            if docs_dir.exists():
                doc_count = len(list(docs_dir.glob("*.md")))
                self.result.add_check(
                    "documents_directory", "success", 
                    f"Répertoire documents présent ({doc_count} fichiers .md) ✓"
                )
            else:
                self.result.add_check("documents_directory", "warning", "Répertoire documents manquant")
                if self.auto_fix:
                    docs_dir.mkdir(parents=True, exist_ok=True)
                    self.log("✓ Répertoire documents créé", "success")
            
            if vector_dir.exists():
                index_files = list(vector_dir.glob("*"))
                if index_files:
                    self.result.add_check("vectorstore", "success", f"Vectorstore présent ({len(index_files)} fichiers) ✓")
                else:
                    self.result.add_check("vectorstore", "warning", "Vectorstore vide")
            else:
                self.result.add_check("vectorstore", "warning", "Répertoire vectorstore manquant")
                if self.auto_fix:
                    vector_dir.mkdir(parents=True, exist_ok=True)
                    self.log("✓ Répertoire vectorstore créé", "success")
            
        except Exception as e:
            self.result.add_check("database_check", "error", f"Erreur vérification données: {e}")
    
    def run_functional_tests(self):
        """Lance des tests fonctionnels de base."""
        self.log("Tests fonctionnels...", "info")
        
        # Test 1: Import et instanciation des modèles
        self.test_pydantic_models()
        
        # Test 2: Routes API (sans démarrer le serveur)
        self.test_api_routes_import()
        
        # Test 3: Logique métier
        self.test_business_logic()
    
    def test_pydantic_models(self):
        """Teste les modèles Pydantic."""
        try:
            from app.models.requests import ChatRequest
            from app.models.responses import ChatResponse
            
            # Test création modèle valide
            test_request = ChatRequest(
                message="Test diagnostic automatique",
                education_level="basic"
            )
            
            self.result.add_check("pydantic_models", "success", "Modèles Pydantic fonctionnels ✓")
            
            # Test validation
            try:
                invalid_request = ChatRequest(message="")  # Message vide = invalide
                self.result.add_check("pydantic_validation", "error", "Validation Pydantic défaillante")
            except Exception:
                self.result.add_check("pydantic_validation", "success", "Validation Pydantic fonctionnelle ✓")
                
        except Exception as e:
            self.result.add_check("pydantic_models", "error", f"Erreur modèles Pydantic: {e}")
    
    def test_api_routes_import(self):
        """Teste l'import des routes API."""
        try:
            from app.api.routes import api_router
            from app.api.endpoints.health import router as health_router
            from app.api.endpoints.rag import router as rag_router
            from app.api.endpoints.audio import router as audio_router
            
            self.result.add_check("api_routes", "success", "Routes API importables ✓")
            
            # Comptage des routes
            route_count = len(api_router.routes)
            self.result.add_check(
                "route_count", "success", 
                f"{route_count} routes API configurées ✓",
                {"route_count": route_count}
            )
            
        except Exception as e:
            self.result.add_check("api_routes", "error", f"Erreur import routes: {e}")
    
    def test_business_logic(self):
        """Teste la logique métier de base."""
        try:
            # Test des fonctions utilitaires
            from app.api.endpoints.rag import (
                detect_learning_module,
                detect_digital_experience_level,
                get_educational_context
            )
            
            # Test détection module
            module = detect_learning_module("comment utiliser whatsapp pour vendre")
            if module == "whatsapp_commerce":
                self.result.add_check("module_detection", "success", "Détection module fonctionnelle ✓")
            else:
                self.result.add_check("module_detection", "warning", f"Détection module inattendue: {module}")
            
            # Test détection niveau
            niveau = detect_digital_experience_level("je débute avec le téléphone")
            if niveau == "debutante":
                self.result.add_check("level_detection", "success", "Détection niveau fonctionnelle ✓")
            else:
                self.result.add_check("level_detection", "warning", f"Détection niveau inattendue: {niveau}")
            
            # Test contexte éducatif
            context = get_educational_context("whatsapp", "whatsapp_commerce", "debutante")
            if context:
                self.result.add_check("educational_context", "success", "Contexte éducatif fonctionnel ✓")
            else:
                self.result.add_check("educational_context", "warning", "Contexte éducatif vide")
                
        except Exception as e:
            self.result.add_check("business_logic", "error", f"Erreur logique métier: {e}")
    
    def check_performance(self):
        """Vérifie les aspects performance."""
        self.log("Tests de performance...", "info")
        
        # Test temps de démarrage des imports
        start_time = time.time()
        try:
            from app.config import get_settings
            from app.dependencies import get_openai_client
            import_time = (time.time() - start_time) * 1000
            
            if import_time < 1000:  # Moins d'1 seconde
                self.result.add_check(
                    "import_performance", "success", 
                    f"Imports rapides ({import_time:.0f}ms) ✓"
                )
            elif import_time < 3000:  # Moins de 3 secondes
                self.result.add_check(
                    "import_performance", "warning",
                    f"Imports moyens ({import_time:.0f}ms)"
                )
            else:
                self.result.add_check(
                    "import_performance", "error",
                    f"Imports lents ({import_time:.0f}ms)"
                )
                self.result.add_suggestion("Optimiser les imports ou vérifier les dépendances")
                
        except Exception as e:
            self.result.add_check("import_performance", "error", f"Erreur test performance: {e}")
        
        # Test mémoire basique
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent < 80:
                self.result.add_check("memory_usage", "success", f"Mémoire disponible ({100-memory_percent:.1f}% libre) ✓")
            else:
                self.result.add_check("memory_usage", "warning", f"Mémoire limitée ({100-memory_percent:.1f}% libre)")
        except ImportError:
            self.result.add_check("memory_check", "warning", "psutil non installé - impossible de vérifier la mémoire")
    
    async def test_api_endpoints(self):
        """Teste les endpoints API en mode async."""
        if not self.result.get_summary()["overall_status"] == "success":
            self.log("Tests API ignorés à cause d'erreurs critiques", "warning")
            return
        
        self.log("Tests des endpoints API...", "info")
        
        try:
            # Test import de FastAPI et création app
            from main import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test endpoint racine
            try:
                response = client.get("/")
                if response.status_code == 200:
                    self.result.add_check("api_root", "success", "Endpoint racine fonctionnel ✓")
                else:
                    self.result.add_check("api_root", "error", f"Endpoint racine erreur {response.status_code}")
            except Exception as e:
                self.result.add_check("api_root", "error", f"Erreur endpoint racine: {e}")
            
            # Test health endpoint
            try:
                response = client.get("/api/v1/health")
                if response.status_code == 200:
                    self.result.add_check("api_health", "success", "Endpoint health fonctionnel ✓")
                else:
                    self.result.add_check("api_health", "error", f"Endpoint health erreur {response.status_code}")
            except Exception as e:
                self.result.add_check("api_health", "error", f"Erreur endpoint health: {e}")
            
            # Test chat endpoint (sans clé API réelle)
            try:
                response = client.post("/api/v1/chat", json={
                    "message": "Test diagnostic",
                    "education_level": "basic"
                })
                # On s'attend à une erreur si pas de clé API valide
                if response.status_code in [200, 503]:  # 503 = service indisponible (normal)
                    self.result.add_check("api_chat", "success", "Endpoint chat accessible ✓")
                else:
                    self.result.add_check("api_chat", "warning", f"Endpoint chat statut {response.status_code}")
            except Exception as e:
                self.result.add_check("api_chat", "error", f"Erreur endpoint chat: {e}")
                
        except Exception as e:
            self.result.add_check("api_test", "error", f"Erreur tests API: {e}")
    
    def print_summary(self):
        """Affiche le résumé final du diagnostic."""
        summary = self.result.get_summary()
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}📋 RÉSUMÉ DU DIAGNOSTIC{Colors.RESET}")
        print(f"{'='*60}")
        
        # Statut global
        overall_status = summary["overall_status"]
        if overall_status == "success":
            status_icon = Colors.SUCCESS
            status_color = Colors.GREEN
        elif overall_status == "warning":
            status_icon = Colors.WARNING
            status_color = Colors.YELLOW
        else:
            status_icon = Colors.ERROR
            status_color = Colors.RED
        
        print(f"\n{Colors.BOLD}🎯 STATUT GLOBAL: {status_color}{overall_status.upper()}{Colors.RESET} {status_icon}")
        
        # Statistiques
        print(f"\n📊 STATISTIQUES:")
        print(f"   • Total vérifications: {summary['total_checks']}")
        print(f"   • {Colors.GREEN}✅ Succès: {summary['success_count']}{Colors.RESET}")
        print(f"   • {Colors.YELLOW}⚠️  Avertissements: {summary['warning_count']}{Colors.RESET}")
        print(f"   • {Colors.RED}❌ Erreurs: {summary['error_count']}{Colors.RESET}")
        print(f"   • 📈 Taux de réussite: {summary['success_rate']:.1f}%")
        print(f"   • ⏱️  Durée: {summary['duration_seconds']:.1f}s")
        
        # Recommandations prioritaires
        if self.result.suggestions:
            print(f"\n{Colors.BOLD}💡 RECOMMANDATIONS PRIORITAIRES:{Colors.RESET}")
            for i, suggestion in enumerate(self.result.suggestions[:5], 1):
                print(f"   {i}. {suggestion}")
            
            if len(self.result.suggestions) > 5:
                print(f"   ... et {len(self.result.suggestions) - 5} autres")
        
        # Erreurs critiques
        if self.result.errors:
            print(f"\n{Colors.BOLD}{Colors.RED}🚨 ERREURS CRITIQUES:{Colors.RESET}")
            for error in self.result.errors[:3]:
                print(f"   • {error}")
        
        # Action recommandée
        print(f"\n{Colors.BOLD}🎯 ACTION RECOMMANDÉE:{Colors.RESET}")
        if overall_status == "success":
            print(f"   {Colors.GREEN}✅ Chatbot prêt ! Lancez: python main.py{Colors.RESET}")
        elif overall_status == "warning":
            print(f"   {Colors.YELLOW}⚠️  Corrigez les avertissements puis lancez: python main.py{Colors.RESET}")
        else:
            print(f"   {Colors.RED}🔧 Corrigez les erreurs critiques avant de démarrer{Colors.RESET}")
        
        # Aide
        print(f"\n{Colors.BOLD}ℹ️  AIDE:{Colors.RESET}")
        print(f"   • Documentation: README.md")
        print(f"   • Debug détaillé: python debug.py --verbose")
        print(f"   • Correction auto: python debug.py --fix")
        print(f"   • Test API: python debug.py --test-api")
        
        print(f"\n{Colors.CYAN}🌍 Chatbot AmazoOn du Web - Inclusion Numérique Femmes Rurales{Colors.RESET}")
        print(f"{'='*60}")
    
    def save_report(self, filename: str = None):
        """Sauvegarde le rapport de diagnostic."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "project": "Chatbot AmazoOn du Web",
                "diagnostic_version": "1.0.0"
            },
            "summary": self.result.get_summary(),
            "checks": self.result.checks,
            "errors": self.result.errors,
            "warnings": self.result.warnings,
            "suggestions": self.result.suggestions
        }
        
        try:
            report_path = self.project_root / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log(f"Rapport sauvegardé: {filename}", "success")
            return str(report_path)
        except Exception as e:
            self.log(f"Erreur sauvegarde rapport: {e}", "error")
            return None


async def main():
    """Fonction principale du script de debug."""
    parser = argparse.ArgumentParser(
        description="Script de diagnostic professionnel - Chatbot AmazoOn du Web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python debug.py                    # Diagnostic standard
  python debug.py --verbose          # Diagnostic détaillé
  python debug.py --fix              # Correction automatique
  python debug.py --test-api         # Test des endpoints API
  python debug.py --check-all        # Diagnostic complet + API
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affichage détaillé des vérifications"
    )
    
    parser.add_argument(
        "--fix", "-f",
        action="store_true", 
        help="Correction automatique des problèmes détectés"
    )
    
    parser.add_argument(
        "--test-api", "-a",
        action="store_true",
        help="Test des endpoints API (nécessite configuration valide)"
    )
    
    parser.add_argument(
        "--check-all", "-c",
        action="store_true",
        help="Diagnostic complet incluant tests API"
    )
    
    parser.add_argument(
        "--save-report", "-s",
        metavar="FILENAME",
        help="Sauvegarder le rapport de diagnostic"
    )
    
    args = parser.parse_args()
    
    # Initialisation du debugger
    debugger = ChatbotDebugger(
        verbose=args.verbose,
        auto_fix=args.fix
    )
    
    # Diagnostic principal
    result = debugger.run_full_diagnostic()
    
    # Tests API si demandés
    if args.test_api or args.check_all:
        if result.get_summary()["error_count"] == 0:
            await debugger.test_api_endpoints()
        else:
            debugger.log("Tests API ignorés à cause d'erreurs critiques", "warning")
    
    # Affichage du résumé
    debugger.print_summary()
    
    # Sauvegarde du rapport
    if args.save_report:
        debugger.save_report(args.save_report)
    elif args.check_all:
        # Sauvegarde automatique pour diagnostic complet
        debugger.save_report()
    
    # Code de sortie selon les résultats
    summary = result.get_summary()
    if summary["overall_status"] == "success":
        sys.exit(0)  # Succès
    elif summary["overall_status"] == "warning":
        sys.exit(1)  # Avertissements
    else:
        sys.exit(2)  # Erreurs critiques


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Diagnostic interrompu par l'utilisateur{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.ERROR} Erreur fatale: {e}{Colors.RESET}")
        sys.exit(1)