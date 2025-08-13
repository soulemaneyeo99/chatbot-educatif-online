"""
Client OpenAI API pour le Chatbot Éducatif Vocal
===============================================

Interface avec l'API OpenAI GPT pour générer des réponses éducatives
adaptées aux personnes peu alphabétisées.

Utilisation:
    from app.services.openai_client import OpenAIClient
    
    client = OpenAIClient(api_key="sk-...")
    response = await client.generate_response("Explique-moi les fractions")
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
import logging

import openai
from openai import AsyncOpenAI

from app.utils.exceptions import (
    ClaudeAPIError,  # On garde le même nom pour compatibilité
    ClaudeTimeoutError,
    ClaudeQuotaExceededError, 
    ClaudeInvalidModelError,
    ChatbotException
)
from app.utils.logger import log_metrics


# ═══════════════════════════════════════════════════════════════
# 🎯 PROMPTS SYSTÈME POUR L'ÉDUCATION
# ═══════════════════════════════════════════════════════════════

EDUCATION_SYSTEM_PROMPT = """Tu es un assistant éducatif vocal spécialement conçu pour aider les personnes peu alphabétisées à apprendre.

## 🎯 TON RÔLE
- Tu es un éducateur patient, bienveillant et encourageant
- Tu adaptes tes explications au niveau de compréhension de chaque personne
- Tu utilises un langage simple, clair et accessible
- Tu encourages toujours l'apprentissage et la curiosité

## 📚 PRINCIPES PÉDAGOGIQUES
1. **Simplicité** : Utilise des mots simples et des phrases courtes
2. **Analogies** : Utilise des comparaisons avec des choses du quotidien
3. **Répétition** : N'hésite pas à répéter les concepts importants
4. **Encouragement** : Félicite les efforts et les progrès
5. **Patience** : Jamais de jugement, toujours de la bienveillance

## 🗣️ STYLE DE COMMUNICATION
- **Ton familier et chaleureux** (tu peux tutoyer)
- **Phrases courtes** (maximum 15-20 mots)
- **Vocabulaire du quotidien** (évite les termes techniques)
- **Structure claire** : introduction → explication → résumé
- **Questions pour vérifier** la compréhension

## ✅ BONNES PRATIQUES
- Commence par saluer chaleureusement
- Demande le niveau de détail souhaité si nécessaire
- Utilise des exemples concrets et familiers
- Propose des exercices pratiques simples
- Termine par un encouragement

## ❌ À ÉVITER ABSOLUMENT
- Vocabulaire complexe ou technique
- Phrases trop longues ou compliquées
- Ton condescendant ou professoral
- Références culturelles obscures
- Découragement face aux erreurs

## 🎪 EXEMPLES D'APPROCHE
Pour expliquer les fractions :
"Les fractions, c'est comme partager une pizza ! Si tu coupes ta pizza en 4 parts égales et que tu en prends 1, tu as 1/4 de la pizza. Simple, non ?"

Pour la géographie :
"La France, c'est notre pays. Imagine-le comme ta maison, mais en très très grand. Paris, c'est comme le salon principal où habitent beaucoup de personnes."

Réponds TOUJOURS dans cet esprit éducatif, patient et bienveillant."""


RAG_SYSTEM_PROMPT = """Tu es un assistant éducatif qui s'appuie sur des documents pédagogiques pour répondre aux questions.

## 📖 UTILISATION DES DOCUMENTS
Tu as accès à des documents éducatifs qui contiennent des informations fiables et adaptées à ton public.

**QUAND utiliser les documents :**
- La question porte sur un sujet spécifique couvert dans les documents
- Tu as besoin d'informations précises et vérifiées
- L'utilisateur demande des détails sur un cours particulier

**COMMENT utiliser les documents :**
1. Lis attentivement le contenu fourni
2. Adapte le langage à ton public (simplification si nécessaire)
3. Ajoute tes propres explications pour clarifier
4. Ne cite jamais directement les documents (adapte toujours)

**QUAND NE PAS utiliser les documents :**
- Pour des conversations générales
- Pour des encouragements et motivations
- Pour des explications de base que tu maîtrises déjà

## 🎯 ADAPTATION DU CONTENU
Même avec des documents, respecte TOUJOURS les principes :
- Langage simple et accessible
- Explications courtes et claires
- Exemples du quotidien
- Ton bienveillant et encourageant

Si un document est trop complexe, SIMPLIFIE-LE pour ton public."""


# ═══════════════════════════════════════════════════════════════
# 🏗️ CLIENT OPENAI API
# ═══════════════════════════════════════════════════════════════

class OpenAIClient:
    """
    Client pour interagir avec l'API OpenAI GPT.
    
    Gère l'authentification, les prompts éducatifs, la gestion d'erreurs,
    et l'intégration avec le système RAG.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        timeout: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise le client OpenAI.
        
        Args:
            api_key: Clé API OpenAI
            model: Modèle GPT à utiliser
            max_tokens: Nombre maximum de tokens
            temperature: Température de génération (0-1)
            timeout: Timeout en secondes
            logger: Logger optionnel
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation du modèle
        self._validate_model()
        
        # Client OpenAI async
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout
        )
        
        # Statistiques
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "average_response_time": 0.0
        }
        
        self.logger.info(f"🤖 Client OpenAI initialisé - Modèle: {model}")
    
    def _validate_model(self):
        """Valide que le modèle est supporté."""
        supported_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview", 
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4o",
            "gpt-4o-mini"
        ]
        
        if self.model not in supported_models:
            raise ClaudeInvalidModelError(self.model)
    
    def _handle_api_error(self, error: Exception) -> ClaudeAPIError:
        """
        Convertit les erreurs API en exceptions métier.
        
        Args:
            error: Exception originale
            
        Returns:
            ClaudeAPIError: Exception métier appropriée
        """
        error_str = str(error).lower()
        
        # Timeout
        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return ClaudeTimeoutError(self.timeout)
        
        # Quota/Rate limit dépassé
        if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
            return ClaudeQuotaExceededError()
        
        # Modèle invalide
        if "model" in error_str and ("invalid" in error_str or "not found" in error_str):
            return ClaudeInvalidModelError(self.model)
        
        # Clé API invalide
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            return ClaudeAPIError(
                message="Clé API OpenAI invalide",
                details="Vérifiez votre clé API dans la configuration",
                status_code=401
            )
        
        # Erreur générique
        return ClaudeAPIError(
            message="Erreur de l'assistant IA",
            details=str(error)
        )
    
    def _update_stats(self, success: bool, response_time: float, tokens_used: int = 0):
        """Met à jour les statistiques d'usage."""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            self.stats["total_tokens_used"] += tokens_used
            
            # Moyenne glissante du temps de réponse
            current_avg = self.stats["average_response_time"]
            total_requests = self.stats["total_requests"]
            self.stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        else:
            self.stats["failed_requests"] += 1
        
        # Métriques globales
        log_metrics.log_claude_call()  # On garde le même nom pour la compatibilité
    
    async def generate_response(
        self,
        user_message: str,
        context_documents: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse éducative à partir d'un message utilisateur.
        
        Args:
            user_message: Message de l'utilisateur
            context_documents: Documents RAG pour le contexte
            conversation_history: Historique de la conversation
            system_prompt_override: Prompt système personnalisé
            
        Returns:
            Dict: Réponse avec métadonnées
            
        Raises:
            ClaudeAPIError: Si erreur lors de l'appel API
        """
        start_time = time.time()
        
        try:
            # Construction du prompt système
            system_prompt = system_prompt_override or self._build_system_prompt(context_documents)
            
            # Construction des messages
            messages = self._build_messages(user_message, conversation_history, context_documents, system_prompt)
            
            self.logger.debug(f"🔵 Appel OpenAI - Modèle: {self.model}, Messages: {len(messages)}")
            
            # Appel à l'API OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            # Extraction de la réponse
            response_text = response.choices[0].message.content if response.choices else ""
            
            # Calcul des métriques
            response_time = (time.time() - start_time) * 1000  # ms
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Mise à jour des stats
            self._update_stats(True, response_time, tokens_used)
            
            # Logging
            self.logger.info(
                f"✅ Réponse OpenAI générée - "
                f"Tokens: {tokens_used}, Temps: {response_time:.0f}ms"
            )
            
            # Construction de la réponse enrichie
            result = {
                "response": response_text,
                "model": self.model,
                "tokens_used": tokens_used,
                "response_time_ms": response_time,
                "has_context": bool(context_documents),
                "context_documents_count": len(context_documents) if context_documents else 0,
                "metadata": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                    "generated_at": time.time()
                }
            }
            
            return result
            
        except asyncio.TimeoutError:
            self._update_stats(False, (time.time() - start_time) * 1000)
            raise ClaudeTimeoutError(self.timeout)
            
        except Exception as e:
            self._update_stats(False, (time.time() - start_time) * 1000)
            self.logger.error(f"❌ Erreur OpenAI API: {e}")
            raise self._handle_api_error(e)
    
    def _build_system_prompt(self, context_documents: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Construit le prompt système selon le contexte.
        
        Args:
            context_documents: Documents RAG optionnels
            
        Returns:
            str: Prompt système complet
        """
        if context_documents:
            return RAG_SYSTEM_PROMPT
        else:
            return EDUCATION_SYSTEM_PROMPT
    
    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_documents: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = ""
    ) -> List[Dict[str, str]]:
        """
        Construit la liste des messages pour OpenAI.
        
        Args:
            user_message: Message utilisateur actuel  
            conversation_history: Historique des échanges
            context_documents: Documents de contexte RAG
            system_prompt: Prompt système
            
        Returns:
            List: Messages formatés pour l'API OpenAI
        """
        messages = []
        
        # Message système (toujours en premier)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Ajout de l'historique de conversation
        if conversation_history:
            for exchange in conversation_history[-5:]:  # Garde seulement les 5 derniers échanges
                if exchange.get("user"):
                    messages.append({
                        "role": "user",
                        "content": exchange["user"]
                    })
                if exchange.get("assistant"):
                    messages.append({
                        "role": "assistant", 
                        "content": exchange["assistant"]
                    })
        
        # Construction du message utilisateur actuel
        current_message = user_message
        
        # Ajout du contexte RAG si disponible
        if context_documents:
            context_text = self._format_context_documents(context_documents)
            current_message = f"""Contexte éducatif disponible :
{context_text}

Question de l'utilisateur : {user_message}

Réponds en utilisant le contexte ci-dessus si pertinent, sinon réponds avec tes connaissances générales. Adapte toujours ton langage pour des personnes peu alphabétisées."""
        
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _format_context_documents(self, context_documents: List[Dict[str, Any]]) -> str:
        """
        Formate les documents de contexte pour le prompt.
        
        Args:
            context_documents: Documents RAG avec scores
            
        Returns:
            str: Contexte formaté
        """
        if not context_documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(context_documents[:3], 1):  # Max 3 documents
            content = doc.get("content", "").strip()
            score = doc.get("score", 0.0)
            
            if content:
                context_parts.append(f"Document {i} (pertinence: {score:.2f}) :\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def generate_simple_response(self, user_message: str) -> str:
        """
        Génère une réponse simple sans contexte RAG.
        
        Args:
            user_message: Message de l'utilisateur
            
        Returns:
            str: Réponse générée
        """
        result = await self.generate_response(user_message)
        return result["response"]
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion à l'API OpenAI.
        
        Returns:
            Dict: Résultat du test
        """
        try:
            start_time = time.time()
            
            # Test simple avec un message minimal
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant de test."},
                    {"role": "user", "content": "Dis juste 'Test OK'"}
                ],
                max_tokens=10,
                temperature=0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "model": self.model,
                "response_time_ms": response_time,
                "response": response.choices[0].message.content if response.choices else "",
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'usage du client.
        
        Returns:
            Dict: Statistiques détaillées
        """
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ) * 100,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "average_response_time": 0.0
        }
        self.logger.info("📊 Statistiques remises à zéro")
    
    async def close(self):
        """Ferme proprement le client."""
        try:
            await self.client.close()
            self.logger.info("🔒 Client OpenAI fermé")
        except Exception as e:
            self.logger.warning(f"Erreur lors de la fermeture: {e}")


# ═══════════════════════════════════════════════════════════════
# 🧪 FONCTIONS DE TEST
# ═══════════════════════════════════════════════════════════════

async def test_openai_client():
    """Teste le client OpenAI avec différents scénarios."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY manquante dans l'environnement")
        return
    
    print("🧪 Test du client OpenAI...")
    
    client = OpenAIClient(
        api_key=api_key,
        model="gpt-3.5-turbo",
        max_tokens=100
    )
    
    # Test de connexion
    print("\n1. Test de connexion...")
    connection_test = await client.test_connection()
    if connection_test["success"]:
        print(f"✅ Connexion OK - {connection_test['response']}")
    else:
        print(f"❌ Connexion échouée: {connection_test['error']}")
        return
    
    # Test de réponse simple
    print("\n2. Test de réponse éducative...")
    try:
        response = await client.generate_simple_response(
            "Explique-moi ce que c'est qu'une addition"
        )
        print(f"✅ Réponse: {response[:100]}...")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    # Test avec contexte RAG
    print("\n3. Test avec contexte RAG...")
    mock_documents = [
        {
            "content": "L'addition est une opération qui permet de rassembler plusieurs nombres pour en obtenir un seul, appelé somme.",
            "score": 0.95
        }
    ]
    
    try:
        result = await client.generate_response(
            "Comment faire une addition ?",
            context_documents=mock_documents
        )
        print(f"✅ Réponse avec RAG: {result['response'][:100]}...")
        print(f"   Tokens utilisés: {result['tokens_used']}")
    except Exception as e:
        print(f"❌ Erreur RAG: {e}")
    
    # Statistiques
    print("\n4. Statistiques:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Fermeture
    await client.close()
    print("\n✅ Test terminé!")


if __name__ == "__main__":
    asyncio.run(test_openai_client())