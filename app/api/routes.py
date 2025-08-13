"""
Router Principal - Chatbot Inclusion Numérique AmazoOn du Web
============================================================

Rassemble tous les endpoints de l'API d'accompagnement numérique pour les femmes
des groupements coopératifs en Afrique de l'Ouest.

Organisation par domaines d'accompagnement:
- /health           → Monitoring et diagnostics système
- /accompagnement   → Conversation et assistance personnalisée  
- /modules          → Gestion des 10 modules de formation
- /audio            → Services vocaux (TTS/STT) - prioritaire pour accessibilité
- /admin            → Administration (suivi progression, analytics)
"""

from fastapi import APIRouter

from app.api.endpoints import health, rag, audio
# from app.api.endpoints import admin  # À créer pour le suivi des utilisatrices

# ═══════════════════════════════════════════════════════════════
# 🏗️ ROUTER PRINCIPAL AMAZOON DU WEB
# ═══════════════════════════════════════════════════════════════

api_router = APIRouter()

# ═══════════════════════════════════════════════════════════════
# 🏥 ROUTES DE SANTÉ ET MONITORING SYSTÈME
# ═══════════════════════════════════════════════════════════════

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Santé & Monitoring Système"],
    responses={
        503: {"description": "Services d'accompagnement indisponibles"},
        500: {"description": "Erreur interne du système"}
    }
)

# ═══════════════════════════════════════════════════════════════
# 🌍 ROUTES D'ACCOMPAGNEMENT NUMÉRIQUE ET MODULES DE FORMATION
# ═══════════════════════════════════════════════════════════════

api_router.include_router(
    rag.router,
    prefix="",  # Pas de préfixe pour les routes principales d'accompagnement
    tags=["Accompagnement Numérique & Formation"],
    responses={
        422: {"description": "Données de demande d'accompagnement invalides"},
        503: {"description": "Assistante numérique indisponible"},
        429: {"description": "Trop de demandes d'accompagnement"}
    }
)

# ═══════════════════════════════════════════════════════════════
# 🎙️ ROUTES AUDIO - PRIORITAIRES POUR ACCESSIBILITÉ
# ═══════════════════════════════════════════════════════════════

api_router.include_router(
    audio.router,
    prefix="/audio",
    tags=["Services Audio & Accessibilité"],
    responses={
        413: {"description": "Fichier audio trop volumineux"},
        415: {"description": "Format audio non supporté"},
        503: {"description": "Services vocaux indisponibles - Impact sur accessibilité"}
    }
)

# ═══════════════════════════════════════════════════════════════
# 🔧 ROUTES D'ADMINISTRATION AMAZOON (FUTURES)
# ═══════════════════════════════════════════════════════════════

# Note: Routes d'administration pour les équipes AmazoOn du Web
# api_router.include_router(
#     admin.router,
#     prefix="/admin",
#     tags=["Administration AmazoOn"],
#     dependencies=[Depends(require_admin_auth)],  # Sécurité équipe AmazoOn
#     responses={
#         401: {"description": "Authentification équipe AmazoOn requise"},
#         403: {"description": "Droits administrateur AmazoOn requis"}
#     }
# )

# ═══════════════════════════════════════════════════════════════
# 📊 MÉTADONNÉES POUR LA DOCUMENTATION OPENAPI
# ═══════════════════════════════════════════════════════════════

# Tags pour organiser la documentation selon les domaines d'accompagnement
TAGS_METADATA = [
    {
        "name": "Santé & Monitoring Système",
        "description": "Vérification de l'état des services d'accompagnement et métriques d'usage par les femmes.",
        "externalDocs": {
            "description": "Guide de monitoring AmazoOn",
            "url": "https://docs.amazoonduweb.com/monitoring"
        }
    },
    {
        "name": "Accompagnement Numérique & Formation", 
        "description": """Cœur du chatbot d'inclusion numérique : conversation personnalisée, 
        recherche dans les 10 modules de formation, ingestion de contenus pédagogiques, 
        et pipeline vocal complet pour l'accessibilité.""",
        "externalDocs": {
            "description": "Guide d'accompagnement numérique",
            "url": "https://docs.amazoonduweb.com/accompagnement"
        }
    },
    {
        "name": "Services Audio & Accessibilité",
        "description": """Services vocaux essentiels : synthèse vocale (TTS) et reconnaissance 
        vocale (STT) adaptés aux femmes peu alphabétisées. Voix bienveillantes, 
        vitesse adaptée, support multilingue évolutif.""",
        "externalDocs": {
            "description": "Guide accessibilité audio",
            "url": "https://docs.amazoonduweb.com/accessibilite"
        }
    },
    {
        "name": "Administration AmazoOn",
        "description": """Outils d'administration pour les équipes AmazoOn du Web : 
        suivi de progression des utilisatrices, analytics d'usage des modules, 
        gestion des contenus pédagogiques, tableaux de bord d'impact.""",
        "externalDocs": {
            "description": "Guide administrateur AmazoOn", 
            "url": "https://docs.amazoonduweb.com/admin"
        }
    }
]

# ═══════════════════════════════════════════════════════════════
# 🌍 ROUTE RACINE DE L'API AMAZOON
# ═══════════════════════════════════════════════════════════════

@api_router.get(
    "/",
    summary="Informations du Chatbot AmazoOn du Web",
    description="Point d'entrée principal de l'API d'inclusion numérique",
    response_description="Informations sur la mission et les services d'accompagnement",
    tags=["Accueil & Présentation"]
)
async def api_amazoon_info():
    """
    Retourne les informations sur le chatbot d'inclusion numérique d'AmazoOn du Web.
    
    Présente la mission, le public cible, les modules de formation disponibles,
    et les fonctionnalités d'accompagnement personnalisé.
    """
    return {
        "organisation": "AmazoOn du Web",
        "mission": "Inclusion numérique des femmes en Afrique de l'Ouest",
        "chatbot": "Assistant d'accompagnement numérique personnalisé",
        "version": "1.0.0",
        "region": "Afrique de l'Ouest (Côte d'Ivoire + sous-région)",
        "public_cible": {
            "principal": "Femmes des groupements coopératifs et associatifs",
            "zones": "Rurales et périurbaines",
            "caracteristiques": [
                "Faible niveau d'alphabétisation ou expérience numérique limitée",
                "Utilisatrices de smartphones basiques",
                "Accès Internet limité",
                "Entrepreneures dynamiques (agriculture, artisanat, commerce)"
            ]
        },
        "objectifs_accompagnement": [
            "Renforcer l'autonomie numérique des femmes",
            "Améliorer la visibilité et promotion des produits locaux",
            "Accompagner la vente locale, régionale et transfrontalière",
            "Encourager bonnes pratiques de gestion commerciale simplifiée",
            "Favoriser confiance et sécurité dans les usages numériques",
            "Créer un espace de motivation continue et progression autonome"
        ],
        "modules_formation": {
            "total": 10,
            "liste": {
                "1": {
                    "titre": "Découverte du smartphone et Internet mobile",
                    "description": "Navigation de base, applications essentielles, connexion"
                },
                "2": {
                    "titre": "WhatsApp pour le commerce",
                    "description": "Messages, statuts, catalogues, groupes clients"
                },
                "3": {
                    "titre": "Facebook et Messenger pour débutantes", 
                    "description": "Profil professionnel, publications, interaction clients"
                },
                "4": {
                    "titre": "Créer une fiche produit attrayante",
                    "description": "Photos, descriptions, prix compétitifs"
                },
                "5": {
                    "titre": "Marketing digital adapté au contexte local",
                    "description": "Promotion contextuelle, visuels, communication authentique"
                },
                "6": {
                    "titre": "Techniques de vente par téléphone et en ligne",
                    "description": "Négociation, service client, fidélisation"
                },
                "7": {
                    "titre": "Introduction à la gestion de micro-entreprise",
                    "description": "Budget simple, suivi ventes, planification"
                },
                "8": {
                    "titre": "Mobile Money et paiements à distance",
                    "description": "Transactions sécurisées, gestion financière mobile"
                },
                "9": {
                    "titre": "Notions de base sur la livraison locale et sous-régionale",
                    "description": "Organisation logistique, partenariats transporteurs"
                },
                "10": {
                    "titre": "Cybersécurité - Protection et confidentialité",
                    "description": "Arnaques, usurpation, bonnes pratiques de sécurité"
                }
            }
        },
        "fonctionnalites_principales": {
            "accompagnement_personnalise": {
                "endpoint": "POST /api/v1/chat",
                "description": "Conversation adaptée au niveau et aux besoins de chaque femme"
            },
            "pipeline_vocal_complet": {
                "endpoint": "POST /api/v1/pipeline", 
                "description": "Audio → Transcription → IA → Synthèse vocale (prioritaire pour non-alphabétisées)"
            },
            "recherche_modules": {
                "endpoint": "POST /api/v1/search",
                "description": "Recherche dans les contenus pédagogiques par module ou compétence"
            },
            "synthese_vocale": {
                "endpoint": "POST /api/v1/audio/tts",
                "description": "Conversion texte → audio avec voix bienveillante et vitesse adaptée"
            },
            "reconnaissance_vocale": {
                "endpoint": "POST /api/v1/audio/stt",
                "description": "Transcription audio → texte avec support langues locales"
            },
            "ingestion_pedagogique": {
                "endpoint": "POST /api/v1/ingest",
                "description": "Ajout de contenus éducatifs audio/visuels à la base"
            }
        },
        "accessibilite": {
            "priorite": "Femmes peu/non alphabétisées",
            "formats_supportes": ["Audio (prioritaire)", "Texte simplifié", "Images", "Vidéos courtes"],
            "navigation": "Mots-clés, boutons, menus guidés",
            "langues": {
                "actuelle": "Français",
                "evolution": "Langues locales (Baoulé, Dioula, etc.)"
            }
        },
        "interfaces_cibles": {
            "prioritaire": "WhatsApp Business",
            "complementaires": ["Messenger", "SMS (zones sans Internet)", "Web/Mobile", "Vocal (IVR)"]
        },
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc", 
            "openapi": "/openapi.json"
        },
        "contact_amazoon": {
            "organisation": "AmazoOn du Web",
            "email": "inclusion@amazoonduweb.com",
            "website": "https://amazoonduweb.com",
            "mission": "Inclusion numérique des femmes en Afrique"
        },
        "metriques_impact": {
            "objectif": "Mesurer l'autonomisation économique par le numérique",
            "indicateurs": [
                "Progression par module",
                "Adoption des outils numériques", 
                "Amélioration des revenus",
                "Élargissement des marchés",
                "Confiance numérique"
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════
# 🎯 ROUTES SPÉCIALISÉES SUPPLÉMENTAIRES (FUTURES)
# ═══════════════════════════════════════════════════════════════

@api_router.get(
    "/modules",
    summary="Liste des 10 modules de formation",
    description="Retourne la liste complète des modules d'inclusion numérique",
    tags=["Accompagnement Numérique & Formation"]
)
async def list_formation_modules():
    """
    Retourne la liste détaillée des 10 modules de formation.
    
    Utilisé pour l'orientation des utilisatrices et la navigation dans l'apprentissage.
    """
    return {
        "total_modules": 10,
        "progression_recommandee": "Progressive selon le niveau de chaque femme",
        "duree_estimee_totale": "3-6 mois selon rythme d'apprentissage",
        "modules": [
            {
                "numero": 1,
                "titre": "Découverte du smartphone et Internet mobile",
                "objectifs": ["Navigation de base", "Applications essentielles", "Connexion Internet"],
                "prerequis": "Aucun - Module d'entrée",
                "duree_estimee": "2-3 semaines",
                "niveau": "Débutant absolu"
            },
            {
                "numero": 2, 
                "titre": "WhatsApp pour le commerce",
                "objectifs": ["Messages commerciaux", "Statuts promotionnels", "Catalogues produits", "Groupes clients"],
                "prerequis": "Module 1 ou usage basique smartphone",
                "duree_estimee": "2-4 semaines", 
                "niveau": "Débutant à intermédiaire"
            },
            {
                "numero": 3,
                "titre": "Facebook et Messenger pour débutantes",
                "objectifs": ["Page professionnelle", "Publications produits", "Interaction clients"],
                "prerequis": "Module 1 + bases WhatsApp",
                "duree_estimee": "3-4 semaines",
                "niveau": "Débutant à intermédiaire"
            },
            {
                "numero": 4,
                "titre": "Créer une fiche produit attrayante", 
                "objectifs": ["Photos de qualité", "Descriptions vendeuses", "Prix compétitifs"],
                "prerequis": "Usage base smartphone (photos)",
                "duree_estimee": "1-2 semaines",
                "niveau": "Tous niveaux"
            },
            {
                "numero": 5,
                "titre": "Marketing digital adapté au contexte local",
                "objectifs": ["Promotion locale", "Visuels attractifs", "Communication authentique"],
                "prerequis": "Modules 2 ou 3 + Module 4",
                "duree_estimee": "3-4 semaines",
                "niveau": "Intermédiaire"
            },
            {
                "numero": 6,
                "titre": "Techniques de vente par téléphone et en ligne",
                "objectifs": ["Négociation", "Service client", "Suivi commandes", "Fidélisation"],
                "prerequis": "Expérience vente + Modules 2 ou 3",
                "duree_estimee": "2-3 semaines", 
                "niveau": "Intermédiaire"
            },
            {
                "numero": 7,
                "titre": "Introduction à la gestion de micro-entreprise",
                "objectifs": ["Budget simplifié", "Suivi des ventes", "Planification"],
                "prerequis": "Expérience commerce + usage numérique de base",
                "duree_estimee": "4-5 semaines",
                "niveau": "Intermédiaire à avancé"
            },
            {
                "numero": 8,
                "titre": "Mobile Money et paiements à distance", 
                "objectifs": ["Transactions sécurisées", "Réception paiements", "Gestion argent mobile"],
                "prerequis": "Confiance usage smartphone + Module cybersécurité recommandé",
                "duree_estimee": "2-3 semaines",
                "niveau": "Intermédiaire"
            },
            {
                "numero": 9,
                "titre": "Notions de base sur la livraison",
                "objectifs": ["Organisation locale", "Partenariats transporteurs", "Livraison sous-régionale"],
                "prerequis": "Expérience vente en ligne (Modules 2, 3, 6)",
                "duree_estimee": "2-3 semaines",
                "niveau": "Intermédiaire à avancé"
            },
            {
                "numero": 10,
                "titre": "Cybersécurité et protection",
                "objectifs": ["Reconnaître arnaques", "Protéger données", "Usage sécurisé"],
                "prerequis": "Usage numérique régulier",
                "duree_estimee": "1-2 semaines",
                "niveau": "Tous niveaux - PRIORITAIRE"
            }
        ],
        "parcours_suggeres": {
            "debutante_complete": [1, 10, 2, 4, 8],
            "commerce_whatsapp": [1, 2, 4, 6, 10],
            "presence_facebook": [1, 3, 4, 5, 10],
            "gestion_avancee": [7, 8, 9, 10, 5],
            "securite_prioritaire": [10, 1, 8, 6, 9]
        }
    }


@api_router.get(
    "/niveaux-experience",
    summary="Niveaux d'expérience et adaptation",
    description="Explique l'adaptation du chatbot selon le niveau d'expérience numérique",
    tags=["Accompagnement Numérique & Formation"]
)
async def experience_levels_info():
    """
    Explique comment le chatbot s'adapte aux différents niveaux d'expérience numérique.
    
    Aide à la compréhension de la personnalisation de l'accompagnement.
    """
    return {
        "principe": "Accompagnement personnalisé selon l'expérience numérique de chaque femme",
        "detection_automatique": "Le chatbot détecte le niveau via les questions et le langage utilisé",
        "niveaux": {
            "debutante": {
                "definition": "Première expérience avec smartphone/Internet ou usage très limité",
                "caracteristiques": [
                    "Smartphone depuis moins de 6 mois ou usage rare",
                    "Difficultés avec navigation de base",
                    "Appréhension face aux nouvelles technologies",
                    "Besoins d'explications très détaillées"
                ],
                "approche_chatbot": {
                    "langage": "Très simple, pas de jargon technique",
                    "rythme": "Très progressif, une étape à la fois", 
                    "encouragement": "Constant et rassurant",
                    "exemples": "Comparaisons avec activités connues (marché, cuisine)",
                    "repetition": "Acceptée et encouragée"
                },
                "modules_recommandes": [1, 10, 2, 4],
                "duree_accompagnement": "6-12 mois"
            },
            "intermediaire": {
                "definition": "Usage régulier smartphone, quelques applications, apprentissage en cours",
                "caracteristiques": [
                    "Utilise WhatsApp, appels, SMS couramment",
                    "Quelques expériences Facebook ou autres apps",
                    "Motivation d'améliorer ses compétences",
                    "Capable d'apprendre de nouvelles fonctions"
                ],
                "approche_chatbot": {
                    "langage": "Accessible mais plus précis",
                    "rythme": "Progression normale avec défis graduels",
                    "encouragement": "Focus sur l'amélioration et l'optimisation", 
                    "exemples": "Cas concrets d'amélioration business",
                    "autonomie": "Encouragée avec support"
                },
                "modules_recommandes": [2, 3, 4, 5, 6, 8],
                "duree_accompagnement": "3-6 mois"
            },
            "avancee": {
                "definition": "Utilise plusieurs outils numériques, cherche à optimiser et développer",
                "caracteristiques": [
                    "Maîtrise WhatsApp, Facebook, peut-être autres plateformes",
                    "Déjà expérience vente en ligne",
                    "Cherche stratégies avancées et optimisation",
                    "Capable de former d'autres femmes"
                ],
                "approche_chatbot": {
                    "langage": "Efficace et stratégique",
                    "rythme": "Accéléré avec conseils avancés",
                    "encouragement": "Focus sur leadership et innovation",
                    "exemples": "Stratégies de croissance et expansion",
                    "role": "Mentor et facilitatrice pour d'autres"
                },
                "modules_recommandes": [5, 6, 7, 9, "formation_formatrice"],
                "duree_accompagnement": "1-3 mois + mentorat continu"
            }
        },
        "adaptation_dynamique": {
            "principe": "Le niveau peut évoluer pendant l'accompagnement",
            "suivi_progression": "Ajustement automatique selon les interactions",
            "validation_niveau": "Questions périodiques pour confirmer l'adaptation"
        },
        "indicateurs_niveau": {
            "debutante": ["je ne sais pas", "comment faire", "première fois", "aide-moi"],
            "intermediaire": ["améliorer", "optimiser", "j'ai déjà essayé", "conseil"],
            "avancee": ["stratégie", "développer", "scaling", "former autres", "partenariat"]
        }
    }


# ═══════════════════════════════════════════════════════════════
# 🎯 INFORMATIONS CONTEXTUELLES AFRIQUE DE L'OUEST
# ═══════════════════════════════════════════════════════════════

@api_router.get(
    "/contexte-regional",
    summary="Contexte régional d'intervention",
    description="Informations sur le contexte d'intervention en Afrique de l'Ouest",
    tags=["Accueil & Présentation"]
)
async def contexte_regional():
    """
    Présente le contexte régional et les spécificités d'intervention d'AmazoOn du Web.
    
    Important pour comprendre l'adaptation culturelle et économique du chatbot.
    """
    return {
        "region_principale": "Afrique de l'Ouest",
        "pays_prioritaire": "Côte d'Ivoire", 
        "extension_prevue": "Sous-région (Mali, Burkina Faso, Ghana, Sénégal, etc.)",
        "contexte_economique": {
            "role_femmes": "Central dans l'économie locale (agriculture, artisanat, transformation, commerce)",
            "defis_actuels": [
                "Accès réduit à l'information",
                "Faible maîtrise du numérique", 
                "Difficultés de mise en marché des produits",
                "Isolement géographique",
                "Alphabétisation limitée"
            ],
            "opportunites_numeriques": [
                "Explosion usage téléphones portables",
                "Adoption WhatsApp, Facebook, Mobile Money",
                "Potentiel d'autonomisation économique",
                "Ouverture nouveaux marchés sous-régionaux"
            ]
        },
        "specificites_techniques": {
            "smartphones": "Basiques à moyens (Android principalement)",
            "connexion": "Internet mobile limitée (3G/4G par intermittence)",
            "applications_populaires": ["WhatsApp", "Facebook", "Orange Money", "MTN MoMo"],
            "contraintes": [
                "Coût de la data",
                "Coupures électriques fréquentes", 
                "Zones avec couverture réseau faible"
            ]
        },
        "adaptation_culturelle": {
            "langues": {
                "officielle": "Français",
                "locales_principales": ["Baoulé", "Dioula", "Bété", "Sénoufo"],
                "strategie_chatbot": "Français adapté + évolution vers langues locales"
            },
            "approche_communication": {
                "ton": "Respectueux, patient, encourageant",
                "exemples": "Contextualisés (marché, agriculture, artisanat local)",
                "valeurs": "Entraide, solidarité, progression collective"
            },
            "considerations_sociales": [
                "Respect des aînées et hiérarchies traditionnelles",
                "Importance du collectif (groupements)",
                "Valorisation des compétences existantes",
                "Prudence face aux innovations (confiance progressive)"
            ]
        },
        "impact_vise": {
            "economique": [
                "Augmentation revenus par élargissement marchés",
                "Réduction intermédiaires commerciaux",
                "Amélioration productivité et gestion",
                "Accès services financiers (Mobile Money)"
            ],
            "social": [
                "Renforcement confiance en soi",
                "Leadership féminin dans communautés",
                "Transmission compétences numériques",
                "Autonomisation et empowerment"
            ],
            "numerique": [
                "Réduction fracture numérique genrée",
                "Inclusion dans économie numérique",
                "Adoption outils adaptés aux besoins",
                "Sécurité et protection en ligne"
            ]
        }
    }