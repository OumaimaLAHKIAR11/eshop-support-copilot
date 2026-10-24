# E-Shop Support Copilot
### Assistant intelligent pour le support client d'une boutique en ligne

**Projet académique — Generative AI | ENSIAS**

## 1. Présentation du projet

E-Shop Support Copilot est un assistant intelligent conçu pour faciliter le traitement des demandes clients d'une boutique en ligne fictive, NovaShop.

Le système analyse les tickets clients, identifie la nature du problème et le sentiment exprimé, recherche les règles pertinentes dans une base de connaissances et prépare une réponse adaptée.

L'objectif est de proposer aux agents du support client une assistance fondée sur les informations disponibles, tout en évitant d'inventer des informations relatives aux commandes, aux livraisons ou aux remboursements.

Le projet repose sur une architecture combinant traitement du langage naturel, recherche augmentée par récupération (RAG), génération de réponses et interface utilisateur.

## 2. Problématique et objectifs

Le traitement des réclamations clients nécessite de comprendre rapidement le problème, d'identifier son niveau de priorité et de fournir une réponse cohérente avec les politiques de la boutique.

Le projet vise donc à :

- Classifier automatiquement les tickets clients.
- Détecter le sentiment exprimé dans les messages.
- Estimer la priorité de traitement.
- Retrouver les informations pertinentes dans les politiques de la boutique.
- Générer une réponse accompagnée de ses sources.
- Signaler les cas nécessitant une vérification humaine.
- Proposer une interface permettant de tester et de visualiser les résultats.

## 3. Architecture générale

Le traitement d'un ticket suit le parcours suivant :

1. Le client ou l'utilisateur saisit un message dans l'interface.
2. Le frontend transmet la demande au backend via une API HTTP.
3. Le module de classification identifie la catégorie, le sentiment et la priorité.
4. Le module RAG recherche les règles pertinentes dans la base de connaissances.
5. Le module de génération prépare une réponse à partir du ticket, de son analyse et des passages récupérés.
6. Le backend renvoie un résultat structuré comprenant l'analyse, la réponse, les sources et l'indicateur de revue humaine.
7. Le frontend affiche les résultats à l'utilisateur.

### Technologies principales

| Technologie | Rôle |
|---|---|
| Python | Développement des modules et du backend |
| LangChain | Intégration des modèles et organisation des traitements |
| Azure OpenAI | Classification des tickets avec le déploiement configuré pour M1 |
| OpenAI API | Fournisseur prévu par le guide commun pour le LLM et les embeddings |
| Pydantic | Validation des données et structuration des réponses |
| Chroma | Index vectoriel utilisé pour la recherche documentaire |
| FastAPI | Exposition des fonctionnalités du backend |
| Streamlit | Interface utilisateur et tableau de bord |
| pytest | Tests automatisés |
| scikit-learn | Évaluation des performances de classification |
| Git et GitHub | Versionnement et collaboration |

**Point d'intégration :** le rapport M1 documente Azure OpenAI, tandis que le guide commun définit une configuration OpenAI pour l'ensemble du projet. Le fournisseur et les variables de configuration doivent être harmonisés dans la version finale afin que la classification, les embeddings et la génération fonctionnent ensemble.

## 4. Réalisation du projet étape par étape

### Étape 1 — Préparation du projet et des données

Nous avons organisé le dépôt GitHub et préparé les ressources nécessaires au traitement des demandes clients.

Les données concernent une boutique fictive et des tickets synthétiques. Les politiques documentaires couvrent quatre domaines principaux :

- `delivery_policy.txt` : livraison.
- `refund_policy.txt` : remboursement.
- `return_policy.txt` : retours.
- `damaged_products_policy.txt` : produits endommagés.

Ces documents constituent la base de référence utilisée pour préparer les réponses.

Les catégories prévues sont `delivery_issue`, `refund_request`, `return_request`, `damaged_product` et `other`.

### Étape 2 — Développement du module de classification (M1)

Le module est situé dans `backend/app/nlp/classification.py`.

La fonction principale est :

```python
analyze_ticket(text: str) -> dict
```

Elle reçoit le texte du ticket et retourne trois informations :

- `category` : catégorie du problème.
- `sentiment` : sentiment positif, neutre ou négatif.
- `priority` : priorité faible, moyenne ou élevée.

L'implémentation documentée utilise Python, Azure OpenAI, LangChain et Pydantic pour obtenir une sortie structurée.

Le prompt a été affiné pour mieux traiter les cas ambigus. Par exemple, un produit défectueux qui est ensuite remboursé doit rester classé comme `damaged_product` si le problème principal est le défaut du produit.

Le module rejette les entrées vides et remonte explicitement les erreurs d'appel au service.

**Évaluation documentée dans le rapport M1 :**

| Indicateur | Résultat rapporté |
|---|---:|
| Accuracy de la catégorie | 100 % |
| Macro-F1 de la catégorie | 1,00 |
| Macro-F1 du sentiment | 0,9034 |
| Accuracy de la priorité | 95 % |
| Macro-F1 de la priorité | 0,9453 |

Ces résultats proviennent d'un jeu de 40 tickets utilisé également pour ajuster le prompt. Ils représentent donc une validation interne et non une mesure indépendante définitive.

Le rapport M1 mentionne également 11 tests réussis lors d'une exécution antérieure. Une nouvelle exécution est nécessaire pour confirmer le résultat dans l'environnement final.

### Étape 3 — Mise en place de la recherche RAG (M2)

Le module RAG permet de retrouver les informations utiles dans les politiques de la boutique.

Les principaux fichiers sont :

- `backend/app/rag/ingest.py` : préparation et indexation des connaissances.
- `backend/app/rag/retriever.py` : recherche des passages pertinents.

L'indexation transforme les documents en représentations numériques, appelées embeddings, puis les stocke dans Chroma. La recherche sémantique permet ensuite de retrouver les passages correspondant au sens de la question.

La fonction de recherche est appelée avec un paramètre `k` qui détermine le nombre maximal de résultats souhaités.

Un test a été exécuté depuis le terminal avec la commande suivante :

```powershell
python -c "from backend.app.rag.retriever import retrieve_context; print(retrieve_context('My package has not arrived yet', k=3))"
```

Le chargement du modèle d'embeddings a été observé pendant l'exécution, ce qui confirme que le composant a été lancé dans cet environnement.

Cette commande vérifie l'exécution du retriever, mais il reste nécessaire d'examiner les résultats retournés pour confirmer que les bons passages de la politique de livraison sont retrouvés.

La validation complète du module doit également vérifier le format commun attendu :

```python
{
    "is_relevant": True,
    "results": [
        {
            "text": "...",
            "document": "delivery_policy.txt",
            "section": "6",
            "score": 0.0
        }
    ]
}
```

Les valeurs ci-dessus illustrent le format attendu et ne constituent pas un résultat réel de test.

Le module doit aussi pouvoir signaler qu'aucun passage suffisamment pertinent n'a été trouvé. Cette abstention empêche le système de générer une réponse non fondée.

### Étape 4 — Génération des réponses et backend (M3)

Le module de génération est prévu dans `backend/app/generation/generation.py`.

Sa fonction principale est :

```python
generate_reply(ticket, analysis, context) -> dict
```

Elle reçoit le ticket initial, son analyse et les résultats de la recherche documentaire.

La réponse attendue comprend :

- `suggested_reply` : réponse proposée au client.
- `sources` : documents et sections utilisés.
- `needs_human_review` : indicateur signalant la nécessité d'une vérification humaine.

Le backend est organisé autour de FastAPI et doit exposer les routes suivantes :

| Méthode | Endpoint | Fonction |
|---|---|---|
| GET | `/health` | Vérifier que le backend répond |
| POST | `/api/v1/analyze` | Classifier un ticket |
| POST | `/api/v1/retrieve` | Rechercher les informations pertinentes |
| POST | `/api/v1/process-ticket` | Exécuter le traitement complet |

Le parcours complet doit connecter les trois modules dans l'ordre suivant : classification, recherche documentaire, puis génération.

Le système doit éviter d'inventer un statut de commande, une date de livraison ou un remboursement confirmé. Lorsque les sources sont insuffisantes, une réponse prudente et une demande de vérification humaine sont préférables.

### Étape 5 — Développement du frontend (M4)

Le frontend est prévu avec Streamlit.

Les fichiers principaux sont :

- `frontend/app.py` : interface et affichage des résultats.
- `frontend/api_client.py` : communication avec le backend.

L'interface doit permettre de saisir un ticket ou de sélectionner un exemple depuis le jeu de données, puis d'afficher la catégorie, le sentiment, la priorité, la réponse proposée et les sources.

Elle doit également présenter un avertissement lorsque le ticket nécessite une revue humaine.

Une vue d'ensemble peut afficher les résultats sur plusieurs tickets ainsi que la répartition des catégories et des sentiments.

Le raccordement au backend réel doit utiliser le contrat JSON commun défini par l'équipe.

### Étape 6 — Résolution et vérification de la connexion frontend/backend

La communication entre Streamlit et FastAPI constitue un point essentiel de l'intégration.

Le principe est que le frontend envoie une requête HTTP à l'endpoint `/api/v1/process-ticket`, puis traite la réponse JSON retournée par le backend.

Les vérifications à effectuer pour résoudre les problèmes de connexion sont les suivantes :

1. Vérifier que le backend démarre sans erreur.
2. Vérifier que l'endpoint `/health` répond.
3. Contrôler l'URL et le port utilisés dans `frontend/api_client.py`.
4. Vérifier que le frontend envoie le JSON attendu par l'API.
5. Vérifier que le backend retourne une réponse au format convenu.
6. Contrôler les délais d'attente et les messages d'erreur réseau.
7. Vérifier les erreurs de configuration des modèles et les exceptions remontées par les modules.
8. Tester le parcours complet depuis l'interface jusqu'à la génération de la réponse.

Pour démarrer le backend conformément au guide commun :

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

La documentation interactive de l'API est ensuite accessible à l'adresse locale :

`http://127.0.0.1:8000/docs`

Si le backend fonctionne mais que le frontend ne parvient pas à l'appeler, il faut vérifier en priorité l'URL configurée côté client, le port utilisé et le format de la requête.

**État des corrections récentes :** les documents fournis ne décrivent pas les modifications précises apportées au code pour résoudre les derniers problèmes de connexion. Cette section présente donc les contrôles de dépannage à documenter, sans affirmer qu'une correction particulière a déjà été appliquée.

### Étape 7 — Tests et validation globale

Les tests sont nécessaires pour vérifier les modules séparément avant de tester leur intégration.

Les fichiers de tests prévus comprennent :

- `tests/test_classification.py`
- `tests/test_retrieval.py`
- `tests/test_api.py`

La suite des tests doit couvrir les entrées valides, les messages vides, les erreurs de service, les demandes sans source pertinente et les réponses de l'API.

La commande générale pour lancer les tests est :

```powershell
python -m pytest -v
```

Les résultats doivent être interprétés à partir de la sortie réelle de la commande. Le nombre de tests réussis précédemment rapporté ne garantit pas que tous les tests passent dans la version intégrée.

## 5. Contrat commun entre les modules

Afin de permettre le travail en parallèle, les modules doivent respecter un format commun.

| Module | Entrée | Sortie principale |
|---|---|---|
| M1 — Classification | Texte du ticket | Catégorie, sentiment, priorité |
| M2 — RAG | Question et nombre de résultats | Pertinence, passages, documents, sections, scores |
| M3 — Génération | Ticket, analyse et contexte | Réponse proposée, sources, revue humaine |
| M4 — Frontend | Ticket saisi par l'utilisateur | Affichage des résultats de l'API |

L'endpoint `/api/v1/process-ticket` doit regrouper ces résultats dans une seule réponse JSON.

Ce contrat commun est essentiel pour éviter les erreurs d'intégration lorsque les modules sont développés séparément.

## 6. Gestion des erreurs et sécurité

Le système doit prévoir des comportements contrôlés pour les situations suivantes :

- Message vide ou invalide.
- Indisponibilité du service de modèle de langage.
- Erreur de connexion entre frontend et backend.
- Absence de source pertinente.
- Réponse incomplète ou invalide.
- Besoin d'informations de commande non disponibles.

La revue humaine doit être déclenchée lorsque les sources sont insuffisantes, lorsque la catégorie est `other`, lorsque des informations de commande sont nécessaires ou lorsqu'un module rencontre une erreur.

Les clés API et les paramètres privés doivent rester dans le fichier local `.env`, exclu de Git. Le fichier `.env.example` doit contenir uniquement des valeurs fictives ou des placeholders.

## 7. Organisation du dépôt

La structure commune définie dans le guide est la suivante :

```text
eshop-support-copilot/
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       │   └── routes.py
│       ├── nlp/
│       │   └── classification.py
│       ├── rag/
│       │   ├── ingest.py
│       │   └── retriever.py
│       └── generation/
│           └── generation.py
├── frontend/
│   ├── app.py
│   └── api_client.py
├── data/
│   ├── tickets.csv
│   ├── test_tickets.csv
│   └── knowledge_base/
├── notebooks/
├── scripts/
├── tests/
├── outputs/
├── presentation/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Cette structure correspond à l'organisation de référence. Les fichiers réellement présents doivent être vérifiés dans le dépôt final.

## 8. Versionnement et collaboration Git

Le projet est développé en plusieurs branches afin de séparer les contributions des membres.

Les branches prévues sont :

- `feature/nlp` : classification et analyse des tickets.
- `feature/rag` : indexation et recherche documentaire.
- `feature/backend` : génération et API.
- `feature/frontend` : interface et évaluation globale.

Le rapport M1 documente le commit `f82f08a` sur `feature/nlp` et une Pull Request ouverte vers `develop`. La fusion n'était pas confirmée au moment de la rédaction du rapport.

Des difficultés de permissions GitHub ont également été signalées lors d'une tentative de push depuis la branche backend. La collaboration nécessite donc de vérifier les droits d'accès au dépôt et de publier les branches avec un compte autorisé.

Les modifications doivent être intégrées via des Pull Requests relues, puis testées dans la branche commune.

## 9. État d'avancement et points à confirmer

| Élément | État documenté |
|---|---|
| Module de classification | Implémentation et évaluation décrites dans le rapport M1 |
| Tests M1 | 11 tests réussis lors d'une exécution antérieure |
| Recherche RAG | Fonction exécutée en ligne de commande |
| Pertinence des résultats RAG | Évaluation complète à confirmer |
| Génération de réponses | Implémentation et fonctionnement bout en bout à vérifier |
| API FastAPI | Routes définies dans le guide commun ; état réel à vérifier |
| Interface Streamlit | Fonctionnement et connexion réelle à confirmer |
| Connexion frontend/backend | Corrections précises récentes non documentées dans les fichiers fournis |
| Intégration des branches | État des Pull Requests et des fusions à vérifier |
| Évaluation complète du pipeline | À confirmer après intégration |

## 10. Perspectives

Les prochaines étapes consistent à finaliser l'intégration entre les modules, confirmer le fonctionnement du frontend et du backend dans le même environnement, vérifier la cohérence des variables de configuration et exécuter les tests de bout en bout.

Il faudra également évaluer la pertinence des passages retrouvés, la qualité des réponses générées, la latence moyenne et le comportement du système lorsque les informations nécessaires ne sont pas disponibles.

## 11. Conclusion

E-Shop Support Copilot met en œuvre les principes de l'intelligence artificielle générative pour assister le service client d'une boutique en ligne.

Les travaux documentés couvrent la classification des tickets, l'évaluation du module NLP, la mise en place d'un composant de recherche sémantique et l'organisation d'une architecture combinant RAG, génération, API et interface utilisateur.

L'objectif final est de disposer d'un système intégré qui analyse les demandes, retrouve les règles pertinentes et propose des réponses explicables et prudentes, tout en conservant la possibilité d'une validation humaine.

**Projet :** E-Shop Support Copilot  
**Module :** Generative AI  
**Établissement :** ENSIAS  
**Dépôt :** `eshop-support-copilot`