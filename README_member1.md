# E-Shop Support Copilot — fichiers M1 (données et classification)

## Ce que contient ce dossier
- `data/tickets.csv` : 200 tickets de développement, avec les étiquettes de référence.
- `data/test_tickets.csv` : 40 tickets réservés à l'évaluation, sans chevauchement avec `tickets.csv`. Le champ `expected_document` est prévu pour le membre RAG.
- `data/source/tickets_uploaded_240.csv` : copie complète des 240 tickets fournis, conservée pour référence. Ne l'utilisez pas directement pour calculer les métriques si vous avez déjà utilisé les fichiers de développement/test séparés.
- `data/knowledge_base/` : quatre politiques fictives pour la recherche RAG.
- `backend/app/nlp/classification.py` : module M1 qui appelle un LLM existant via LangChain; il n'entraîne pas un modèle de zéro.
- `tests/test_classification.py` : tests locaux du schéma et tests d'intégration optionnels.
- `scripts/evaluate_classification.py` : évaluation sur les 40 tickets réservés au test.

Les tickets et les politiques sont synthétiques et concernent la boutique fictive NovaShop.

## Labels autorisés
- `category`: `delivery_issue`, `refund_request`, `return_request`, `damaged_product`, `other`
- `sentiment`: `positive`, `neutral`, `negative`
- `priority`: `low`, `medium`, `high`
- `expected_document`: nom de la politique attendue; vide pour la catégorie `other`.

## Installation (Windows PowerShell ou CMD)
Depuis le dossier `eshop_member1` :

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Ouvre `.env` et remplace `your_openai_api_key_here` par ta clé API OpenAI. Ne partage pas cette clé et ne commite jamais le fichier `.env`. Les appels API peuvent être payants selon ton compte.

## Tester sans appeler l'API
```powershell
pytest -q
```
Les tests d'intégration qui interrogent réellement le LLM sont ignorés par défaut.

## Tester le LLM sur 5 tickets
```powershell
$env:RUN_LLM_TESTS="1"
pytest -q
Remove-Item Env:RUN_LLM_TESTS
```
Cela envoie des requêtes à l'API.

## Évaluer sur 10 tickets d'abord, puis les 40
```powershell
python -m scripts.evaluate_classification --limit 10
python -m scripts.evaluate_classification
```
Ces commandes appellent le LLM et peuvent engendrer des coûts d'API. Les résultats sont sauvegardés dans `outputs/` (ignoré par Git). Ne présentez pas de métriques avant d'avoir réellement exécuté l'évaluation.

## Contrat pour les autres membres
M1 fournit la fonction :
`analyze_ticket(text: str) -> dict`

Exemple de résultat :
```json
{"category": "delivery_issue", "sentiment": "negative", "priority": "high"}
```
M3 peut importer cette fonction et utiliser le dictionnaire retourné. M2 peut utiliser `category` pour vérifier/orienter la recherche documentaire, mais le module de classification ne remplace pas le retriever RAG. Le module ne rédige pas la réponse finale au client.

## Notes importantes
- Le split 200/40 a été effectué de façon stratifiée par catégorie, avec `random_state=42`, et les identifiants des deux fichiers ne se chevauchent pas.
- Les labels sont des annotations synthétiques/manuelles, pas une vérité terrain provenant d'une vraie entreprise. Les scores mesurent l'accord avec ces annotations, pas une garantie de performance en production.
- Le LLM peut se tromper. Vérifiez les erreurs, ajustez le prompt prudemment, puis réévaluez sur le jeu de test sans vous en servir pour ajuster les consignes à répétition.
- Le dossier ne remplace pas le travail des membres M2 (retrieval), M3 (génération/API) et M4 (interface/rapport global).
