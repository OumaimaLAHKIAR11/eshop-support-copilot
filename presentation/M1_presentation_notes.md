# Notes de présentation — M1 (environ 2 minutes)

1. Le dataset contient des tickets synthétiques NovaShop annotés selon trois dimensions : catégorie, sentiment et priorité.
2. La classification utilise un LLM pré-entraîné via LangChain avec une sortie Pydantic structurée. Aucun entraînement de modèle n'est réalisé dans cette version.
3. Le prompt définit les catégories permises, des critères de priorité et le traitement du texte du client comme contenu non fiable.
4. Le jeu de test séparé permet de mesurer l'accord des prédictions avec les annotations : accuracy et macro-F1 pour la catégorie, macro-F1 pour le sentiment, et analyse d'erreurs.
5. Les limites sont la nature synthétique des données, les éventuelles ambiguïtés des annotations et la variabilité des réponses d'un LLM.

À compléter après exécution : nombre de tickets évalués, métriques observées et deux ou trois erreurs réellement constatées. Ne pas inventer de résultats.
