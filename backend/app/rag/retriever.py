
"""Recherche sémantique dans l'index Chroma."""

from functools import lru_cache

from langchain_chroma import Chroma

from backend.app.rag.ingest import (
    CHROMA_DIR,
    COLLECTION,
    get_embeddings,
)

# Valeur provisoire : elle devra être calibrée par les expériences.
RELEVANCE_THRESHOLD = 0.35


@lru_cache(maxsize=4)
def _get_store(persist_dir: str = CHROMA_DIR) -> Chroma:
    """Charge la collection Chroma persistée."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def _search(
    query: str,
    k: int = 3,
    persist_dir: str = CHROMA_DIR,
) -> list[dict]:
    """Retourne les k passages les mieux classés, sans appliquer de seuil."""
    if not query or not query.strip():
        return []

    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1")

    try:
        hits = _get_store(
            persist_dir
        ).similarity_search_with_score(query.strip(), k=k)
    except Exception as exc:
        raise RuntimeError(
            f"Échec de la recherche vectorielle : {exc}"
        ) from exc

    results = []

    for doc, distance in hits:
        results.append(
            {
                "text": doc.page_content,
                "document": doc.metadata["document"],
                "section": doc.metadata["section"],
                "score": round(1.0 - float(distance), 4),
            }
        )

    return sorted(
        results,
        key=lambda result: result["score"],
        reverse=True,
    )


def retrieve_context(query: str, k: int = 3) -> dict:
    """Renvoie les passages pertinents selon le contrat commun."""
    if not query or not query.strip():
        return {
            "is_relevant": False,
            "results": [],
        }

    results = [
        result
        for result in _search(query.strip(), k=k)
        if result["score"] >= RELEVANCE_THRESHOLD
    ]

    return {
        "is_relevant": bool(results),
        "results": results,
    }
