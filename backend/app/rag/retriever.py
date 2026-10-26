
"""Recherche sémantique dans l'index Chroma."""

import os
from functools import lru_cache

from langchain_chroma import Chroma

from backend.app.rag.ingest import (
    CHROMA_DIR,
    COLLECTION,
    get_embeddings,
    active_collection,
)

# Calibrated on a text-disjoint split; see outputs/rag_threshold_evaluation.json.
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.40"))
if not 0 <= RELEVANCE_THRESHOLD <= 1:
    raise ValueError("RAG_RELEVANCE_THRESHOLD must be between 0 and 1")


@lru_cache(maxsize=4)
def _get_store(persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION) -> Chroma:
    """Charge la collection Chroma persistée."""
    return Chroma(
        collection_name=collection_name,
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
            persist_dir, active_collection(persist_dir)
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


def expand_policy_context(results: list[dict], preferred_document: str | None = None) -> list[dict]:
    """Include neighboring rules so a semantic hit does not omit procedure conditions."""
    if not results:
        return []
    documents = list(dict.fromkeys(result["document"] for result in results))[:3]
    if preferred_document in documents:
        documents.remove(preferred_document)
        documents.insert(0, preferred_document)
    store = _get_store(CHROMA_DIR, active_collection(CHROMA_DIR))
    data = store.get(where={"document": {"$in": documents}})
    expanded = list(results)
    seen = {(item["document"], item["section"]) for item in results}
    neighbors = list(zip(data["documents"], data["metadatas"]))
    neighbors.sort(key=lambda pair: (documents.index(pair[1]["document"]),
                                    int(pair[1]["section"].split("-")[0])))
    for text, metadata in neighbors:
        key = (metadata["document"], metadata["section"])
        if key not in seen:
            expanded.append({"text": text, **metadata, "score": 0.0,
                             "retrieval_method": "policy_neighbor"})
            seen.add(key)
    return expanded
