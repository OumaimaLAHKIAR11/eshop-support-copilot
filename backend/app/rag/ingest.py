
"""Construction de l'index Chroma à partir des politiques NovaShop."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Chemins relatifs à la racine du projet
KB_DIR = Path("data/knowledge_base")
CHROMA_DIR = "chroma_db"
COLLECTION = "eshop_support"

RULE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def parse_rules(path: Path) -> list[tuple[int, str]]:
    """Extrait les règles numérotées, y compris les règles multilignes."""
    rules = []
    current_num = None
    buffer = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = RULE_RE.match(line)

        if match:
            if current_num is not None:
                rules.append(
                    (current_num, " ".join(buffer).strip())
                )

            current_num = int(match.group(1))
            buffer = [match.group(2).strip()]

        elif current_num is not None and line.strip():
            buffer.append(line.strip())

    if current_num is not None:
        rules.append(
            (current_num, " ".join(buffer).strip())
        )

    return rules


def build_chunks(group_size: int = 1) -> list[Document]:
    """Transforme les règles en chunks avec leurs métadonnées."""
    if group_size < 1:
        raise ValueError("group_size doit être supérieur ou égal à 1")

    if not KB_DIR.is_dir():
        raise FileNotFoundError(
            f"Base de connaissances introuvable : {KB_DIR}"
        )

    paths = sorted(KB_DIR.glob("*.txt"))

    if not paths:
        raise ValueError(
            f"Aucun fichier .txt trouvé dans {KB_DIR}"
        )

    documents = []

    for path in paths:
        rules = parse_rules(path)

        if not rules:
            raise ValueError(
                f"Aucune règle numérotée dans {path.name}"
            )

        topic = path.stem.replace("_", " ")

        for i in range(0, len(rules), group_size):
            group = rules[i:i + group_size]
            first = group[0][0]
            last = group[-1][0]

            section = (
                str(first) if first == last
                else f"{first}-{last}"
            )

            content = f"{topic}: " + " ".join(
                text for _, text in group
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "document": path.name,
                        "section": section,
                    },
                )
            )

    return documents


def build_index(
    group_size: int = 1,
    persist_dir: str = CHROMA_DIR,
) -> int:
    """Reconstruit la collection Chroma sans conserver les anciens chunks."""
    docs = build_chunks(group_size)
    embeddings = get_embeddings()

    # Supprimer l'ancienne collection si elle existe.
    existing = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    try:
        existing.delete_collection()
    except Exception:
        # Une collection inexistante peut être normale au premier lancement.
        pass

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return len(docs)


if __name__ == "__main__":
    count = build_index()
    print(
        f"Index construit : {count} chunks "
        f"dans '{CHROMA_DIR}/' ({COLLECTION})"
    )
