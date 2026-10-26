
"""Construction de l'index Chroma à partir des politiques NovaShop."""

import os
import re
import hashlib
import json
import uuid
from functools import lru_cache
from filelock import FileLock
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


@lru_cache(maxsize=1)
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


def knowledge_fingerprint() -> str:
    paths = sorted(KB_DIR.glob("*.txt"))
    if not paths:
        raise ValueError("No knowledge-base policy files found")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _publish_index(group_size: int, persist_dir: str) -> int:
    fingerprint = knowledge_fingerprint()
    docs = build_chunks(group_size)
    # Readers continue using the previous immutable collection during construction.
    name = COLLECTION + "_" + uuid.uuid4().hex
    store = Chroma.from_documents(
        documents=docs, embedding=get_embeddings(), collection_name=name,
        persist_directory=persist_dir, collection_metadata={"hnsw:space": "cosine"},
    )
    if store._collection.count() != len(docs):
        raise RuntimeError("Incomplete index; previous collection remains active")
    if fingerprint != knowledge_fingerprint():
        raise RuntimeError("Policies changed during indexing; retry the build")
    directory = Path(persist_dir)
    manifest = {"collection": name, "fingerprint": fingerprint, "group_size": group_size,
                "chunks": len(docs)}
    temporary = directory / ("active-" + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(temporary, directory / "active.json")
    return len(docs)


def build_index(group_size: int = 1, persist_dir: str = CHROMA_DIR) -> int:
    """Build a new collection and atomically publish it only after validation."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(persist_dir) / "index.lock"), timeout=120):
        return _publish_index(group_size, persist_dir)


def active_collection(persist_dir: str = CHROMA_DIR) -> str:
    """Refresh changed policies under a process-safe lock; never use stale evidence."""
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "index.lock"), timeout=120):
        manifest_path = directory / "active.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if manifest.get("fingerprint") != knowledge_fingerprint():
            _publish_index(manifest.get("group_size", 1), persist_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["collection"]


if __name__ == "__main__":
    count = build_index()
    print(
        f"Index construit : {count} chunks "
        f"dans '{CHROMA_DIR}/' ({COLLECTION})"
    )
