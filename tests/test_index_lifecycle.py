import json
from unittest.mock import Mock

import pytest
from backend.app.rag import ingest


@pytest.fixture
def kb(monkeypatch, tmp_path):
    source = tmp_path / 'policies'
    source.mkdir()
    (source / 'policy.txt').write_text('1. First rule\n2. Second rule', encoding='utf-8')
    monkeypatch.setattr(ingest, 'KB_DIR', source)
    monkeypatch.setattr(ingest, 'get_embeddings', lambda: object())
    store = Mock()
    store._collection.count.return_value = 2
    factory = Mock(return_value=store)
    monkeypatch.setattr(ingest.Chroma, 'from_documents', factory)
    return tmp_path / 'index', source, factory


def test_failed_rebuild_preserves_active_manifest(kb):
    directory, source, factory = kb
    ingest.build_index(persist_dir=str(directory))
    previous = (directory / 'active.json').read_bytes()
    factory.side_effect = RuntimeError('embedding failed')
    with pytest.raises(RuntimeError):
        ingest.build_index(persist_dir=str(directory))
    assert (directory / 'active.json').read_bytes() == previous


def test_incomplete_index_is_never_published(kb):
    directory, source, factory = kb
    factory.return_value._collection.count.return_value = 1
    with pytest.raises(RuntimeError):
        ingest.build_index(persist_dir=str(directory))
    assert not (directory / 'active.json').exists()


def test_changed_policies_publish_new_collection(kb):
    directory, source, factory = kb
    first = ingest.active_collection(str(directory))
    assert ingest.active_collection(str(directory)) == first
    assert factory.call_count == 1
    (source / 'policy.txt').write_text('1. Changed rule\n2. Second rule', encoding='utf-8')
    second = ingest.active_collection(str(directory))
    assert first != second
    assert factory.call_count == 2
    assert json.loads((directory / 'active.json').read_text())['collection'] == second
