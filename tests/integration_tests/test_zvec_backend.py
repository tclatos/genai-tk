"""Real local-vector persistence and retired-Chroma migration regressions."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
import zvec
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.embeddings_store import EmbeddingsStore
from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory
from genai_tk.core.factories.retriever_factory import RetrieverFactory, VectorDocumentStore
from genai_tk.core.vector_backends.zvec import ZvecVectorStore, create_zvec_vector_store

pytestmark = [pytest.mark.integration, pytest.mark.fake_models]


@pytest.fixture
def store_factory(tmp_path: Path, fake_embeddings):
    def create():
        return ZvecVectorStore(
            embedding=fake_embeddings,
            path=str(tmp_path / "index"),
            collection_name="regression",
            dimension=len(fake_embeddings.embed_query("dimension")),
        )

    return create


def test_documents_persist_after_reopening(store_factory) -> None:
    store_factory().add_documents([Document(id="doc1", page_content="persisted", metadata={"source": "fixture"})])
    assert store_factory().get_by_ids(["doc1"]) == [
        Document(id="doc1", page_content="persisted", metadata={"source": "fixture"})
    ]


def test_upsert_updates_existing_document(store_factory) -> None:
    store = store_factory()
    store.add_texts(["before"], ids=["same"])
    store.add_texts(["after"], ids=["same"])
    assert [doc.page_content for doc in store_factory().iter_documents()] == ["after"]


def test_delete_persists_after_reopening(store_factory) -> None:
    store = store_factory()
    store.add_texts(["remove"], ids=["doc1"])
    store.delete(["doc1"])
    assert store_factory().document_count() == 0


def test_clear_preserves_reusable_collection(store_factory) -> None:
    store = store_factory()
    store.add_texts(["remove"], ids=["old"])
    store.delete_collection()
    store_factory().add_texts(["new"], ids=["new"])
    assert [doc.id for doc in store_factory().iter_documents()] == ["new"]


def test_equality_filter_preserves_matches(store_factory) -> None:
    store = store_factory()
    store.add_texts(["one", "two"], ids=["one", "two"], metadatas=[{"group": "a"}, {"group": "b"}])
    assert [doc.id for doc in store.similarity_search("query", k=1, filter={"group": "b"})] == ["two"]


def test_parallel_factories_serialize_native_handles(store_factory) -> None:
    def add(index: int):
        store_factory().add_texts([str(index)], ids=[str(index)])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(add, range(8)))
    assert store_factory().document_count() == 8


def test_existing_non_zvec_directory_is_not_replaced(tmp_path: Path, fake_embeddings) -> None:
    path = tmp_path / "legacy"
    path.mkdir()
    marker = path / "chroma.sqlite3"
    marker.write_bytes(b"existing database")
    with pytest.raises(RuntimeError):
        ZvecVectorStore(
            embedding=fake_embeddings,
            path=str(path),
            collection_name="legacy",
            dimension=len(fake_embeddings.embed_query("dimension")),
        )
    assert marker.read_bytes() == b"existing database"


def test_legacy_chroma_selection_does_not_touch_data(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "chroma.sqlite3"
    marker.write_bytes(b"existing database")
    config = global_config()
    monkeypatch.setattr(config, "root", deepcopy(config.root))
    original = config.get_dict("embeddings_store.local_fast")
    config.set("embeddings_store.local_fast.config.storage", str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="Existing Chroma data has not been opened or changed"):
            EmbeddingsStore.create_from_config("local_fast")
    finally:
        config.set("embeddings_store.local_fast", original)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chroma.sqlite3"]


def test_clear_removes_record_manager_entries(tmp_path: Path, fake_embeddings_id: str, monkeypatch) -> None:
    config = global_config()
    monkeypatch.setattr(config, "root", deepcopy(config.root))
    original = config.get_dict("embeddings_store.zvec")
    old_url = config.get("retrievers.persistent_zvec.record_manager_url", default=None)
    config.set(
        "embeddings_store.zvec",
        {
            "backend": "genai_tk.core.vector_backends.ZvecBackend",
            "embeddings": fake_embeddings_id,
            "config": {"storage": str(tmp_path / "zvec")},
        },
    )
    config.set("retrievers.persistent_zvec.record_manager_url", f"sqlite:///{tmp_path}/records.db")
    try:
        managed = RetrieverFactory.create("persistent_zvec")
        documents = [Document(page_content="re-ingest", metadata={"source": "fixture"})]
        managed.add_documents(documents)
        assert managed.delete_store()
        managed.add_documents(documents)
        reopened = EmbeddingsStore.create_from_config("zvec").get_vector_store()
        assert isinstance(reopened, ZvecVectorStore)
        assert reopened.document_count() == 1
        assert len(next(reopened.iter_documents()).id) == 64
    finally:
        config.set("embeddings_store.zvec", original)
        config.set("retrievers.persistent_zvec.record_manager_url", old_url)


@pytest.mark.parametrize("metric", ["COSINE", "IP", "L2"])
def test_equality_filter_ranks_without_unbounded_native_query(
    tmp_path: Path, fake_embeddings, monkeypatch, metric
) -> None:
    dimension = len(fake_embeddings.embed_query("dimension"))
    query = [1.0] + [0.0] * (dimension - 1)
    far = [0.0, 1.0] + [0.0] * (dimension - 2)
    store = ZvecVectorStore(
        embedding=fake_embeddings,
        path=str(tmp_path / "index"),
        collection_name="filtered",
        dimension=dimension,
        metric=metric,
    )
    collection = zvec.open(store.path)
    try:
        collection.insert(
            [
                zvec.Doc(
                    id="near",
                    vectors={"embedding": query},
                    fields={"page_content": "near", "metadata": '{"group":"included"}'},
                ),
                zvec.Doc(
                    id="far",
                    vectors={"embedding": far},
                    fields={"page_content": "far", "metadata": '{"group":"included"}'},
                ),
            ]
        )
        collection.flush()
    finally:
        collection.close()

    def reject_native_query(*args, **kwargs):
        raise AssertionError("Equality filtering must not request an unbounded native topk")

    monkeypatch.setattr(zvec.Collection, "query", reject_native_query)
    assert [doc.id for doc in store.similarity_search_by_vector(query, k=1, filter={"group": "included"})] == ["near"]


def test_legacy_zvec_helper_keeps_storage_and_collection_name(
    tmp_path: Path, fake_embeddings_id: str, monkeypatch
) -> None:
    config = global_config()
    monkeypatch.setattr(config, "root", deepcopy(config.root))
    config.set("paths.data_root", str(tmp_path))
    factory = EmbeddingsFactory(embeddings=fake_embeddings_id)
    details = {}
    store = create_zvec_vector_store(factory, "legacy_table", {"collection_name": "custom_name"}, details)
    store.add_texts(["preserved"], ids=["doc1"])
    reopened = create_zvec_vector_store(factory, "legacy_table", {"collection_name": "custom_name"}, {})
    assert details == {"path": str(tmp_path / "zvec" / "legacy_table"), "collection_name": "custom_name"}
    assert reopened.get_by_ids(["doc1"])[0].page_content == "preserved"


@pytest.mark.asyncio
async def test_other_backends_keep_existing_index_keys(tmp_path: Path, fake_embeddings) -> None:
    vector_store = InMemoryVectorStore(embedding=fake_embeddings)
    record_manager = SQLRecordManager("compatibility", db_url=f"sqlite:///{tmp_path}/records.db")
    record_manager.create_schema()
    store = VectorDocumentStore(vector_store=vector_store, record_manager=record_manager)
    await store.aadd_documents([Document(page_content="unchanged keys", metadata={"source": "fixture"})])
    assert [len(key) for key in vector_store.store] == [128]


def test_reopened_index_uses_persisted_metric_for_filtered_ranking(tmp_path: Path, fake_embeddings) -> None:
    dimension = len(fake_embeddings.embed_query("dimension"))
    query = [1.0] + [0.0] * (dimension - 1)
    high_dot = [100.0, 10.0] + [0.0] * (dimension - 2)
    path = str(tmp_path / "index")
    ZvecVectorStore(
        embedding=fake_embeddings,
        path=path,
        collection_name="persisted_metric",
        dimension=dimension,
        metric="IP",
    )
    collection = zvec.open(path)
    try:
        collection.insert(
            [
                zvec.Doc(
                    id="high_dot",
                    vectors={"embedding": high_dot},
                    fields={"page_content": "high dot", "metadata": '{"group":"included"}'},
                ),
                zvec.Doc(
                    id="high_cosine",
                    vectors={"embedding": query},
                    fields={"page_content": "high cosine", "metadata": '{"group":"included"}'},
                ),
            ]
        )
        collection.flush()
    finally:
        collection.close()
    # Default COSINE is a creation option, not a migration of the stored IP schema.
    reopened = ZvecVectorStore(
        embedding=fake_embeddings,
        path=path,
        collection_name="persisted_metric",
        dimension=dimension,
    )
    before = reopened.get_by_ids(["high_dot", "high_cosine"])
    native_ids = [doc.id for doc in reopened.similarity_search_by_vector(query, k=1)]
    filtered_ids = [doc.id for doc in reopened.similarity_search_by_vector(query, k=1, filter={"group": "included"})]
    assert native_ids == filtered_ids == ["high_dot"]
    assert reopened.get_by_ids(["high_dot", "high_cosine"]) == before
