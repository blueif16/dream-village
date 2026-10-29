"""memobench.embed (gemini-embedding-2, addendum §5) and its wiring into Memora's store (port-plan 1b, 1c).

The genai client is a hand-written fake that behaves like the real SDK for gemini-embedding-2:
a list of `types.Content` yields one embedding per Content, while a bare list of strings is merged
into ONE Content and yields ONE embedding (the silent-failure trap). No network calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from google.genai import errors, types
from omegaconf import OmegaConf

import memobench.embed as embed_mod
from memobench.embed import Embedder

REPO = Path(__file__).resolve().parent.parent
MEMORA_SRC = REPO / "vendor" / "Memora" / "src"
if str(MEMORA_SRC) not in sys.path:
    sys.path.insert(0, str(MEMORA_SRC))

DIMS = 3072
DOC = "title: none | text: {}"
QUERY = "task: question answering | query: {}"
CONSOL = "task: sentence similarity | query: {}"


def fake_vector(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [(h[i % 32] + 1) / 256.0 for i in range(DIMS)]


class FakeGenai:
    """`client.models.embed_content(model=, contents=, config=)` like google-genai 2.x on the Gemini API."""

    def __init__(self, *, combine_always: bool = False, failures: list[Exception] | None = None):
        self.calls: list[list[str]] = []  # texts of each request, one entry per Content
        self.model_names: list[str] = []
        self.combine_always = combine_always
        self.failures = list(failures or [])
        self.models = self

    def embed_content(self, *, model, contents, config=None):
        self.model_names.append(model)
        if self.failures:
            raise self.failures.pop(0)
        if all(isinstance(c, types.Content) for c in contents):
            texts = ["".join(p.text for p in c.parts) for c in contents]
        else:  # bare strings: the SDK merges them into one Content -> one embedding
            texts = ["".join(contents)]
        self.calls.append(texts)
        if self.combine_always:
            texts = ["".join(texts)]
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=fake_vector(t)) for t in texts])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def fake():
    return FakeGenai()


@pytest.fixture
def embedder(tmp_path, fake):
    return Embedder(root=tmp_path, run_id="r1", client=fake, sleep=lambda s: None)


# --------------------------------------------------------------------------- embed()


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("consolidation", "task: sentence similarity | query: Alice likes tea"),
        ("document", "title: none | text: Alice likes tea"),
        ("query", "task: question answering | query: Alice likes tea"),
    ],
)
def test_each_kind_gets_its_documented_prefix(embedder, fake, kind, expected):
    embedder.embed(["Alice likes tea"], kind)

    assert fake.calls == [[expected]]
    assert fake.model_names == ["gemini-embedding-2"]


def test_returns_one_vector_per_input_in_input_order(embedder):
    out = embedder.embed(["a", "b", "c"], "document")

    assert out == [fake_vector(DOC.format(t)) for t in ("a", "b", "c")]
    assert all(len(v) == DIMS for v in out)


def test_raises_when_api_returns_fewer_vectors_than_inputs(tmp_path):
    e = Embedder(root=tmp_path, client=FakeGenai(combine_always=True), sleep=lambda s: None)

    with pytest.raises(embed_mod.EmbeddingCountError):
        e.embed(["a", "b", "c"], "document")


def test_cache_hit_on_a_new_instance_makes_zero_api_calls(tmp_path, embedder):
    first = embedder.embed(["a", "b"], "query")
    fresh = FakeGenai()
    again = Embedder(root=tmp_path, client=fresh, sleep=lambda s: None).embed(["b", "a"], "query")

    assert fresh.calls == []
    assert again == [first[1], first[0]]


def test_same_text_with_another_kind_is_a_cache_miss(embedder, fake):
    embedder.embed(["a"], "query")
    embedder.embed(["a"], "consolidation")

    assert fake.calls == [[QUERY.format("a")], [CONSOL.format("a")]]


def test_more_than_100_texts_are_sent_in_chunks_of_at_most_100(embedder, fake):
    texts = [f"t{i:03d}" for i in range(250)]

    out = embedder.embed(texts, "document")

    assert [len(c) for c in fake.calls] == [100, 100, 50]
    assert out == [fake_vector(DOC.format(t)) for t in texts]


def test_each_api_call_appends_one_embed_ledger_line_priced_at_chars_over_4(tmp_path, embedder):
    texts = [f"t{i:03d}" for i in range(250)]  # "title: none | text: t000" = 24 chars each

    embedder.embed(texts, "document")

    lines = read_jsonl(tmp_path / "ledger.jsonl")
    assert [l["stage"] for l in lines] == ["embed"] * 3
    assert all(l["run_id"] == "r1" for l in lines)
    expected = [n * 24 / 4 * 0.20 / 1e6 for n in (100, 100, 50)]
    assert [l["cost_usd"] for l in lines] == pytest.approx(expected)


def test_429_and_5xx_are_retried_then_succeed(tmp_path):
    slept: list[float] = []
    flaky = FakeGenai(
        failures=[
            errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
            errors.ServerError(503, {"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}}),
        ]
    )
    e = Embedder(root=tmp_path, client=flaky, sleep=slept.append)

    out = e.embed(["a"], "document")

    assert out == [fake_vector(DOC.format("a"))]
    assert len(slept) == 2 and all(s > 0 for s in slept)
    assert len(read_jsonl(tmp_path / "logs" / "retries.jsonl")) == 2


def test_400_is_not_retried(tmp_path):
    bad = FakeGenai(failures=[errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})])
    e = Embedder(root=tmp_path, client=bad, sleep=lambda s: None)

    with pytest.raises(errors.ClientError):
        e.embed(["a"], "document")
    assert len(bad.model_names) == 1


# --------------------------------------------------------------------------- Memora store wiring


USER = "user1"
RETRIEVAL = f"agent_memory_{USER}"
CONSOLIDATION = f"agent_memory_{USER}__consol"


def memora_cfg(tmp_path: Path, embed_field: str = "index"):
    return OmegaConf.create(
        {
            "llm": {"model": "claude-haiku-5-5", "seed": 42},
            "openai": {"api_type": "azure", "embedding_model": "text-embedding-3-small"},
            "memory": {
                "persist_path": str(tmp_path / "store"),
                "distance": "cosine",
                "collection_name": "agent_memory",
                "query_score_threshold": -1.0,  # fake vectors: keep every hit
                "update_score_threshold": -1.0,
                "enable_cue_index": True,
                "cue_top_k": 20,
                "multimodal_support": False,
                "embed_field": embed_field,
            },
        }
    )


@pytest.fixture
def store_fake(monkeypatch, tmp_path):
    fake = FakeGenai()
    monkeypatch.setattr(embed_mod, "_default", Embedder(root=tmp_path / "runs", client=fake, sleep=lambda s: None))
    return fake


@pytest.fixture
def queried(monkeypatch):
    """Names of the Chroma collections hit by vector queries, in order."""
    from memora.db_clients.chromadb_client import ChromaDBClient

    names: list[str] = []
    original = ChromaDBClient.query

    def spy(self, collection, *args, **kwargs):
        names.append(collection.name)
        return original(self, collection, *args, **kwargs)

    monkeypatch.setattr(ChromaDBClient, "query", spy)
    return names


def make_memory(cfg):
    from memora.core.memory import AgentMemory

    return AgentMemory(cfg, user_id=USER)


def collection_ids(cfg, name: str) -> set[str]:
    from memora.db_clients.chromadb_client import ChromaDBClient

    client = ChromaDBClient._get_or_create_client(cfg.memory.persist_path)
    return set(client.get_collection(name).get()["ids"])


def factual(index: str, value: str, cues: str = "", creation_time: str = "t0"):
    from memora.core.memory_entry import MemoryEntry

    return MemoryEntry(memory_type="factual", index=index, value=value, cue_indices=cues, creation_time=creation_time)


def test_factual_entry_goes_to_both_collections_cues_and_episodes_to_retrieval_only(tmp_path, store_fake):
    from memora.core.memory_entry import MemoryEntry
    from memora.utils.misc import index_to_id

    cfg = memora_cfg(tmp_path)
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea.", cues="green tea habit"))
    mem.add(MemoryEntry(memory_type="episodic", index="[EPISODIC] tea chat (segment 0)", value="A: tea?"))

    fact_id = index_to_id("Alice's beverage preference")
    assert collection_ids(cfg, RETRIEVAL) == {
        fact_id,
        index_to_id("green tea habit"),
        index_to_id("[EPISODIC] tea chat (segment 0)"),
    }
    assert collection_ids(cfg, CONSOLIDATION) == {fact_id}
    embedded = [t for call in store_fake.calls for t in call]
    assert DOC.format("Alice's beverage preference") in embedded
    assert CONSOL.format("Alice's beverage preference") in embedded
    assert CONSOL.format("green tea habit") not in embedded


def test_delete_removes_a_factual_entry_from_both_collections(tmp_path, store_fake):
    cfg = memora_cfg(tmp_path)
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea."))
    mem.add(factual("Bob's instrument", "Bob plays guitar."))

    mem.delete("Alice's beverage preference")

    from memora.utils.misc import index_to_id

    assert collection_ids(cfg, RETRIEVAL) == {index_to_id("Bob's instrument")}
    assert collection_ids(cfg, CONSOLIDATION) == {index_to_id("Bob's instrument")}


def test_retrieval_query_hits_retrieval_collection_with_query_prefix(tmp_path, store_fake, queried):
    cfg = memora_cfg(tmp_path)
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea."))

    hits = mem.query("What does Alice drink?", top_k=5, enhance_query=False)

    assert queried == [RETRIEVAL]
    assert store_fake.calls[-1] == [QUERY.format("What does Alice drink?")]
    assert [h.index for h in hits] == ["Alice's beverage preference"]


def test_consolidation_candidate_search_hits_consolidation_collection(tmp_path, store_fake, queried):
    from memora.builder.chat_memory_builder import ChatMemoryBuilder
    from memora.utils.llm import ChatCompletionModel

    cfg = memora_cfg(tmp_path)
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea.", creation_time="t0"))
    builder = ChatMemoryBuilder(cfg, mem, ChatCompletionModel(cfg))

    candidates = builder._query_update_candidates(factual("Alice's favorite drink", "Alice loves tea.", creation_time="t1"))

    assert queried == [CONSOLIDATION]
    assert store_fake.calls[-1] == [CONSOL.format("Alice's favorite drink")]
    assert [c.index for c in candidates] == ["Alice's beverage preference"]


# --------------------------------------------------------------------------- 1c: memory.embed_field


def test_embed_field_value_embeds_the_value_in_both_collections(tmp_path, store_fake):
    cfg = memora_cfg(tmp_path, embed_field="value")
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea.", cues="green tea habit"))

    embedded = [t for call in store_fake.calls for t in call]
    assert DOC.format("Alice likes green tea.") in embedded
    assert CONSOL.format("Alice likes green tea.") in embedded
    assert not any("Alice's beverage preference" in t for t in embedded)
    assert DOC.format("green tea habit") in embedded  # cue anchors keep their own text


def test_embed_field_value_consolidation_query_embeds_the_new_value(tmp_path, store_fake, queried):
    from memora.builder.chat_memory_builder import ChatMemoryBuilder
    from memora.utils.llm import ChatCompletionModel

    cfg = memora_cfg(tmp_path, embed_field="value")
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea.", creation_time="t0"))
    builder = ChatMemoryBuilder(cfg, mem, ChatCompletionModel(cfg))

    builder._query_update_candidates(factual("Alice's favorite drink", "Alice loves tea.", creation_time="t1"))

    assert queried == [CONSOLIDATION]
    assert store_fake.calls[-1] == [CONSOL.format("Alice loves tea.")]


@pytest.mark.parametrize("embed_field, index_word_scores", [("index", True), ("value", False)])
def test_bm25_text_of_a_factual_entry_follows_embed_field(tmp_path, store_fake, embed_field, index_word_scores):
    cfg = memora_cfg(tmp_path, embed_field=embed_field)
    mem = make_memory(cfg)
    mem.add(factual("Alice's beverage preference", "Alice likes green tea."))
    mem.add(factual("Bob's instrument", "Bob plays guitar."))
    mem.add(factual("Carol's trip", "Carol visited Paris."))

    hits = mem._store.bm25_search("beverage", top_k=3)  # "beverage" appears only in Alice's index

    score = {h.index: h.score for h in hits}.get("Alice's beverage preference", 0.0)
    assert (score > 0) is index_word_scores


def test_real_gemini_client_has_a_finite_timeout(tmp_path, monkeypatch):
    # a Gemini read with no timeout hung two runs for an hour (threads parked in SSL_read)
    import google.genai as genai

    made = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            made.update(kwargs)

    monkeypatch.setattr(genai, "Client", RecordingClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    Embedder(root=tmp_path).client
    opts = made.get("http_options")
    timeout = getattr(opts, "timeout", None) if opts is not None else None
    assert timeout is not None and 0 < timeout <= 120_000  # google-genai timeout is in milliseconds


def test_read_timeout_is_retried_then_succeeds(tmp_path):
    import httpx

    class TimesOutOnce(FakeGenai):
        def __init__(self):
            super().__init__()
            self.failed = False

    client = TimesOutOnce()
    real = client.models.embed_content

    def flaky(**kw):
        if not client.failed:
            client.failed = True
            raise httpx.ReadTimeout("read timed out")
        return real(**kw)

    client.models.embed_content = flaky
    e = Embedder(root=tmp_path, client=client, sleep=lambda s: None)
    assert len(e.embed(["a", "b"], "document")) == 2
    assert client.failed
