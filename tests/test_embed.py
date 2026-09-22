import math

from askmynotes.embed import FakeEmbedder, get_embedder


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder(dim=64)
    a, b = e.embed(["hello"])[0], e.embed(["hello"])[0]
    assert a == b
    assert len(a) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)


def test_different_texts_differ():
    e = FakeEmbedder(dim=64)
    a, b = e.embed(["hello", "goodbye"])
    assert a != b


def test_batch_order_preserved():
    e = FakeEmbedder(dim=32)
    texts = ["a", "b", "c"]
    batched = e.embed(texts)
    assert batched == [e.embed([t])[0] for t in texts]


def test_factory():
    assert get_embedder("fake").model.startswith("fake-")
