import pytest

from askmynotes.limits import LimitExceeded, check_file_size, check_token_budget


def test_file_size_under_limit(tmp_path):
    p = tmp_path / "small.pdf"
    p.write_bytes(b"x" * 100)
    assert check_file_size(p, 1000) == 100


def test_file_size_over_limit(tmp_path):
    p = tmp_path / "big.pdf"
    p.write_bytes(b"x" * 2000)
    with pytest.raises(LimitExceeded, match="limit is"):
        check_file_size(p, 1000)


def test_token_budget():
    assert check_token_budget(49_000, 50_000) == 49_000
    with pytest.raises(LimitExceeded):
        check_token_budget(50_001, 50_000)
