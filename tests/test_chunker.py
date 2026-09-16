import pytest

from enterprise_rag.chunker import chunk_text


def test_chunk_text_with_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert chunks[1].start_char == 4


def test_short_text_returns_one_chunk() -> None:
    chunks = chunk_text("短文本", chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "短文本"


def test_invalid_overlap_raises_error() -> None:
    with pytest.raises(ValueError):
        chunk_text("abcdef", chunk_size=5, overlap=5)