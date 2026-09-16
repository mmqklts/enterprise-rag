from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            TextChunk(
                chunk_index=len(chunks) + 1,
                text=text[start:end],
                start_char=start,
                end_char=end,
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks