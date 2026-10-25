import math
import re
from dataclasses import dataclass

from enterprise_rag.article_chunker import chunk_legal_articles
from enterprise_rag.chunker import chunk_text
from enterprise_rag.embedder import TextEmbedder
from enterprise_rag.pdf_reader import PdfPage
from enterprise_rag.structure_detector import (
    ChunkingStrategy,
    StructureProfile,
    detect_structure,
)

HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+.+|第\s*[一二三四五六七八九十百千万零〇两]+"
    r"\s*[编章节]\s+.*)$",
    re.MULTILINE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")


@dataclass(frozen=True)
class RoutedChunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    strategy: ChunkingStrategy
    section_title: str | None = None
    article_number: str | None = None
    article_title: str | None = None


@dataclass(frozen=True)
class ChunkingResult:
    strategy: ChunkingStrategy
    profile: StructureProfile
    chunks: list[RoutedChunk]


class ChunkingRouter:
    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        chunk_size: int = 500,
        overlap: int = 100,
        semantic_threshold: float = 0.72,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")
        if not 0 <= semantic_threshold <= 1:
            raise ValueError("semantic_threshold 必须在 0 到 1 之间")

        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.semantic_threshold = semantic_threshold

    def route(
        self,
        pages: list[PdfPage],
        strategy: ChunkingStrategy | None = None,
    ) -> ChunkingResult:
        text = "\n\n".join(page.text for page in pages)
        profile = detect_structure(text)
        selected = strategy or profile.strategy
        chunks: list[RoutedChunk] = []

        if selected == ChunkingStrategy.ARTICLE:
            chunks = self._article_chunks(pages)
        elif selected == ChunkingStrategy.HEADING:
            chunks = self._heading_chunks(pages)
        elif selected == ChunkingStrategy.SEMANTIC and self.embedder is not None:
            chunks = self._semantic_chunks(pages)

        if not chunks:
            selected = ChunkingStrategy.FIXED
            chunks = self._fixed_chunks(pages)

        normalized: list[RoutedChunk] = []
        for index, chunk in enumerate(chunks, start=1):
            normalized.append(
                RoutedChunk(
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=index,
                    strategy=selected,
                    section_title=chunk.section_title,
                    article_number=chunk.article_number,
                    article_title=chunk.article_title,
                )
            )

        return ChunkingResult(
            strategy=selected,
            profile=profile,
            chunks=normalized,
        )

    def _article_chunks(self, pages: list[PdfPage]) -> list[RoutedChunk]:
        chunks: list[RoutedChunk] = []

        for article in chunk_legal_articles(pages):
            pieces = chunk_text(
                article.text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
            for piece in pieces:
                chunks.append(
                    RoutedChunk(
                        text=piece.text,
                        page_start=article.page_start,
                        page_end=article.page_end,
                        chunk_index=0,
                        strategy=ChunkingStrategy.ARTICLE,
                        article_number=article.article_number,
                        article_title=article.article_title,
                    )
                )

        return chunks

    def _heading_chunks(self, pages: list[PdfPage]) -> list[RoutedChunk]:
        text, page_ranges = self._joined_pages(pages)
        headings = list(HEADING_PATTERN.finditer(text))
        if not headings:
            return []

        chunks: list[RoutedChunk] = []

        for index, heading in enumerate(headings):
            section_start = heading.start()
            section_end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(text)
            )
            section = text[section_start:section_end].strip()
            if not section:
                continue

            title = heading.group(0).lstrip("#").strip()
            for piece in chunk_text(
                section,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            ):
                global_start = section_start + piece.start_char
                global_end = section_start + piece.end_char - 1
                chunks.append(
                    RoutedChunk(
                        text=piece.text,
                        page_start=self._page_for_offset(
                            page_ranges,
                            global_start,
                        ),
                        page_end=self._page_for_offset(
                            page_ranges,
                            global_end,
                        ),
                        chunk_index=0,
                        strategy=ChunkingStrategy.HEADING,
                        section_title=title,
                    )
                )

        return chunks

    def _semantic_chunks(self, pages: list[PdfPage]) -> list[RoutedChunk]:
        if self.embedder is None:
            return []

        blocks: list[tuple[str, int]] = []
        for page in pages:
            page_blocks = [
                block.strip()
                for block in re.split(r"\n\s*\n", page.text)
                if block.strip()
            ]
            if len(page_blocks) == 1:
                page_blocks = [
                    block.strip()
                    for block in SENTENCE_PATTERN.split(page.text)
                    if block.strip()
                ]

            blocks.extend((block, page.page_number) for block in page_blocks)

        if len(blocks) < 2:
            return []

        vectors = self.embedder.embed_texts([block for block, _ in blocks])
        chunks: list[RoutedChunk] = []
        current_blocks: list[str] = []
        current_start = blocks[0][1]
        current_end = blocks[0][1]

        for index, (block, page_number) in enumerate(blocks):
            if current_blocks:
                combined_length = len("\n".join([*current_blocks, block]))
                similarity = self._cosine(vectors[index - 1], vectors[index])
                if (
                    combined_length > self.chunk_size
                    or similarity < self.semantic_threshold
                ):
                    chunks.append(
                        RoutedChunk(
                            text="\n".join(current_blocks),
                            page_start=current_start,
                            page_end=current_end,
                            chunk_index=0,
                            strategy=ChunkingStrategy.SEMANTIC,
                        )
                    )
                    current_blocks = []

            if not current_blocks:
                current_start = page_number

            current_blocks.append(block)
            current_end = page_number

        if current_blocks:
            chunks.append(
                RoutedChunk(
                    text="\n".join(current_blocks),
                    page_start=current_start,
                    page_end=current_end,
                    chunk_index=0,
                    strategy=ChunkingStrategy.SEMANTIC,
                )
            )

        return chunks

    def _fixed_chunks(self, pages: list[PdfPage]) -> list[RoutedChunk]:
        chunks: list[RoutedChunk] = []

        for page in pages:
            for piece in chunk_text(
                page.text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            ):
                chunks.append(
                    RoutedChunk(
                        text=piece.text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_index=0,
                        strategy=ChunkingStrategy.FIXED,
                    )
                )

        return chunks

    @staticmethod
    def _joined_pages(
        pages: list[PdfPage],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        parts: list[str] = []
        page_ranges: list[tuple[int, int, int]] = []
        cursor = 0

        for page in pages:
            if parts:
                parts.append("\n\n")
                cursor += 2

            start = cursor
            parts.append(page.text)
            cursor += len(page.text)
            page_ranges.append((start, cursor, page.page_number))

        return "".join(parts), page_ranges

    @staticmethod
    def _page_for_offset(
        page_ranges: list[tuple[int, int, int]],
        offset: int,
    ) -> int:
        for start, end, page_number in page_ranges:
            if start <= offset < end:
                return page_number
        return page_ranges[-1][2]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
