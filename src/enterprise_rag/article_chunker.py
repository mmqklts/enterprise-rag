import re
from dataclasses import dataclass

from enterprise_rag.pdf_reader import PdfPage

ARTICLE_HEADING_PATTERN = re.compile(
    r"第\s*(?P<number>[一二三四五六七八九十百千万零〇两]+)\s*条"
    r"\s*【(?P<title>[^】]{1,100})】"
)


@dataclass(frozen=True)
class ArticleChunk:
    article_number: str
    article_title: str
    text: str
    page_start: int
    page_end: int


def _page_for_offset(
    page_ranges: list[tuple[int, int, int]],
    offset: int,
) -> int:
    for start, end, page_number in page_ranges:
        if start <= offset < end:
            return page_number

    return page_ranges[-1][2]


def chunk_legal_articles(pages: list[PdfPage]) -> list[ArticleChunk]:
    if not pages:
        return []

    parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        if parts:
            parts.append("\n")
            cursor += 1

        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        page_ranges.append((start, cursor, page.page_number))

    document_text = "".join(parts)
    headings = list(ARTICLE_HEADING_PATTERN.finditer(document_text))
    articles: list[ArticleChunk] = []

    for index, heading in enumerate(headings):
        section_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(document_text)
        )
        section = document_text[heading.start() : section_end].strip()
        if not section:
            continue

        page_start = _page_for_offset(page_ranges, heading.start())
        page_end = _page_for_offset(
            page_ranges,
            heading.start() + len(section) - 1,
        )

        articles.append(
            ArticleChunk(
                article_number=f"第{heading.group('number')}条",
                article_title=heading.group("title").strip(),
                text=section,
                page_start=page_start,
                page_end=page_end,
            )
        )

    return articles
