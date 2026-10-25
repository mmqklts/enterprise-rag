import re
from dataclasses import dataclass
from enum import StrEnum

from enterprise_rag.article_chunker import ARTICLE_HEADING_PATTERN

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+\S+",
    re.MULTILINE,
)
CHAPTER_HEADING_PATTERN = re.compile(
    r"^第\s*[一二三四五六七八九十百千万零〇两]+\s*[编章节]\s*",
    re.MULTILINE,
)
CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}


class ChunkingStrategy(StrEnum):
    ARTICLE = "article"
    HEADING = "heading"
    SEMANTIC = "semantic"
    FIXED = "fixed"


@dataclass(frozen=True)
class StructureProfile:
    strategy: ChunkingStrategy
    confidence: float
    article_count: int
    article_sequence_ratio: float
    markdown_heading_count: int
    chapter_heading_count: int
    paragraph_count: int
    reasons: tuple[str, ...]


def chinese_number_to_int(value: str) -> int:
    if value.isdigit():
        return int(value)

    total = 0
    current = 0

    for character in value:
        if character in CHINESE_DIGITS:
            current = CHINESE_DIGITS[character]
            continue

        unit = CHINESE_UNITS.get(character)
        if unit is None:
            continue

        if current == 0:
            current = 1

        if unit == 10000:
            total = (total + current) * unit
        else:
            total += current * unit
        current = 0

    return total + current


def _article_sequence_ratio(article_numbers: list[int]) -> float:
    unique_numbers = sorted(set(article_numbers))
    if len(unique_numbers) < 2:
        return 1.0 if unique_numbers else 0.0

    span = unique_numbers[-1] - unique_numbers[0] + 1
    return len(unique_numbers) / span


def detect_structure(text: str) -> StructureProfile:
    stripped = text.strip()
    article_numbers = [
        chinese_number_to_int(match.group("number"))
        for match in ARTICLE_HEADING_PATTERN.finditer(stripped)
    ]
    markdown_headings = MARKDOWN_HEADING_PATTERN.findall(stripped)
    chapter_headings = CHAPTER_HEADING_PATTERN.findall(stripped)
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", stripped)
        if paragraph.strip()
    ]
    article_count = len(article_numbers)
    article_sequence_ratio = _article_sequence_ratio(article_numbers)

    if article_count >= 3 and article_sequence_ratio >= 0.9:
        confidence = min(
            0.99,
            0.75 + min(article_count, 100) / 1000,
        )
        return StructureProfile(
            strategy=ChunkingStrategy.ARTICLE,
            confidence=round(confidence, 3),
            article_count=article_count,
            article_sequence_ratio=round(article_sequence_ratio, 3),
            markdown_heading_count=len(markdown_headings),
            chapter_heading_count=len(chapter_headings),
            paragraph_count=len(paragraphs),
            reasons=(
                f"检测到 {article_count} 条带标题法条",
                f"法条编号连续率 {article_sequence_ratio:.2%}",
            ),
        )

    if len(markdown_headings) >= 3:
        confidence = min(0.95, 0.65 + len(markdown_headings) * 0.03)
        return StructureProfile(
            strategy=ChunkingStrategy.HEADING,
            confidence=round(confidence, 3),
            article_count=article_count,
            article_sequence_ratio=round(article_sequence_ratio, 3),
            markdown_heading_count=len(markdown_headings),
            chapter_heading_count=len(chapter_headings),
            paragraph_count=len(paragraphs),
            reasons=(f"检测到 {len(markdown_headings)} 个 Markdown 标题",),
        )

    if len(chapter_headings) >= 3:
        confidence = min(0.9, 0.6 + len(chapter_headings) * 0.03)
        return StructureProfile(
            strategy=ChunkingStrategy.HEADING,
            confidence=round(confidence, 3),
            article_count=article_count,
            article_sequence_ratio=round(article_sequence_ratio, 3),
            markdown_heading_count=len(markdown_headings),
            chapter_heading_count=len(chapter_headings),
            paragraph_count=len(paragraphs),
            reasons=(f"检测到 {len(chapter_headings)} 个章节标题",),
        )

    if len(paragraphs) >= 4:
        average_length = sum(map(len, paragraphs)) / len(paragraphs)
        if 15 <= average_length <= 1200:
            return StructureProfile(
                strategy=ChunkingStrategy.SEMANTIC,
                confidence=0.7,
                article_count=article_count,
                article_sequence_ratio=round(article_sequence_ratio, 3),
                markdown_heading_count=len(markdown_headings),
                chapter_heading_count=len(chapter_headings),
                paragraph_count=len(paragraphs),
                reasons=(
                    f"检测到 {len(paragraphs)} 个自然段",
                    "未发现稳定标题或法条结构",
                ),
            )

    return StructureProfile(
        strategy=ChunkingStrategy.FIXED,
        confidence=0.5,
        article_count=article_count,
        article_sequence_ratio=round(article_sequence_ratio, 3),
        markdown_heading_count=len(markdown_headings),
        chapter_heading_count=len(chapter_headings),
        paragraph_count=len(paragraphs),
        reasons=("未检测到足够稳定的文档结构",),
    )
