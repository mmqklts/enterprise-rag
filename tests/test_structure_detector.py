from enterprise_rag.structure_detector import (
    ChunkingStrategy,
    chinese_number_to_int,
    detect_structure,
)


def test_detects_article_structure() -> None:
    text = """
第十条 【第一条规则】这里是正文。
第十一条 【第二条规则】这里是正文。
第十二条 【第三条规则】这里是正文。
"""

    profile = detect_structure(text)

    assert profile.strategy == ChunkingStrategy.ARTICLE
    assert profile.article_count == 3
    assert profile.article_sequence_ratio == 1.0


def test_detects_markdown_heading_structure() -> None:
    text = """
# 员工手册

## 请假制度

员工请假需要提交申请。

## 报销制度

报销需要提供发票。

## 安全制度

密码不得少于十二位。
"""

    profile = detect_structure(text)

    assert profile.strategy == ChunkingStrategy.HEADING
    assert profile.markdown_heading_count == 4


def test_detects_semantic_paragraph_structure() -> None:
    text = """
这是第一段普通说明文字，用于描述一个完整主题。

这是第二段普通说明文字，它与第一段相关。

这是第三段普通说明文字，继续展开同一个主题。

这是第四段普通说明文字，用于测试段落识别。
"""

    profile = detect_structure(text)

    assert profile.strategy == ChunkingStrategy.SEMANTIC
    assert profile.paragraph_count == 4


def test_falls_back_to_fixed_for_unstructured_text() -> None:
    profile = detect_structure("短文本。")

    assert profile.strategy == ChunkingStrategy.FIXED
    assert profile.confidence == 0.5


def test_converts_chinese_article_numbers() -> None:
    assert chinese_number_to_int("十三") == 13
    assert chinese_number_to_int("一百八十八") == 188
    assert chinese_number_to_int("一千二百六十") == 1260
