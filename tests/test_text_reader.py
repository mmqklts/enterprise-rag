from pathlib import Path

from enterprise_rag.text_reader import read_text_file


def test_read_text_file() -> None:
    text = read_text_file(Path("data/samples/sample.txt"))
    assert "知识库文档需要定期更新" in text
def test_read_markdown_file() -> None:
    text = read_text_file(Path("data/samples/sample.md"))
    assert "# 企业制度" in text