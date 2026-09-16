from pathlib import Path


def read_text_file(file_path: str | Path) -> str:
    """读取 UTF-8 文本文件并返回内容。"""
    path = Path(file_path)
    return path.read_text(encoding="utf-8")