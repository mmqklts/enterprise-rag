from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PdfPage:
    page_number: int
    text: str


def read_pdf_pages(file_path: str | Path) -> list[PdfPage]:
    path = Path(file_path)
    reader = PdfReader(str(path))
    pages: list[PdfPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PdfPage(page_number=page_number, text=text))

    return pages