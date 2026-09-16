from enterprise_rag.pdf_reader import read_pdf_pages


def test_read_pdf_pages() -> None:
    pages = read_pdf_pages("data/samples/sample.pdf")

    assert len(pages) >= 1
    assert pages[0].page_number == 1
    assert pages[0].text.strip()