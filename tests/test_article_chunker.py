from enterprise_rag.article_chunker import chunk_legal_articles
from enterprise_rag.pdf_reader import PdfPage


def test_chunk_legal_articles_preserves_metadata_and_cross_page_text() -> None:
    pages = [
        PdfPage(
            page_number=1,
            text="目录\n第十三条\n",
        ),
        PdfPage(
            page_number=2,
            text=(
                "第十三条 【自然人民事权利能力的起止】"
                "自然人从出生时起到死亡时止，具有民事权利能力。"
            ),
        ),
        PdfPage(
            page_number=3,
            text=(
                "第十四条 【自然人民事权利能力平等】"
                "自然人的民事权利能力一律平等。"
            ),
        ),
    ]

    articles = chunk_legal_articles(pages)

    assert len(articles) == 2
    assert articles[0].article_number == "第十三条"
    assert articles[0].article_title == "自然人民事权利能力的起止"
    assert articles[0].page_start == 2
    assert articles[0].page_end == 2
    assert articles[1].article_number == "第十四条"
    assert articles[1].page_start == 3


def test_chunk_legal_articles_tracks_cross_page_article() -> None:
    pages = [
        PdfPage(
            page_number=1,
            text="第一条 【立法目的】为了保护民事主体的合法权益，",
        ),
        PdfPage(
            page_number=2,
            text="维护社会和经济秩序。\n第二条 【调整范围】民法调整平等主体之间的关系。",
        ),
    ]

    articles = chunk_legal_articles(pages)

    assert len(articles) == 2
    assert articles[0].page_start == 1
    assert articles[0].page_end == 2
    assert "维护社会和经济秩序" in articles[0].text
