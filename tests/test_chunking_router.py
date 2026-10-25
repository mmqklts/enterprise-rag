from enterprise_rag.chunking_router import ChunkingRouter
from enterprise_rag.pdf_reader import PdfPage
from enterprise_rag.structure_detector import ChunkingStrategy


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "报销" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_router_selects_article_strategy() -> None:
    pages = [
        PdfPage(
            page_number=1,
            text=(
                "第一条 【规则一】正文一。"
                "第二条 【规则二】正文二。"
                "第三条 【规则三】正文三。"
            ),
        )
    ]

    result = ChunkingRouter().route(pages)

    assert result.strategy == ChunkingStrategy.ARTICLE
    assert [chunk.article_number for chunk in result.chunks] == [
        "第一条",
        "第二条",
        "第三条",
    ]


def test_router_selects_heading_strategy() -> None:
    pages = [
        PdfPage(
            page_number=1,
            text=(
                "# 制度\n\n## 请假\n员工请假需要提交申请。\n\n"
                "## 报销\n报销需要提供发票。\n\n"
                "## 安全\n密码不得少于十二位。"
            ),
        )
    ]

    result = ChunkingRouter().route(pages)

    assert result.strategy == ChunkingStrategy.HEADING
    assert [chunk.section_title for chunk in result.chunks] == [
        "制度",
        "请假",
        "报销",
        "安全",
    ]


def test_router_uses_embedding_similarity_for_semantic_text() -> None:
    pages = [
        PdfPage(
            page_number=1,
            text=(
                "第一段讨论请假制度的具体流程。\n\n"
                "第二段继续说明请假申请的办理方式。\n\n"
                "第三段介绍完全不同的报销材料要求。\n\n"
                "第四段继续解释不同报销凭证的提交方式。"
            ),
        )
    ]
    router = ChunkingRouter(
        embedder=FakeEmbedder(),
        semantic_threshold=0.72,
    )

    result = router.route(
        pages,
        strategy=ChunkingStrategy.SEMANTIC,
    )

    assert result.strategy == ChunkingStrategy.SEMANTIC
    assert len(result.chunks) == 2
    assert "请假" in result.chunks[0].text
    assert "报销" in result.chunks[1].text


def test_router_falls_back_to_fixed_strategy() -> None:
    pages = [PdfPage(page_number=1, text="短文本。")]

    result = ChunkingRouter().route(pages)

    assert result.strategy == ChunkingStrategy.FIXED
    assert len(result.chunks) == 1
