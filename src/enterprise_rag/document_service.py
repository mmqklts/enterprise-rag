from sqlalchemy import select
from sqlalchemy.orm import Session

from enterprise_rag.models import DocumentRecord


def upsert_document_record(
    session: Session,
    knowledge_base_id: int,
    created_by: int,
    document_id: str,
    filename: str,
    stored_path: str,
    chunking_strategy: str,
    chunk_count: int,
) -> DocumentRecord:
    record = session.scalar(
        select(DocumentRecord).where(
            DocumentRecord.knowledge_base_id == knowledge_base_id,
            DocumentRecord.document_id == document_id,
        )
    )

    if record is None:
        record = DocumentRecord(
            knowledge_base_id=knowledge_base_id,
            created_by=created_by,
            document_id=document_id,
            filename=filename,
            stored_path=stored_path,
            chunking_strategy=chunking_strategy,
            chunk_count=chunk_count,
        )
        session.add(record)
    else:
        record.filename = filename
        record.stored_path = stored_path
        record.chunking_strategy = chunking_strategy
        record.chunk_count = chunk_count

    session.commit()
    session.refresh(record)
    return record


def list_document_records(
    session: Session,
    knowledge_base_id: int,
) -> list[DocumentRecord]:
    return list(
        session.scalars(
            select(DocumentRecord)
            .where(DocumentRecord.knowledge_base_id == knowledge_base_id)
            .order_by(DocumentRecord.created_at.desc())
        )
    )


def get_document_record(
    session: Session,
    knowledge_base_id: int,
    document_id: str,
) -> DocumentRecord:
    record = session.scalar(
        select(DocumentRecord).where(
            DocumentRecord.knowledge_base_id == knowledge_base_id,
            DocumentRecord.document_id == document_id,
        )
    )
    if record is None:
        raise LookupError("文档不存在")
    return record


def delete_document_record(
    session: Session,
    record: DocumentRecord,
) -> None:
    session.delete(record)
    session.commit()
