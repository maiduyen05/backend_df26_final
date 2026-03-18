"""
in_memory.py
Lưu trữ BatchRecord trong RAM.
Interface cố ý giữ đơn giản để sau này swap sang Redis/PostgreSQL
chỉ cần thay file này, không đụng vào service layer.
"""

from __future__ import annotations

from app.core.exceptions import BatchNotFoundError
from app.models.schemas import BatchRecord


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, BatchRecord] = {}

    def save(self, batch_id: str, record: BatchRecord) -> None:
        self._data[batch_id] = record

    def get(self, batch_id: str) -> BatchRecord:
        record = self._data.get(batch_id)
        if record is None:
            raise BatchNotFoundError(batch_id)
        return record

    def exists(self, batch_id: str) -> bool:
        return batch_id in self._data

    def delete(self, batch_id: str) -> None:
        self._data.pop(batch_id, None)

    def all_ids(self) -> list[str]:
        return list(self._data.keys())


# Singleton toàn cục
store = InMemoryStore()
