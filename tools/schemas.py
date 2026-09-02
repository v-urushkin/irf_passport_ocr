from dataclasses import dataclass, field
from typing import Any
from PIL import Image

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """Одна страница документа.

    Attributes:
        image: PIL-изображение страницы (после выравнивания ориентации).
        ocr_texts: Распознанные PaddleOCR строки текста.
        ocr_scores: Уверенность распознавания для каждой строки (0–1).
    """

    image: Image.Image
    ocr_texts: list[str] = field(default_factory=list)
    ocr_scores: list[float] = field(default_factory=list)


@dataclass
class Document:
    """Документ (изображение или PDF), разбитый на страницы.

    Attributes:
        path: Исходный путь к файлу документа.
        pages: Страницы документа.
        mrz: Разобранный MRZ (словарь парсера), заглушка с ``valid=False``
            при ошибке разбора или ``None``, если MRZ не найден.
        vlm: VLM-поля (``PassportVLM.model_dump()``), словарь с ``error``
            при неудаче или ``None`` до обработки.
        vlm_meta: Метрики ollama (``n_tokens_sent``, ``n_tokens_generated``,
            ``prefill_elapsed_sec``, ``generation_elapsed_sec``) или ``None``.
        timings: Замеры времени этапов в секундах.
    """

    path: str
    pages: list[Page]
    mrz: dict[str, Any] | None = None
    vlm: dict[str, Any] | None = None
    vlm_meta: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PassportVLM(BaseModel):
    """Результат извлечения полей паспорта через VLM.

    Attributes:
        issued_by: Кем выдан паспорт («Паспорт выдан»).
        birth_place: Место рождения («Место рождения»).
        registration_address: Адрес последней актуальной регистрации
            («Место регистрации»); ``""``, если актуальных регистраций нет.
    """

    issued_by: str
    birth_place: str
    registration_address: str
