from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from PIL import Image

from pydantic import BaseModel, Field


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
        vlm_meta: Метрики VLM-запроса (``n_tokens_sent``,
            ``n_tokens_generated``, ``prefill_elapsed_sec``,
            ``generation_elapsed_sec``) или ``None``.
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

    Поля с форматом (даты, код подразделения, серия, номер, пол)
    допускают пустую строку ``""`` — признак «поле не найдено».

    Attributes:
        issued_by: Кем выдан паспорт («Паспорт выдан»).
        issue_date: Дата выдачи, ``dd.mm.yyyy`` (или ``""``).
        department_code: Код подразделения, ``XXX-XXX`` (или ``""``).
        series: Серия паспорта, только цифры (или ``""``).
        number: Номер паспорта, только цифры (или ``""``).
        surname: Фамилия.
        first_name: Имя.
        patronymic: Отчество (``""``, если отсутствует).
        gender: Пол — ``"МУЖ."`` или ``"ЖЕН."`` (или ``""``).
        birth_date: Дата рождения, ``dd.mm.yyyy`` (или ``""``).
        birth_place: Место рождения.
        last_registration: Последняя актуальная регистрация
            (без отметки «Снят с регистрационного учета»);
            ``""``, если актуальных регистраций нет.
    """

    issued_by: str
    issue_date: Annotated[str, Field(pattern=r"^(?:\d{2}\.\d{2}\.\d{4})?$")]
    department_code: Annotated[str, Field(pattern=r"^(?:\d{3}-\d{3})?$")]
    series: Annotated[str, Field(pattern=r"^\d*$")]
    number: Annotated[str, Field(pattern=r"^\d*$")]
    surname: str
    patronymic: str
    first_name: str
    gender: Literal["МУЖ.", "ЖЕН.", ""]
    birth_date: Annotated[str, Field(pattern=r"^(?:\d{2}\.\d{2}\.\d{4})?$")]
    birth_place: str
    last_registration: str
