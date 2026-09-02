"""Функции-этапы пайплайна обработки паспорта РФ (``pipeline_1.py``).

Инициализация PaddleOCR на локальных моделях, загрузка документов,
классификация ориентации + OCR, извлечение MRZ и VLM-полей.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pypdfium2 as pdfium
from openai import OpenAI
from paddleocr import PaddleOCR
from PIL import Image

from tools.mrz import RussianPassportMRZParser, extract_mrz_lines
from tools.prompts import PASSPORT_VLM_PROMPT
from tools.schemas import Document, PassportVLM, Page

logger = logging.getLogger("pipeline_1.tools")


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------


def build_ocr(
    paddle_models: Path,
    lang: str,
    det_limit_side_len: int,
    det_limit_type: str,
) -> PaddleOCR:
    """Инициализирует PaddleOCR на локальных моделях PP-OCRv6.

    Args:
        paddle_models: Каталог с локальными моделями PaddleOCR.
        lang: Язык распознавания.
        det_limit_side_len: Ограничение стороны изображения для детекции.
        det_limit_type: Тип ограничения ('max'/'min').

    Returns:
        Настроенный экземпляр PaddleOCR.
    """
    return PaddleOCR(
        doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
        doc_orientation_classify_model_dir=str(
            paddle_models / "PP-LCNet_x1_0_doc_ori_safetensors"
        ),
        textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
        textline_orientation_model_dir=str(
            paddle_models / "PP-LCNet_x1_0_textline_ori_safetensors"
        ),
        text_detection_model_name="PP-OCRv6_medium_det",
        text_detection_model_dir=str(
            paddle_models / "PP-OCRv6_medium_det_safetensors"
        ),
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_recognition_model_dir=str(
            paddle_models / "PP-OCRv6_medium_rec_safetensors"
        ),
        engine="transformers",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang=lang,
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type=det_limit_type,
    )


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_document(
    path: str,
    dpi: int,
    max_pages: int,
) -> Document:
    """Загружает документ (``.jpg`` или ``.pdf``) как набор страниц.

    Число страниц ограничено ``max_pages``: лишние страницы PDF
    отбрасываются с предупреждением.

    Args:
        path: Путь к файлу.
        dpi: DPI рендеринга для PDF.
        max_pages: Максимум страниц на документ.

    Returns:
        Экземпляр ``Document`` с одной или несколькими страницами.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат файла не поддерживается.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if p.suffix.lower() == ".pdf":
        pdf = pdfium.PdfDocument(path)
        total = len(pdf)
        if total > max_pages:
            logger.warning(
                f"{path}: в PDF {total} стр., обрабатываются первые {max_pages}"
            )
        pages = [
            Page(image=pdf[i].render(scale=dpi / 72).to_pil())
            for i in range(min(total, max_pages))
        ]
    elif p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        pages = [Page(image=Image.open(path))]
    else:
        raise ValueError(f"Неподдерживаемый формат: {p.suffix}")

    return Document(path=path, pages=pages)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def classify_orientation(
    ocr: PaddleOCR, page: Page, det_limit_side_len: int, det_limit_type: str
) -> None:
    """Классифицирует ориентацию страницы, поворачивает и заполняет OCR.

    Один вызов ``ocr.predict`` с ``use_doc_orientation_classify=True``
    возвращает угол поворота и распознанные тексты. Изображение
    поворачивается по этому углу и сохраняется в ``page.image`` для
    последующего использования VLM.

    Args:
        ocr: Инициализированный PaddleOCR.
        page: Страница для обработки (``image`` заменяется на повёрнутое).
        det_limit_side_len: Ограничение стороны изображения для детекции.
        det_limit_type: Тип ограничения стороны детекции.
    """
    results = ocr.predict(
        np.array(page.image),
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type=det_limit_type,
    )

    for result in results:
        dpr = result.get("doc_preprocessor_res", {})
        angle = dpr.get("angle", 0)
        if angle:
            page.image = page.image.rotate(-angle, expand=True)

        page.ocr_texts = list(result.get("rec_texts", []))
        page.ocr_scores = list(result.get("rec_scores", []))


def process_mrz(doc: Document) -> None:
    """Извлекает и валидирует MRZ — один на весь документ.

    MRZ физически один на паспорт, поэтому страницы просматриваются по
    порядку до первого найденного якоря MRZ. Если строки найдены, но
    не разобрались (например, длина ≠ 44 из-за шума OCR), в ``doc.mrz``
    сохраняется заглушка с ``valid=False`` и причиной ошибки.

    Args:
        doc: Документ с заполненными ``ocr_texts`` у страниц.
    """
    for page in doc.pages:
        mrz_lines = extract_mrz_lines(page.ocr_texts)
        if mrz_lines is None:
            continue
        try:
            doc.mrz = RussianPassportMRZParser.parse(mrz_lines)
        except ValueError as e:
            doc.mrz = {"valid": False, "error": str(e), "lines": mrz_lines}
        return
    doc.mrz = None


def process_vlm(
    doc: Document,
    model: str,
    stem: str,
    min_texts: int = 2,
    backend: str = "ollama",
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Извлекает поля паспорта через VLM по всем страницам сразу.

    Все развёрнутые страницы документа передаются одним запросом:
    регистраций может быть несколько на разных страницах, и выбор
    последней актуальной требует весь документ целиком. Страницы с
    числом распознанных строк не более ``min_texts`` исключаются из
    запроса как неинформативные (в ``doc.pages`` сохраняются). Если
    информативных страниц нет, VLM-запрос пропускается и ``doc.vlm``
    остаётся ``None``. Результат (или ошибка) попадает в ``doc.vlm``
    и сохраняется в общий ``{stem}.json``. Метрики запроса (число
    токенов; длительности prefill/генерации, если бэкенд их отдаёт)
    попадают в ``doc.vlm_meta``.

    Args:
        doc: Документ с развёрнутыми ``image`` у страниц.
        model: Имя VLM-модели.
        stem: Базовое имя для логирования.
        min_texts: Порог: страницы с числом распознанных строк не
            более этого значения исключаются из VLM-запроса.
        backend: Бэкенд VLM: 'ollama' — нативное API ollama;
            'openai_like_endpoint' — любой OpenAI-совместимый API.
        base_url: Base URL OpenAI-совместимого API (только для
            'openai_like_endpoint').
        api_key: API-ключ OpenAI-совместимого API (только для
            'openai_like_endpoint'; ollama ключ игнорирует).
    """
    pages = [p for p in doc.pages if len(p.ocr_texts) > min_texts]
    if len(pages) < len(doc.pages):
        logger.info(
            f"{stem}: в VLM отправлены {len(pages)} из {len(doc.pages)} стр. "
            f"(≤{min_texts} распознанных строк)"
        )
    if not pages:
        logger.info(f"{stem}: все страницы пустые, VLM пропущен")
        return

    images = [
        base64.b64encode(_image_to_bytes(p.image)).decode() for p in pages
    ]

    content: str | None = None
    meta: dict[str, Any] | None = None
    try:
        if backend == "openai_like_endpoint":
            content, meta = _chat_openai(model, images, base_url, api_key)
        else:
            content, meta = _chat_ollama(model, images)
        result = PassportVLM.model_validate_json(content)
        doc.vlm = result.model_dump()
    except Exception as e:
        doc.vlm = {"error": str(e), "raw": content}
        logger.warning(f"{stem}: не удалось извлечь VLM-поля — {e}")

    if meta is not None:
        doc.vlm_meta = meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_bytes(image: Image.Image) -> bytes:
    """Кодирует PIL-изображение в JPEG-байты.

    Args:
        image: PIL-изображение.

    Returns:
        JPEG-байты.

    Raises:
        OSError: Если изображение не сохраняется в JPEG
            (например, режим с альфа-каналом без конвертации в RGB).
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _chat_ollama(model: str, images: list[str]) -> tuple[str, dict[str, Any]]:
    """Отправляет VLM-запрос через нативное API ollama.

    Args:
        model: Имя ollama-модели.
        images: Изображения (JPEG) в base64.

    Returns:
        Кортеж (текст ответа модели, метрики запроса).
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PASSPORT_VLM_PROMPT,
                "images": images,
            }
        ],
        format=PassportVLM.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    meta = {
        "n_tokens_sent": response.prompt_eval_count,
        "n_tokens_generated": response.eval_count,
        "prefill_elapsed_sec": (
            response.prompt_eval_duration / 1e9
            if response.prompt_eval_duration is not None
            else None
        ),
        "generation_elapsed_sec": (
            response.eval_duration / 1e9
            if response.eval_duration is not None
            else None
        ),
    }
    return response.message.content, meta


def _chat_openai(
    model: str,
    images: list[str],
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, dict[str, Any]]:
    """Отправляет VLM-запрос через OpenAI-совместимый API.

    Structured output через ``response_format`` c JSON-схемой;
    ``reasoning_effort='none'`` отключает рассуждения модели —
    аналог ``think=False`` нативного API ollama.

    Args:
        model: Имя модели на эндпоинте.
        images: Изображения (JPEG) в base64.
        base_url: Base URL API (например, ``http://localhost:11434/v1``).
        api_key: API-ключ (локальная ollama ключ игнорирует).

    Returns:
        Кортеж (текст ответа модели, метрики запроса).
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PASSPORT_VLM_PROMPT},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        }
                        for image in images
                    ],
                ],
            }
        ],
        temperature=0,
        reasoning_effort="none",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "passport_vlm",
                "schema": PassportVLM.model_json_schema(),
                "strict": False,
            },
        },
    )
    usage = response.usage
    meta = {
        "n_tokens_sent": usage.prompt_tokens if usage else None,
        "n_tokens_generated": usage.completion_tokens if usage else None,
        "prefill_elapsed_sec": None,
        "generation_elapsed_sec": None,
    }
    return response.choices[0].message.content, meta
