"""Пайплайн обработки паспорта РФ.

Этап 1: PaddleOCR (PP-OCRv6) — каждая страница документа (не более
``--max-pages``, по умолчанию 8) обрабатывается отдельно:
классификация ориентации + поворот страницы + распознавание текста.
MRZ один на паспорт, поэтому извлекается и валидируется на уровне
документа — с первой страницы, где найден якорь MRZ.

Этап 2: локальная VLM (ollama, по умолчанию ``qwen3.5:4b-q8_0``)
получает все развёрнутые страницы документа одним запросом и извлекает
поля «Паспорт выдан», «Место рождения» и «Место регистрации» —
последнюю актуальную регистрацию, т.е. без отметки «Снят
с регистрационного учета» (регистраций в паспорте может быть
несколько на разных страницах).

На вход подаются директории (все ``.pdf``/``.jpg``/``.png`` верхнего
уровня) и/или отдельные файлы (каждая страница — отдельное
изображение, DPI конвертации задаётся аргументом ``--pdf-dpi``).

Параметризация — argparse (не config.yaml).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from tools.schemas import Document
from tools.tools import (
    DEFAULT_MAX_PAGES,
    DEFAULT_PDF_DPI,
    build_ocr,
    classify_orientation,
    load_document,
    process_mrz,
    process_vlm,
)

logger = logging.getLogger("pipeline_1")

DEFAULT_PADDLE_MODELS = "models/PaddlePaddle"
DEFAULT_INPUT = "data/passports"
DOCUMENT_SUFFIXES = {".pdf", ".jpg", ".png"}
DEFAULT_OUTPUT = "data/output"
DEFAULT_VLM_MODEL = "qwen3.5:4b-q8_0"
DEFAULT_LOG_DIR = "logs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str) -> Path:
    """Настраивает логирование в терминал и файл.

    Файл — отдельный на запуск: ``{log_dir}/pipeline_1_YYYYMMDD-HHMMSS.log``.
    Терминал получает сообщения пайплайна как есть (формат ``%(message)s``);
    файл — с меткой времени и уровнем. Сторонние библиотеки (paddleocr
    и т.п.) остаются на уровне WARNING.

    Args:
        log_dir: Каталог для файлов логов (создаётся при необходимости).

    Returns:
        Путь к файлу лога текущего запуска.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / f"pipeline_1_{datetime.now():%Y%m%d-%H%M%S}.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    logging.basicConfig(level=logging.WARNING)
    return log_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер аргументов пайплайна.

    Returns:
        Настроенный ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Пайплайн OCR документов (PaddleOCR + Qwen через ollama).",
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=[DEFAULT_INPUT],
        help="Пути к документам (.pdf/.jpg/.png) или директориям с ними "
        f"(по умолчанию: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Каталог для JSON-результатов (по умолчанию: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--paddle-models",
        default=DEFAULT_PADDLE_MODELS,
        help="Каталог с локальными моделями PaddleOCR "
        f"(по умолчанию: {DEFAULT_PADDLE_MODELS}).",
    )
    parser.add_argument(
        "--lang",
        default="ru",
        help="Язык распознавания PaddleOCR (по умолчанию: ru).",
    )
    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=1280,
        help="Ограничение длины стороны изображения для детекции текста "
        "(по умолчанию: 1280).",
    )
    parser.add_argument(
        "--det-limit-type",
        default="max",
        help="Тип ограничения стороны детекции (по умолчанию: max).",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help="Локальная ollama-модель для шага 2 (VLM) "
        f"(по умолчанию: {DEFAULT_VLM_MODEL}).",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=DEFAULT_PDF_DPI,
        help="DPI рендеринга PDF-страниц (по умолчанию: 150).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Максимум обрабатываемых страниц документа "
        f"(по умолчанию: {DEFAULT_MAX_PAGES}).",
    )
    parser.add_argument(
        "--min-ocr-texts",
        type=int,
        default=2,
        help="Страницы с числом распознанных строк не более N "
        "исключаются из VLM-запроса (по умолчанию: 2).",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Каталог для файлов логов "
        f"(по умолчанию: {DEFAULT_LOG_DIR}).",
    )
    return parser


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------


def collect_documents(inputs: Sequence[str]) -> list[str]:
    """Собирает список документов из путей-файлов и директорий.

    Директория раскрывается нерекурсивно: берутся все файлы верхнего
    уровня с поддерживаемым расширением (``.pdf``/``.jpg``/``.png``,
    регистронезависимо), отсортированные по имени.

    Args:
        inputs: Пути к файлам и/или директориям.

    Returns:
        Отсортированный список путей к документам.
    """
    docs: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            docs += [
                str(f)
                for f in sorted(p.iterdir())
                if f.suffix.lower() in DOCUMENT_SUFFIXES
            ]
        else:
            docs.append(item)
    return docs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_result(doc: Document, output_path: Path) -> None:
    """Сохраняет результат обработки документа в JSON.

    Args:
        doc: Обработанный документ.
        output_path: Путь к выходному JSON-файлу.

    Raises:
        OSError: Если не удаётся записать JSON-файл (нет каталога,
            нет прав на запись, нет места на диске).
    """
    result: dict[str, Any] = {
        "source": doc.path,
        "pages": [
            {"ocr_texts": p.ocr_texts, "ocr_scores": p.ocr_scores}
            for p in doc.pages
        ],
        "mrz": doc.mrz,
        "vlm": doc.vlm,
        "vlm_ollama_meta": doc.vlm_meta,
        "timings": doc.timings,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Запускает пайплайн по списку изображений/PDF.

    Args:
        argv: Опциональный список аргументов; по умолчанию ``sys.argv``.

    Raises:
        FileNotFoundError: Из ``load_document`` — если файл документа
            не найден (прерывает обработку всего списка).
        ValueError: Из ``load_document`` — если формат документа
            не поддерживается (прерывает обработку всего списка).
    """
    args = build_parser().parse_args(argv)

    log_file = setup_logging(args.log_dir)
    logger.info(f"Команда: {' '.join(sys.argv)}")
    logger.info(f"Лог: {log_file}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_ocr(
        Path(args.paddle_models),
        args.lang,
        args.det_limit_side_len,
        args.det_limit_type,
    )

    docs = collect_documents(args.images)
    if not docs:
        logger.warning("Документы не найдены (.pdf/.jpg/.png)")
        return

    for path in docs:
        stem = Path(path).stem
        logger.info(f"=== {path} ===")

        t_start = perf_counter()
        doc = load_document(path, dpi=args.pdf_dpi, max_pages=args.max_pages)
        logger.info(f"Страниц: {len(doc.pages)}")

        t_ocr = perf_counter()
        for i, page in enumerate(doc.pages):
            logger.info(f"[{i}] Ориентация + OCR...")
            classify_orientation(
                ocr, page, args.det_limit_side_len, args.det_limit_type
            )
        doc.timings["ocr_elapsed_sec"] = perf_counter() - t_ocr

        process_mrz(doc)
        if doc.mrz:
            status = "валиден" if doc.mrz.get("valid") else "невалиден"
            logger.info(f"MRZ: {status}")
        else:
            logger.info("MRZ: не найден")

        logger.info(f"VLM ({args.vlm_model})...")
        t_vlm = perf_counter()
        process_vlm(doc, args.vlm_model, stem, min_texts=args.min_ocr_texts)
        doc.timings["vlm_elapsed_sec"] = perf_counter() - t_vlm
        doc.timings["total_elapsed_sec"] = perf_counter() - t_start
        if doc.vlm and "error" not in doc.vlm:
            logger.info(f"выдан: {doc.vlm['issued_by']!r}")
            logger.info(f"место рождения: {doc.vlm['birth_place']!r}")
            logger.info(f"адрес: {doc.vlm['registration_address']!r}")

        out_path = output_dir / f"{stem}.json"
        save_result(doc, out_path)
        logger.info(f"Результат: {out_path}")


if __name__ == "__main__":
    main()
