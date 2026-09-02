"""Разбор и валидация машиночитаемой зоны (МЧЗ/MRZ) паспорта РФ.

Формат — Приложение №12 к Приказу МВД России от 31.03.2021 № 186:
две строки по 44 символа, контрольные цифры по алгоритму «модуль 10»
с весовой последовательностью 731.
"""

from __future__ import annotations

import re
from typing import Any

_MRZ_CHARS = re.compile(r"[^A-Z0-9<]")
_MRZ_LINE1_ANCHOR = re.compile(r"^P[A-Z<]{0,2}RUS")


def extract_mrz_lines(rec_texts: list[str]) -> list[str] | None:
    """Достаёт две строки MRZ из списка строк, распознанных OCR.

    Ищет якорь первой строки (``P`` + тип документа + ``RUS``), склеивает
    все последующие MRZ-совместимые токены и разрезает на две строки по 44
    символа. Длину намеренно не проверяет — это делает парсер: при неполном
    или шумном распознавании возвращаются строки некратной длины, и
    валидация честно проваливается с понятной причиной.

    Args:
        rec_texts: Список распознанных текстовых строк (PaddleOCR
            ``rec_texts``).

    Returns:
        Список ``[line1, line2]`` либо ``None``, если якорь не найден.
    """
    anchor_idx: int | None = None
    for i, text in enumerate(rec_texts):
        if _MRZ_LINE1_ANCHOR.match(text.strip()):
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    joined = "".join(t.strip() for t in rec_texts[anchor_idx:])
    joined = _MRZ_CHARS.sub("", joined)
    return [joined[:44], joined[44:88]]


class RussianPassportMRZParser:
    """Парсер машиночитаемой записи (MRZ) внутреннего паспорта РФ."""

    WEIGHTS = [7, 3, 1]

    @classmethod
    def parse(cls, mrz_lines: list[str]) -> dict[str, Any]:
        """Разбирает две строки MRZ и проверяет контрольные цифры.

        Args:
            mrz_lines: Две строки MRZ по 44 символа.

        Returns:
            Словарь с полями документа и результатами проверок (``checks``,
            ``valid``).

        Raises:
            ValueError: Если число строк не равно двум или их длина не 44.
        """
        if len(mrz_lines) != 2:
            raise ValueError("MRZ должен содержать ровно две строки")

        line1 = mrz_lines[0].strip()
        line2 = mrz_lines[1].strip()

        if len(line1) != 44 or len(line2) != 44:
            raise ValueError(
                "Каждая строка MRZ должна быть длиной 44 символа "
                f"(получено line1={len(line1)}, line2={len(line2)})"
            )

        # ---- Первая строка ----
        doc_type = line1[0:2]
        issuing_state = line1[2:5]
        name_part = line1[5:44]

        parts = name_part.split("<<")
        surname = parts[0].replace("<", "").strip() if parts else ""
        given_names_raw = parts[1] if len(parts) > 1 else ""
        given_parts = given_names_raw.split("<")
        given_name = (
            given_parts[0].replace("<", "").strip() if given_parts else ""
        )
        middle_name = (
            given_parts[1].replace("<", "").strip()
            if len(given_parts) > 1
            else ""
        )

        # ---- Вторая строка ----
        doc_number = line2[0:9]
        doc_number_check = line2[9]
        citizenship = line2[10:13]
        birth_date = line2[13:19]
        birth_date_check = line2[19]
        sex = line2[20]
        expiry_date = line2[21:27]
        expiry_date_check = line2[27]
        personal_code = line2[28:42]
        personal_code_check = line2[42]
        final_check = line2[43]

        # ---- Проверка контрольных цифр ----
        checks = {
            "doc_number": cls._verify_check_digit(doc_number, doc_number_check),
            "birth_date": cls._verify_check_digit(birth_date, birth_date_check),
            "expiry_date": cls._verify_check_digit(
                expiry_date, expiry_date_check, allow_empty=True
            ),
            "personal_code": cls._verify_check_digit(
                personal_code, personal_code_check, allow_empty=True
            ),
            "final": cls._verify_final_check_digit(line2, final_check),
        }

        # ---- Формирование результата ----
        return {
            "document_type": doc_type,
            "issuing_state": issuing_state,
            "surname": surname,
            "given_name": given_name,
            "middle_name": middle_name or None,
            "document_number": doc_number,
            "document_number_check_digit": doc_number_check,
            "citizenship": citizenship,
            "birth_date": cls._format_date(birth_date, threshold=50),
            "birth_date_raw": birth_date,
            "birth_date_check_digit": birth_date_check,
            "sex": sex,
            "expiry_date": cls._format_date(expiry_date, threshold=70)
            if expiry_date != "<<<<<<"
            else None,
            "expiry_date_raw": expiry_date,
            "expiry_date_check_digit": expiry_date_check,
            "personal_code": personal_code.replace("<", "") or None,
            "personal_code_raw": personal_code,
            "personal_code_check_digit": personal_code_check,
            "final_check_digit": final_check,
            "checks": checks,
            "valid": all(checks.values()),
        }

    @classmethod
    def _verify_check_digit(
        cls, data: str, check_digit: str, allow_empty: bool = False
    ) -> bool:
        """Проверка контрольной цифры (модуль 10, вес 731).

        Если ``allow_empty=True`` и все символы ``data`` — заполнители, а
        ``check_digit`` тоже ``<``, возвращает ``True``.

        Args:
            data: Проверяемая последовательность символов.
            check_digit: Ожидаемая контрольная цифра.
            allow_empty: Допускать пустое (заполненное ``<``) поле.

        Returns:
            ``True``, если контрольная цифра корректна.
        """
        if allow_empty and all(c == "<" for c in data) and check_digit == "<":
            return True

        if not data or not check_digit:
            return False

        digits = [int(c) if c.isdigit() else 0 for c in data]
        total = sum(
            d * cls.WEIGHTS[i % len(cls.WEIGHTS)] for i, d in enumerate(digits)
        )

        return str(total % 10) == check_digit

    @classmethod
    def _verify_final_check_digit(cls, line2: str, check_digit: str) -> bool:
        """Заключительная контрольная цифра второй строки.

        Считается по позициям 1–10, 14–20, 22–43 (включая промежуточные
        контрольные цифры) с единой весовой последовательностью 731.

        Args:
            line2: Вторая строка MRZ.
            check_digit: Ожидаемая заключительная контрольная цифра.

        Returns:
            ``True``, если контрольная цифра корректна.
        """
        data = "".join((line2[0:10], line2[13:20], line2[21:43]))
        return cls._verify_check_digit(data, check_digit)

    @classmethod
    def _format_date(cls, date_str: str, threshold: int = 70) -> str | None:
        """Преобразует YYMMDD в YYYY-MM-DD.

        Если год >= threshold — 1900+, иначе 2000+.

        Args:
            date_str: Строка даты в формате YYMMDD.
            threshold: Порог века для двухзначного года.

        Returns:
            Дата в формате ISO (``YYYY-MM-DD``) или ``None``.
        """
        if not date_str or date_str == "<<<<<<":
            return None

        clean = re.sub(r"[^0-9]", "", date_str)
        if len(clean) != 6:
            return None

        year = int(clean[0:2])
        month = clean[2:4]
        day = clean[4:6]

        full_year = 1900 + year if year >= threshold else 2000 + year
        return f"{full_year:04d}-{month}-{day}"
