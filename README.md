# Описание проекта

Двухэтапный пайплайн: классический OCR (PaddleOCR, PP-OCRv6) выравнивает
страницы и распознаёт текст, MRZ валидируется на уровне документа, а
VLM (ollama или OpenAI-совместимый эндпоинт) одним запросом по
информативным страницам извлекает ключевые поля паспорта.

## Этапы

1. **Вход.** Директории (все `.pdf`/`.jpg`/`.png` верхнего уровня,
   регистронезависимо, отсортированы по имени) и/или отдельные файлы.
2. **Загрузка.** PDF рендерится через pypdfium2 с `--pdf-dpi`;
   число страниц ограничено `--max-pages` (лишние отбрасываются
   с предупреждением).
3. **PaddleOCR — постранично.** Классификация ориентации, поворот
   страницы и распознавание текста на локальных safetensors-моделях
   из `--paddle-models`.
4. **MRZ — один на документ.** Берётся с первой страницы, где найден
   якорь MRZ; контрольные цифры проверяются (`tools/mrz.py`). Если строки
   найдены, но не разобрались (например, длина ≠ 44 из-за шума OCR) —
   в результат пишется заглушка с `valid: false` и причиной.
5. **VLM — один запрос на документ.** Все информативные страницы
   отправляются вместе (страницы с числом распознанных строк
   ≤ `--min-ocr-texts` исключаются). VLM извлекает поля «Паспорт выдан»,
   «Место рождения» и «Место регистрации» — последнюю актуальную
   регистрацию без отметки «Снят с регистрационного учета».
   Structured output по pydantic-схеме, `temperature=0`. Бэкенд —
   `--vlm-backend`: нативная ollama или любой OpenAI-совместимый
   эндпоинт (`openai_like_endpoint`).

## CLI-аргументы

| Аргумент | Дефолт | Назначение |
|---|---|---|
| `images` (позиционный) | `data/passports` | Документы (`.pdf`/`.jpg`/`.png`) или директории с ними |
| `-o`, `--output` | `data/output` | Каталог для JSON-результатов |
| `--paddle-models` | `models/PaddlePaddle` | Каталог с локальными моделями PaddleOCR |
| `--lang` | `ru` | Язык распознавания PaddleOCR |
| `--det-limit-side-len` | `1280` | Ограничение стороны изображения для детекции текста |
| `--det-limit-type` | `max` | Тип ограничения стороны детекции |
| `--vlm-model` | `qwen3.5:4b-q8_0` | Модель VLM для шага 2 (для обоих бэкендов) |
| `--vlm-backend` | `ollama` | Бэкенд VLM: `ollama` (нативное API) или `openai_like_endpoint` (OpenAI-совместимый API) |
| `--vlm-base-url` | `http://localhost:11434/v1` | Base URL OpenAI-совместимого API (для `openai_like_endpoint`) |
| `--vlm-api-key` | `OPENAI_API_KEY` → `ollama` | API-ключ для `openai_like_endpoint` (локальная ollama ключ игнорирует) |
| `--pdf-dpi` | `150` | DPI рендеринга PDF-страниц |
| `--max-pages` | `8` | Максимум обрабатываемых страниц документа |
| `--min-ocr-texts` | `2` | Страницы с ≤N распознанных строк исключаются из VLM-запроса |
| `--log-dir` | `logs` | Каталог для файлов логов (отдельный лог на запуск) |

Запуск:

```bash
uv run main.py                     # все документы из data/passports
uv run main.py data/passports_small  # конкретная директория
# VLM через OpenAI-совместимый эндпоинт (по умолчанию — локальная ollama):
uv run main.py --vlm-backend openai_like_endpoint
# VLM через удалённый эндпоинт (ключ в OPENAI_API_KEY):
OPENAI_API_KEY=sk-... uv run main.py --vlm-backend openai_like_endpoint \
  --vlm-base-url https://api.example.com/v1 --vlm-model qwen3.5-vl
```

## Выходной JSON

На каждый документ — `{stem}.json` в `--output` (сокращённый пример):

```json
{
  "source": "data/passports/passport_3.pdf",
  "pages": [
    {"ocr_texts": ["РОССИЙСКАЯ ФЕДЕРАЦИЯ", "..."],
     "ocr_scores": [0.98, "..."]}
  ],
  "mrz": {"valid": true, "...": "..."},
  "vlm": {
    "issued_by": "...",
    "birth_place": "...",
    "registration_address": "..."
  },
  "vlm_meta": {
    "n_tokens_sent": 6084,
    "n_tokens_generated": 85,
    "prefill_elapsed_sec": 60.7,
    "generation_elapsed_sec": 11.7
  },
  "timings": {"ocr_elapsed_sec": 19.3, "vlm_elapsed_sec": 80.4,
              "total_elapsed_sec": 100.0}
}
```
