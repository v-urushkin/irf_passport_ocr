# AGENTS.md

## Project
OCR pipeline for Russian internal passports: PaddleOCR (PP-OCRv6, local
safetensors) per page -> MRZ validation (one per document) -> one VLM call
(native ollama or any OpenAI-compatible endpoint) extracting
issued_by / birth_place / registration_address.

## Public repository
This repo is PUBLIC. Never put private or personal information — real
passport data, names, addresses, MRZ strings, document scans — into
AGENTS.md, README.md, code comments, commits, or any other tracked file.
Inputs and outputs live only in gitignored dirs (`data/`, `logs/`).

## Commands
- Run: `uv run main.py` (defaults to `data/passports`), or
  `uv run main.py <files-or-dirs> -o <output-dir>`
- Sync deps: `uv sync` (Python >= 3.14.5 required)
- No tests, lint, typecheck, or CI are configured. Don't hunt for them.

## Runtime prerequisites (NOT in repo)
- `models/PaddlePaddle/` must contain the 4 local safetensors subdirs:
  `PP-LCNet_x1_0_doc_ori_safetensors`, `PP-LCNet_x1_0_textline_ori_safetensors`,
  `PP-OCRv6_medium_det_safetensors`, `PP-OCRv6_medium_rec_safetensors`
  (present on this machine; NOT in the repo — no download script exists).
  Pipeline fails at `build_ocr` without them.
- VLM backend, one of:
  - ollama server running with `qwen3.5:4b-q8_0` pulled
    (default `--vlm-backend ollama`); or
  - any OpenAI-compatible endpoint via
    `--vlm-backend openai_like_endpoint` (default `--vlm-base-url`
    http://localhost:11434/v1 points at local ollama; API key:
    `--vlm-api-key` or env `OPENAI_API_KEY`, fallback `ollama`).

## Layout
- `main.py` — CLI entrypoint (argparse only; no config files by design)
- `tools/tools.py` — pipeline stages (build_ocr, load_document,
  classify_orientation, process_mrz, process_vlm)
- `tools/mrz.py` — MRZ parser (2 lines x 44 chars, 7-3-1 checksums)
- `tools/schemas.py` — Page/Document dataclasses, PassportVLM pydantic model
- `tools/prompts.py` — VLM prompt (English; everything else is Russian)
- `models/` — local model weights (gitignored, ~450M): `PaddlePaddle/` with
  the 4 `*_safetensors` subdirs consumed by `build_ocr` (see Runtime
  prerequisites); the plain paddle-format dirs beside them and
  `models/PP-DocLayoutV3_safetensors/` are NOT used by code

## Conventions
- Docstrings, log messages, and README are in Russian — keep new code consistent.
- Google-style docstrings (Args/Returns/Raises), ~79-char line wrap.
- CLI args/defaults are duplicated in `main.py` and `README.md` — update both.

## Git quirks
- `uv.lock`, `data/*`, `models/*`, and `*.log` are gitignored on purpose —
  never commit them.
- Inputs: `data/passports`; results: `data/output` (JSON per document);
  per-run logs: `logs/`.
