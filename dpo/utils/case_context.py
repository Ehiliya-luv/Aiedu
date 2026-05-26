# -*- coding: utf-8 -*-
"""Load original case text for DPO prompt construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple


DPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_strip(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    return _safe_strip(value).replace("\r\n", "\n").replace("\r", "\n")


def _resolve_dpo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return DPO_ROOT / p


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object: {path}:{line_no}")
            yield obj


def load_case_contexts(paths: Sequence[str]) -> Dict[Tuple[str, int], str]:
    contexts: Dict[Tuple[str, int], str] = {}
    for raw_path in paths:
        path = _resolve_dpo_path(raw_path)
        for row in _iter_jsonl(path):
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            text = _normalize_text(row.get("input_context"))
            if not text:
                continue
            fk_file_id = meta.get("fk_file_id")
            raw_medical_id = meta.get("raw_medical_id")
            if fk_file_id is not None:
                contexts[("fk_file_id", int(fk_file_id))] = text
            if raw_medical_id is not None:
                contexts[("raw_medical_id", int(raw_medical_id))] = text
    return contexts


def resolve_case_text(contexts: Dict[Tuple[str, int], str], *, fk_file_id: Any = None, raw_medical_id: Any = None) -> str:
    for key_name, value in (("raw_medical_id", raw_medical_id), ("fk_file_id", fk_file_id)):
        if value is None:
            continue
        try:
            text = contexts.get((key_name, int(value)), "")
        except (TypeError, ValueError):
            text = ""
        if text:
            return text
    return ""
