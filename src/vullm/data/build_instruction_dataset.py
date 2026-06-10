import json
import random
from pathlib import Path
from typing import Iterable

from src.vullm.prompts import INSTRUCTION, build_input, build_output


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_record(row: dict) -> dict:
    description = row.get("description") or row.get("summary") or row.get("cve_description") or ""
    code = row.get("code") or row.get("func_before") or row.get("vulnerable_code") or ""
    cwe = row.get("cwe") or row.get("cwe_id") or row.get("cwe_id_label") or "UNKNOWN"
    severity = row.get("severity") or row.get("cvss_severity") or row.get("label_severity") or "UNKNOWN"
    reasoning = row.get("reasoning") or f"该漏洞与 {cwe} 相关，需要结合描述和代码判断风险。"
    fix = row.get("fix") or row.get("patch") or row.get("recommendation") or ""

    return {
        "id": row.get("id") or row.get("cve") or row.get("cve_id") or "",
        "instruction": INSTRUCTION,
        "input": build_input(description, code),
        "output": build_output(cwe, severity, reasoning, fix),
        "cwe": cwe,
        "severity": severity,
        "description": description,
        "code": code,
    }


def build_dataset(
    input_path: str | Path,
    output_path: str | Path,
    eval_path: str | Path | None = None,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    rows = read_jsonl(input_path)
    normalized = [normalize_record(row) for row in rows]
    random.Random(seed).shuffle(normalized)

    if eval_path:
        split = max(1, int(len(normalized) * (1 - eval_ratio)))
        train_rows = normalized[:split]
        eval_rows = normalized[split:]
        write_jsonl(train_rows, output_path)
        write_jsonl(eval_rows, eval_path)
    else:
        write_jsonl(normalized, output_path)
