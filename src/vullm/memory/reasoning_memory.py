import json
import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text.lower())


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_sim(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in common)
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    if denominator == 0:
        return 0.0
    return numerator / denominator


class ReasoningMemory:
    def __init__(self, records: list[dict]):
        self.records = records
        self.vectors = [
            vectorize((r.get("description", "") + " " + r.get("reasoning", "") + " " + r.get("code", "")))
            for r in records
        ]

    @classmethod
    def from_json(cls, path: str | Path) -> "ReasoningMemory":
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return cls(records)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        qv = vectorize(query)
        scored = []
        for record, vec in zip(self.records, self.vectors):
            scored.append((cosine_sim(qv, vec), record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": round(score, 4), **record}
            for score, record in scored[:top_k]
        ]


def build_memory(input_path: str | Path, output_path: str | Path) -> None:
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "id": row.get("id") or row.get("cve") or "",
                "description": row.get("description", ""),
                "code": row.get("code", ""),
                "cwe": row.get("cwe", "UNKNOWN"),
                "severity": row.get("severity", "UNKNOWN"),
                "reasoning": row.get("reasoning", ""),
                "fix": row.get("fix", ""),
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
