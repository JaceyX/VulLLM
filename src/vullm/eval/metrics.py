import json
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_label(text: str, label_name: str) -> str:
    patterns = [
        rf"{label_name}[:：]\s*([A-Za-z0-9_\-/]+)",
        rf"{label_name}\s*为\s*([A-Za-z0-9_\-/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return "UNKNOWN"


def evaluate(pred_path: str | Path, gold_path: str | Path) -> dict:
    preds = read_jsonl(pred_path)
    golds = read_jsonl(gold_path)

    pred_map = {x["id"]: x for x in preds}
    y_true_cwe, y_pred_cwe = [], []
    y_true_sev, y_pred_sev = [], []

    details = []

    for gold in golds:
        pred = pred_map.get(gold["id"], {})
        pred_text = pred.get("prediction", "")

        true_cwe = gold.get("cwe", "UNKNOWN")
        true_sev = gold.get("severity", "UNKNOWN")

        pred_cwe = extract_label(pred_text, "漏洞类型")
        pred_sev = extract_label(pred_text, "严重程度")

        y_true_cwe.append(true_cwe)
        y_pred_cwe.append(pred_cwe)
        y_true_sev.append(true_sev)
        y_pred_sev.append(pred_sev)

        details.append({
            "id": gold["id"],
            "true_cwe": true_cwe,
            "pred_cwe": pred_cwe,
            "true_severity": true_sev,
            "pred_severity": pred_sev,
            "prediction": pred_text,
        })

    result = {
        "cwe_accuracy": accuracy_score(y_true_cwe, y_pred_cwe),
        "cwe_macro_f1": f1_score(y_true_cwe, y_pred_cwe, average="macro", zero_division=0),
        "severity_accuracy": accuracy_score(y_true_sev, y_pred_sev),
        "severity_macro_f1": f1_score(y_true_sev, y_pred_sev, average="macro", zero_division=0),
        "cwe_report": classification_report(y_true_cwe, y_pred_cwe, zero_division=0),
        "severity_report": classification_report(y_true_sev, y_pred_sev, zero_division=0),
        "details": details,
    }

    return result
