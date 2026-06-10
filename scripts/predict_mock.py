import argparse
import json
from pathlib import Path

from src.vullm.serve.inference import mock_analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            result = mock_analyze(row.get("description", ""), row.get("code", ""))
            pred = (
                f"漏洞类型：{result['cwe']}\n"
                f"严重程度：{result['severity']}\n"
                f"推理过程：{result['reasoning']}\n"
                f"修复建议：{result['fix']}"
            )
            fout.write(json.dumps({"id": row.get("id", ""), "prediction": pred}, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {output}")


if __name__ == "__main__":
    main()
