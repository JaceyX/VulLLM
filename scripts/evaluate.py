import argparse
import json
from pathlib import Path

from src.vullm.eval.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default="outputs/eval_result.json")
    args = parser.parse_args()

    result = evaluate(args.pred, args.gold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "cwe_accuracy": result["cwe_accuracy"],
        "cwe_macro_f1": result["cwe_macro_f1"],
        "severity_accuracy": result["severity_accuracy"],
        "severity_macro_f1": result["severity_macro_f1"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
