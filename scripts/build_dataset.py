import argparse

from src.vullm.data.build_instruction_dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default="data/processed/eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    args = parser.parse_args()

    build_dataset(
        input_path=args.input,
        output_path=args.output,
        eval_path=args.eval_output,
        eval_ratio=args.eval_ratio,
    )
    print(f"Dataset saved to {args.output}")
    print(f"Eval saved to {args.eval_output}")


if __name__ == "__main__":
    main()
