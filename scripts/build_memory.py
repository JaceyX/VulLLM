import argparse

from src.vullm.memory.reasoning_memory import build_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/reasoning_memory.json")
    args = parser.parse_args()

    build_memory(args.input, args.output)
    print(f"Reasoning memory saved to {args.output}")


if __name__ == "__main__":
    main()
