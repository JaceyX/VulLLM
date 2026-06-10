import argparse

from openai import OpenAI

from src.vullm.prompts import build_chat_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--description", required=True)
    parser.add_argument("--code", default="")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    response = client.chat.completions.create(
        model=args.model,
        messages=build_chat_prompt(args.description, args.code),
        temperature=0.1,
        max_tokens=512,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
