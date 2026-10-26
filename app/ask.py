"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    contents = [*(history or []), user_message]
    response = client.models.generate_content(model=MODEL, contents=contents)
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage and usage.prompt_token_count is not None else "不明"
    output_tokens = usage.candidates_token_count if usage and usage.candidates_token_count is not None else "不明"
    print(f"[tokens] input={input_tokens} output={output_tokens}")

    answer = response.text or ""
    if history is not None:
        history.extend(
            [
                user_message,
                types.Content(role="model", parts=[types.Part.from_text(text=answer)]),
            ]
        )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    history: list[types.Content] = []
    question = args.question
    try:
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, history if args.chat else None))
            if not args.chat:
                break
            question = input("\n次の質問 (終了: q): ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if not question:
                continue
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
