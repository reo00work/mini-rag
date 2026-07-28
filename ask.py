#!/usr/bin/env python3
"""
mini-rag/ask.py
質問を受け取り、類似チャンクを検索して出典付きで回答を返す CLI。

使い方:
  python ask.py '有給休暇は何日もらえますか？'

オプション:
  --top N     上位 N 件を返す（デフォルト: 3）
  --store PATH  vector_store.json のパスを指定（デフォルト: 同ディレクトリ）

LLM 要約（任意）:
  ANTHROPIC_API_KEY または OPENAI_API_KEY を環境変数に設定すると
  検索結果をもとにした自然文の回答が生成されます。
  未設定でも「抜粋 + 出典」を返す stub モードで動作します。
"""

import json
import math
import os
import re
import sys
from pathlib import Path

STORE_PATH = Path(__file__).parent / "vector_store.json"
DEFAULT_TOP_K = 3


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[぀-ゟ]+|[゠-ヿ]+|[一-鿿]+", text.lower())


def load_store(path: Path) -> dict:
    if not path.exists():
        print(
            f"[ERROR] {path} が見つかりません。先に ingest.py を実行してください。\n"
            "  $ python ingest.py"
        )
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def vectorize_query(tokens: list[str], idf: dict) -> dict:
    """クエリを TF-IDF ベクトル化（L2 正規化）。"""
    tf: dict = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    vec: dict = {}
    for t, c in tf.items():
        val = (c / total) * idf.get(t, 0.1)  # 未知語は微小スコア
        vec[t] = val
    norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def cosine_sim(q_vec: dict, doc_vec: dict) -> float:
    return sum(q_vec.get(t, 0.0) * v for t, v in doc_vec.items())


def search(question: str, store: dict, top_k: int = DEFAULT_TOP_K) -> list[tuple[dict, float]]:
    tokens = tokenize(question)
    if not tokens:
        return []

    idf: dict = store["idf"]
    chunks: list[dict] = store["chunks"]
    vectors: list[dict] = store["vectors"]

    q_vec = vectorize_query(tokens, idf)

    scores = [(cosine_sim(q_vec, vec), i) for i, vec in enumerate(vectors)]
    top = sorted(scores, key=lambda x: -x[0])[:top_k]

    return [(chunks[i], score) for score, i in top if score > 0.0]


def llm_summarize(question: str, results: list[tuple[dict, float]]) -> str | None:
    """
    API キーがある場合のみ LLM で要約。なければ None を返す。
    ANTHROPIC_API_KEY を優先し、次に OPENAI_API_KEY を試みる。
    """
    context = "\n\n".join(
        f"【{r['source']}】\n{r['text']}" for r, _ in results
    )
    prompt = (
        "以下の社内ドキュメント抜粋をもとに、質問に正確かつ簡潔に答えてください。\n"
        "抜粋に記載のない情報は答えず「記載なし」と伝えてください。\n\n"
        f"ドキュメント抜粋:\n{context}\n\n"
        f"質問: {question}\n\n回答:"
    )

    # --- Anthropic ---
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"[LLM] Anthropic API エラー: {e}", file=sys.stderr)

    # --- OpenAI ---
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "社内ドキュメントに基づいて回答するアシスタントです。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] OpenAI API エラー: {e}", file=sys.stderr)

    return None


def format_results(results: list[tuple[dict, float]], show_score: bool = True) -> str:
    lines = []
    for i, (chunk, score) in enumerate(results, 1):
        score_str = f" (スコア: {score:.3f})" if show_score else ""
        excerpt = chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else "")
        lines.append(
            f"\n─── {i}位{score_str} ───\n"
            f"出典: {chunk['source']}  (chunk #{chunk['chunk_index']})\n"
            f"{excerpt}"
        )
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    top_k = DEFAULT_TOP_K
    store_path = STORE_PATH

    # 簡易引数パース
    question_parts: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--store" and i + 1 < len(args):
            store_path = Path(args[i + 1])
            i += 2
        else:
            question_parts.append(args[i])
            i += 1

    if not question_parts:
        print(__doc__)
        sys.exit(0)

    question = " ".join(question_parts)
    print(f"\n質問: {question}\n")

    store = load_store(store_path)
    results = search(question, store, top_k=top_k)

    if not results:
        print("関連する情報が見つかりませんでした。キーワードを変えてお試しください。")
        sys.exit(0)

    # LLM 要約（オプション）
    summary = llm_summarize(question, results)
    if summary:
        print("【AI 要約】")
        print(summary)
        print()

    # 抜粋 + 出典（常に表示）
    mode = "【参照ドキュメント】" if summary else "【回答（抜粋 + 出典）】"
    print(mode)
    print(format_results(results))


if __name__ == "__main__":
    main()
