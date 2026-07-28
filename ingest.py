#!/usr/bin/env python3
"""
mini-rag/ingest.py
ドキュメントをチャンク分割 → TF-IDFベクトル化 → vector_store.json に保存する。
依存ライブラリ: なし（Python 標準ライブラリのみ）
"""

import json
import math
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"
STORE_PATH = Path(__file__).parent / "vector_store.json"
CHUNK_SIZE = 200   # 1チャンクの目標文字数
CHUNK_OVERLAP = 40  # 前チャンクとの重複文字数


def tokenize(text: str) -> list[str]:
    """日本語・英数字を抽出してトークン化する。"""
    # ひらがな・カタカナ・漢字・英数字を個別トークンとして扱う
    tokens = re.findall(r"[a-z0-9]+|[぀-ゟ]+|[゠-ヿ]+|[一-鿿]+", text.lower())
    return tokens


def chunk_text(text: str) -> list[str]:
    """テキストを意味的な塊（段落・箇条書き）でチャンク分割する。"""
    # 空行・改行・句点で分割
    raw_lines = re.split(r"\n{2,}|(?<=[。！？])\n?", text)
    chunks: list[str] = []
    current = ""

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if len(current) + len(line) <= CHUNK_SIZE:
            current += ("" if not current else " ") + line
        else:
            if current:
                chunks.append(current)
            # overlap: 前チャンクの末尾を引き継ぐ
            current = current[-CHUNK_OVERLAP:] + line if current else line

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= 15]


def compute_tfidf(docs: list[dict]) -> tuple[list[dict], dict]:
    """
    TF-IDF ベクトルを計算する（正規化済み）。
    返り値: (vectors_list, idf_dict)
    """
    # TF（単語頻度）
    tf_list: list[dict] = []
    for doc in docs:
        counts: dict = {}
        for t in doc["tokens"]:
            counts[t] = counts.get(t, 0) + 1
        total = len(doc["tokens"]) or 1
        tf_list.append({t: c / total for t, c in counts.items()})

    # IDF（逆文書頻度）
    N = len(docs)
    df: dict = {}
    for tf in tf_list:
        for term in tf:
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log((N + 1) / (d + 1)) + 1.0 for t, d in df.items()}

    # TF-IDF ベクトル（L2 正規化）
    vectors: list[dict] = []
    for tf in tf_list:
        vec = {t: tf[t] * idf.get(t, 1.0) for t in tf}
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})

    return vectors, idf


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"[ERROR] docs ディレクトリが見つかりません: {DOCS_DIR}")
        return

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        print("[ERROR] docs/ 配下に .txt ファイルがありません。")
        return

    print(f"対象ファイル: {len(doc_files)} 件")

    all_chunks: list[dict] = []
    for doc_file in doc_files:
        text = doc_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc_file.name}::chunk{i}",
                "source": doc_file.name,
                "chunk_index": i,
                "text": chunk,
                "tokens": tokenize(chunk),
            })
        print(f"  {doc_file.name}: {len(chunks)} チャンク")

    print(f"チャンク合計: {len(all_chunks)} 件")

    vectors, idf = compute_tfidf(all_chunks)

    # tokens はストア済みなので保存前に除去（ファイルサイズ削減）
    chunks_for_store = [
        {k: v for k, v in c.items() if k != "tokens"}
        for c in all_chunks
    ]

    store = {
        "meta": {
            "version": "1.0",
            "chunk_count": len(all_chunks),
            "doc_count": len(doc_files),
            "vocab_size": len(idf),
        },
        "chunks": chunks_for_store,
        "vectors": vectors,
        "idf": idf,
    }

    STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"\nベクトルストア保存完了: {STORE_PATH}")
    print(f"  チャンク数: {len(all_chunks)}, 語彙数: {len(idf)}")


if __name__ == "__main__":
    main()
