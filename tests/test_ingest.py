"""
mini-rag/tests/test_ingest.py
ingest.py の tokenize() / chunk_text() / compute_tfidf() を検証する。

外部API・ネットワーク依存のない純粋関数のみを対象とする。

実行方法:
    python3 -m unittest discover -s mini-rag/tests -v
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_text,
    compute_tfidf,
    tokenize,
)


class TokenizeTest(unittest.TestCase):
    def test_alphanumeric_tokens(self):
        self.assertEqual(tokenize("RAG2026"), ["rag2026"])

    def test_hiragana_katakana_kanji_split_into_separate_tokens(self):
        # ひらがな・カタカナ・漢字は別々の文字種として個別トークンになる
        result = tokenize("有給休暇はナンボ")
        self.assertEqual(result, ["有給休暇", "は", "ナンボ"])

    def test_mixed_japanese_and_english(self):
        # 漢字とひらがなは文字種境界で別トークンになる（「何」「か」は分離される）
        result = tokenize("RAGとは何か")
        self.assertEqual(result, ["rag", "とは", "何", "か"])

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(tokenize(""), [])

    def test_punctuation_only_returns_empty_list(self):
        self.assertEqual(tokenize("。、！？"), [])


class ChunkTextTest(unittest.TestCase):
    def test_short_text_becomes_single_chunk(self):
        result = chunk_text("これはテストです。今日は晴れです。")
        self.assertEqual(result, ["これはテストです。 今日は晴れです。"])

    def test_fragment_shorter_than_15_chars_is_discarded(self):
        self.assertEqual(chunk_text("短文。"), [])

    def test_fragment_of_exactly_15_chars_or_more_is_kept(self):
        text = "これは十分に長い文章のテストです。"  # 17文字
        self.assertEqual(len(text), 17)
        self.assertEqual(chunk_text(text), [text])

    def test_exceeding_chunk_size_splits_and_applies_overlap(self):
        # 1チャンク目がCHUNK_SIZE(200)を超えないギリギリの長さ、
        # 2チャンク目に入る内容を追加すると新チャンクへ分割され、
        # 新チャンクの先頭に前チャンク末尾CHUNK_OVERLAP(40)文字分の重複が入る。
        first = "あ" * 150 + "。"
        second = "い" * 150 + "。"
        result = chunk_text(first + second)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], first)
        # 2チャンク目 = 1チャンク目末尾CHUNK_OVERLAP(40)文字 + 2チャンク目本文
        self.assertEqual(result[1], first[-CHUNK_OVERLAP:] + second)
        self.assertEqual(len(result[1]), CHUNK_OVERLAP + len(second))

    def test_double_newline_paragraph_separator_does_not_force_split_when_within_size(self):
        # 2つの段落を結合してもCHUNK_SIZE以内なら1チャンクにまとめられる
        result = chunk_text("段落1についての説明文です。\n\n段落2についての説明文です。")
        self.assertEqual(len(result), 1)
        self.assertIn("段落1についての説明文です。", result[0])
        self.assertIn("段落2についての説明文です。", result[0])


class ComputeTfidfTest(unittest.TestCase):
    def test_vectors_are_l2_normalized(self):
        docs = [
            {"tokens": ["猫", "犬", "猫"]},
            {"tokens": ["犬", "鳥"]},
        ]
        vectors, _idf = compute_tfidf(docs)
        for vec in vectors:
            norm = math.sqrt(sum(v ** 2 for v in vec.values()))
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_term_appearing_in_all_docs_has_lower_idf_than_unique_term(self):
        docs = [
            {"tokens": ["共通語", "固有語a"]},
            {"tokens": ["共通語", "固有語b"]},
        ]
        _vectors, idf = compute_tfidf(docs)
        self.assertLess(idf["共通語"], idf["固有語a"])
        self.assertLess(idf["共通語"], idf["固有語b"])

    def test_term_frequency_within_doc_increases_raw_score_before_normalization(self):
        docs = [
            {"tokens": ["猫", "猫", "猫", "犬"]},
        ]
        vectors, _idf = compute_tfidf(docs)
        vec = vectors[0]
        self.assertGreater(vec["猫"], vec["犬"])

    def test_empty_docs_list_returns_empty_results(self):
        vectors, idf = compute_tfidf([])
        self.assertEqual(vectors, [])
        self.assertEqual(idf, {})


if __name__ == "__main__":
    unittest.main()
