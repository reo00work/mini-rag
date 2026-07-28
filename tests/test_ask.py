"""
mini-rag/tests/test_ask.py
ask.py の tokenize() / vectorize_query() / cosine_sim() / search() / format_results() を検証する。

LLM要約（llm_summarize、ANTHROPIC_API_KEY/OPENAI_API_KEY使用）は対象外
（APIキー未設定環境を前提とし、検索・整形のルールベース部分のみ検証する）。

実行方法:
    python3 -m unittest discover -s mini-rag/tests -v
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ask import (  # noqa: E402
    cosine_sim,
    format_results,
    search,
    tokenize,
    vectorize_query,
)


class TokenizeTest(unittest.TestCase):
    def test_mixed_japanese_and_english(self):
        # 漢字とひらがなは文字種境界で別トークンになる（「何」「か」は分離される）
        result = tokenize("RAGとは何か")
        self.assertEqual(result, ["rag", "とは", "何", "か"])

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(tokenize(""), [])


class VectorizeQueryTest(unittest.TestCase):
    def test_result_is_l2_normalized(self):
        idf = {"猫": 1.5, "犬": 0.8}
        vec = vectorize_query(["猫", "犬"], idf)
        norm = math.sqrt(sum(v ** 2 for v in vec.values()))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_unknown_token_uses_small_default_idf_weight(self):
        idf = {"猫": 1.5}
        vec = vectorize_query(["未知語"], idf)
        # 未知語も微小スコア(idf.get未ヒットでも0.1扱い)でベクトルに含まれる
        self.assertIn("未知語", vec)
        self.assertGreater(vec["未知語"], 0.0)

    def test_repeated_token_has_higher_raw_weight_than_singleton(self):
        idf = {"猫": 1.0, "犬": 1.0}
        vec = vectorize_query(["猫", "猫", "犬"], idf)
        self.assertGreater(vec["猫"], vec["犬"])


class CosineSimTest(unittest.TestCase):
    def test_identical_vectors_have_similarity_one(self):
        vec = {"猫": 0.6, "犬": 0.8}
        self.assertAlmostEqual(cosine_sim(vec, vec), 1.0, places=9)

    def test_disjoint_vocabularies_have_zero_similarity(self):
        q_vec = {"猫": 1.0}
        doc_vec = {"犬": 1.0}
        self.assertEqual(cosine_sim(q_vec, doc_vec), 0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        q_vec = {"猫": 0.7, "犬": 0.7}
        doc_vec = {"猫": 0.7, "鳥": 0.7}
        sim = cosine_sim(q_vec, doc_vec)
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)


def _make_store():
    # search() が参照する store 構造を模した最小フィクスチャ
    idf = {"猫": 1.0, "犬": 1.0, "鳥": 1.0}
    chunks = [
        {"id": "a::chunk0", "source": "a.txt", "chunk_index": 0, "text": "猫についての説明文です。"},
        {"id": "b::chunk0", "source": "b.txt", "chunk_index": 0, "text": "犬についての説明文です。"},
        {"id": "c::chunk0", "source": "c.txt", "chunk_index": 0, "text": "鳥についての説明文です。"},
    ]
    vectors = [
        {"猫": 1.0},
        {"犬": 1.0},
        {"鳥": 1.0},
    ]
    return {"idf": idf, "chunks": chunks, "vectors": vectors}


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_empty_question_tokens_returns_empty_list(self):
        self.assertEqual(search("。！？", self.store), [])

    def test_matching_query_returns_relevant_chunk_first(self):
        results = search("猫", self.store, top_k=3)
        self.assertGreaterEqual(len(results), 1)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk["source"], "a.txt")
        self.assertGreater(top_score, 0.0)

    def test_top_k_limits_result_count(self):
        results = search("猫犬鳥", self.store, top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_zero_or_negative_scores_are_excluded(self):
        # クエリに一致する語彙が無いdocはスコア0となり除外される
        results = search("未知語のみ", self.store, top_k=3)
        self.assertEqual(results, [])


class FormatResultsTest(unittest.TestCase):
    def test_includes_source_and_score_and_excerpt(self):
        chunk = {"source": "a.txt", "chunk_index": 0, "text": "猫についての説明文です。"}
        output = format_results([(chunk, 0.987)])
        self.assertIn("a.txt", output)
        self.assertIn("chunk #0", output)
        self.assertIn("0.987", output)
        self.assertIn("猫についての説明文です。", output)

    def test_show_score_false_hides_score(self):
        chunk = {"source": "a.txt", "chunk_index": 0, "text": "猫についての説明文です。"}
        output = format_results([(chunk, 0.987)], show_score=False)
        self.assertNotIn("0.987", output)

    def test_long_text_is_truncated_with_ellipsis(self):
        long_text = "あ" * 400
        chunk = {"source": "a.txt", "chunk_index": 0, "text": long_text}
        output = format_results([(chunk, 0.5)])
        self.assertIn("...", output)
        self.assertNotIn("あ" * 301, output)


if __name__ == "__main__":
    unittest.main()
