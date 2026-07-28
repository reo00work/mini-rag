"""
mini-rag/tests/test_server.py
server.py（ask.py のロジックを再利用するローカルWeb版）を検証する。

実サーバーをスレッドで起動し、urllib.request で実際にHTTPリクエストを送って
GET / と POST /ask の応答を確認する（標準ライブラリのみ・新規pip依存なし）。

実行方法:
    python3 -m unittest discover -s mini-rag/tests -v
"""
import json
import sys
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer_question, load_data, make_handler  # noqa: E402


class ServerHTTPTest(unittest.TestCase):
    """実際にHTTPサーバーを起動してエンドツーエンドで検証する。"""

    @classmethod
    def setUpClass(cls):
        cls.store = load_data()
        handler = make_handler(cls.store)
        # port=0 で空きポートをOSに割り当ててもらう（他プロセスとの衝突回避）
        cls.httpd = ThreadingHTTPServer(("localhost", 0), handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://localhost:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def _post_ask(self, question: str) -> tuple[int, dict]:
        data = json.dumps({"question": question}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/ask",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read().decode("utf-8"))

    def test_get_index_returns_html_with_form(self):
        with urllib.request.urlopen(f"{self.base_url}/") as res:
            self.assertEqual(res.status, 200)
            self.assertIn("text/html", res.headers.get("Content-Type", ""))
            body = res.read().decode("utf-8")
        self.assertIn("mini-rag", body)
        self.assertIn("<form", body)  # チャット入力フォームが存在する

    def test_ask_known_question_returns_answer_with_sources(self):
        status, body = self._post_ask("有給休暇は何日もらえますか？")
        self.assertEqual(status, 200)
        self.assertTrue(body["found"])
        self.assertTrue(body["answer"])
        self.assertTrue(body["sources_text"])
        self.assertIn("出典", body["sources_text"])

    def test_ask_empty_question_returns_prompt_without_search(self):
        status, body = self._post_ask("")
        self.assertEqual(status, 200)
        self.assertFalse(body["found"])
        self.assertEqual(body["sources_text"], "")

    def test_unknown_path_returns_404(self):
        req = urllib.request.Request(f"{self.base_url}/no-such-path")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)


class AnswerQuestionUnitTest(unittest.TestCase):
    """HTTPを介さずロジック単体を検証する。"""

    @classmethod
    def setUpClass(cls):
        cls.store = load_data()

    def test_answer_question_returns_expected_keys(self):
        result = answer_question("テレワークは週何日まで可能ですか？", self.store)
        self.assertEqual(
            set(result.keys()), {"question", "answer", "sources_text", "found"}
        )


if __name__ == "__main__":
    unittest.main()
