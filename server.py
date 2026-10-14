#!/usr/bin/env python3
"""
mini-rag/server.py
mini-rag CLI（ask.py）をブラウザから使える最小Web版にするローカルサーバー。

目的:
  ask.py はターミナル操作（python3 ask.py '質問文'）でしか動かないため、
  非エンジニアの相手に実演を見せられない。
  本サーバーは ask.py の検索・出典整形・LLM要約ロジックをそのまま再利用し、
  ブラウザから質問→回答を返すチャットUIを提供する（ai-me/server.py と同型）。

依存: 標準ライブラリのみ（http.server）。新規pipパッケージは追加しない。
ANTHROPIC_API_KEY / OPENAI_API_KEY 未設定でも ask.py と同じ stub モード
（抜粋 + 出典）で動作する。

使い方:
    python3 server.py
    python3 server.py --port 8765
    → ブラウザで http://localhost:8765 を開き、質問を入力する

他プロセス・テストからの利用:
    from server import answer_question, load_data
    store = load_data()
    result = answer_question("質問文", store)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from ask import (  # noqa: E402
    DEFAULT_TOP_K,
    STORE_PATH,
    format_results,
    llm_summarize,
    load_store,
    search,
)

DEFAULT_PORT = 8000


def load_data() -> dict:
    """vector_store.json を読み込む（ask.py の CLI と同一のロード関数を使う）。"""
    return load_store(STORE_PATH)


def answer_question(question: str, store: dict) -> dict:
    """ask.py の main() が持つ検索→LLM要約→出典整形ロジックをWeb向けに再利用する。"""
    question = (question or "").strip()
    if not question:
        return {
            "question": question,
            "answer": "質問を入力してください。",
            "sources_text": "",
            "found": False,
        }

    results = search(question, store, DEFAULT_TOP_K)

    if not results:
        return {
            "question": question,
            "answer": "関連する情報が見つかりませんでした。キーワードを変えてお試しください。",
            "sources_text": "",
            "found": False,
        }

    summary = llm_summarize(question, results)
    answer = summary or "（APIキー未設定のため抜粋 + 出典を表示します。ANTHROPIC_API_KEY / OPENAI_API_KEY を設定すると自然文の要約が生成されます）"

    return {
        "question": question,
        "answer": answer,
        "sources_text": format_results(results),
        "found": True,
    }


def render_index_html() -> bytes:
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>mini-rag — 社内文書Q&Aデモ</title>
<style>
  body { font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background: #f7f7f8; }
  h1 { font-size: 18px; }
  #log { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 240px; margin-bottom: 12px; }
  .msg { margin-bottom: 14px; white-space: pre-wrap; line-height: 1.6; }
  .msg.user { text-align: right; color: #1a1a1a; }
  .msg.bot { color: #1a1a1a; }
  .sources { font-size: 12px; color: #666; margin-top: 4px; white-space: pre-wrap; }
  form { display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; }
  button { padding: 10px 18px; border: none; border-radius: 6px; background: #2d6cdf; color: #fff; cursor: pointer; }
  button:disabled { background: #999; cursor: default; }
</style>
</head>
<body>
<h1>mini-rag デモ — 社内文書に質問してください</h1>
<div id="log"><div class="msg bot">社員ハンドブック・セキュリティポリシー・経費申請FAQ等について質問できます。（例：「有給休暇は何日もらえますか？」）</div></div>
<form id="form">
  <input type="text" id="question" placeholder="質問を入力してください" autocomplete="off">
  <button type="submit" id="send">送信</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('question');
const send = document.getElementById('send');

function addMessage(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, 'user');
  input.value = '';
  send.disabled = true;
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await res.json();
    const botDiv = addMessage(data.answer, 'bot');
    if (data.sources_text) {
      const srcDiv = document.createElement('div');
      srcDiv.className = 'sources';
      srcDiv.textContent = data.sources_text;
      botDiv.appendChild(document.createElement('br'));
      botDiv.appendChild(srcDiv);
    }
  } catch (err) {
    addMessage('通信エラーが発生しました。サーバーが起動しているか確認してください。', 'bot');
  } finally {
    send.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""
    return html.encode("utf-8")


def make_handler(store: dict):
    """store をクロージャで束縛した BaseHTTPRequestHandler を返す。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 標準エラー出力への毎リクエストログを抑制（動作確認時のノイズ削減）

        def _write_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = render_index_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/ask":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                payload = {}
            question = payload.get("question", "") if isinstance(payload, dict) else ""
            result = answer_question(question, store)
            self._write_json(200, result)

    return Handler


def run(port: int = DEFAULT_PORT) -> None:
    store = load_data()
    handler = make_handler(store)
    server = ThreadingHTTPServer(("localhost", port), handler)
    print(f"mini-rag Web版を起動しました: http://localhost:{port}")
    print("Ctrl+C で終了します。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_port(argv: list[str]) -> int:
    port = DEFAULT_PORT
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    return port


if __name__ == "__main__":
    run(parse_port(sys.argv[1:]))
