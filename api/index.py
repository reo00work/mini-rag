#!/usr/bin/env python3
"""
mini-rag/api/index.py
Vercel Python Runtime向けエントリーポイント。

server.py（ローカル実行用: ThreadingHTTPServerで起動）と同じロジックを、
Vercelのサーバーレス関数として動かすための薄いラッパー。

Vercelの Python Runtime は、api/配下のファイルに
`BaseHTTPRequestHandler` を継承した `handler` という名前のクラスを
置くと、そのままサーバーレス関数としてルーティングする仕様
（1リクエスト = 1関数呼び出し。コンテナは複数リクエストで再利用されうる）。

ロジック本体（load_store/search/format_results/llm_summarize）は
ask.py をそのまま import して再利用する。ask.py 自体は無変更。
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# api/ の1つ上（mini-rag/ ディレクトリ）を sys.path に追加し、
# 同ディレクトリの ask.py をそのまま import できるようにする。
# server.py の `sys.path.insert(0, str(BASE))` と同じ考え方。
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from ask import (  # noqa: E402
    DEFAULT_TOP_K,
    STORE_PATH,
    format_results,
    llm_summarize,
    load_store,
    search,
)

# モジュールロード時（コールドスタート時）に1回だけ vector_store.json を読み込む。
# 同一コンテナがウォームなまま次のリクエストを処理する場合はこの再読み込みは発生しない。
_STORE = load_store(STORE_PATH)


def answer_question(question: str, store: dict) -> dict:
    """server.py と同一ロジック（検索→LLM要約→出典整形）。"""
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
  .demo-intro { background: #eef4ff; border: 1px solid #cfe0ff; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; line-height: 1.7; color: #333; }
  .demo-intro dt { font-weight: bold; margin-top: 8px; }
  .demo-intro dt:first-child { margin-top: 0; }
  .demo-intro dd { margin: 0 0 2px 0; }
</style>
</head>
<body>
<h1>mini-rag デモ — 社内文書に質問してください</h1>
<div class="demo-intro">
  <dl>
    <dt>想定シーン</dt>
    <dd>社内規程・マニュアル・FAQを社員がその都度検索したり総務・情シスに問い合わせたりせず、チャット形式で聞ける状態にしたい場面を想定しています。</dd>
    <dt>読み込んでいるデータ</dt>
    <dd>docs/ 配下の架空社内文書5点（社員ハンドブック・経費申請FAQ・ITヘルプデスク利用マニュアル・情報セキュリティポリシー・テレワーク勤務規程）です。休暇制度や経費申請の期限、パスワード要件など、実在の会社を模した内容ですが全てデモ用の架空データです。</dd>
    <dt>見てほしいポイント</dt>
    <dd>TF-IDFで質問と文書チャンクの類似度を計算し、根拠となった出典（ファイル名・チャンク番号）付きで抜粋を返します。APIキー未設定でもこの抜粋+出典表示だけで動作し、ANTHROPIC_API_KEY / OPENAI_API_KEYを設定すると同じ検索結果をもとにLLMが自然文で要約するようになります。</dd>
  </dl>
</div>
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


class handler(BaseHTTPRequestHandler):
    """Vercel Python Runtime が呼び出すエントリークラス（クラス名は固定で `handler`）。"""

    def log_message(self, fmt, *args):
        pass

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
        result = answer_question(question, _STORE)
        self._write_json(200, result)
