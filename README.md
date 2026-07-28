# mini-rag — APIキー不要で動く最小RAGデモ

「社内文書 RAG チャットボット」の最小実証。  
**ローカルで完結**（課金 API キー不要）で RAG の仕組みを動かして見せられるポートフォリオ用デモ。

---

## これは何か

RAG（Retrieval-Augmented Generation）とは、LLM に社内文書を「丸暗記させる」のではなく、**質問のたびに関連文書を検索して根拠付きで答える**アーキテクチャです。

```
[質問] → [ベクトル検索] → [関連チャンク取得] → [LLM で要約] → [出典付き回答]
```

このデモでは以下を **純粋な Python 標準ライブラリだけ** で実現しています：

| コンポーネント | 実装 |
|---|---|
| 埋め込みモデル | TF-IDF（ライブラリなし・ゼロ依存） |
| ベクトルストア | JSON ファイル（`vector_store.json`） |
| 検索 | コサイン類似度 |
| LLM 要約 | **APIキー無しは抜粋 stub** / APIキーあれば Claude or GPT |

---

## ファイル構成

```
mini-rag/
├── docs/                      # ダミー社内文書（架空・デモ用）
│   ├── employee_handbook.txt       社員ハンドブック
│   ├── it_security_policy.txt      情報セキュリティポリシー
│   ├── expense_claim_faq.txt       経費申請 FAQ
│   ├── remote_work_rules.txt       テレワーク勤務規程
│   └── it_helpdesk_manual.txt      IT ヘルプデスクマニュアル
├── ingest.py                  # ドキュメントをベクトル化して保存
├── ask.py                     # CLI 質問 → 類似検索 → 出典付き回答
├── server.py                  # ask.py のロジックを再利用したローカルWebサーバー（チャットUI＋/ask API）
├── vector_store.json          # ingest.py が生成するローカルベクトルDB
└── README.md                  # このファイル
```

---

## 動かし方

### 前提
- Python 3.10 以上
- 追加パッケージのインストール **不要**（標準ライブラリのみ）

### 手順

**Step 1: 文書を取り込む（ingest）**

```bash
cd mini-rag
python3 ingest.py
```

出力例:
```
対象ファイル: 5 件
  employee_handbook.txt: 6 チャンク
  expense_claim_faq.txt: 9 チャンク
  it_helpdesk_manual.txt: 11 チャンク
  it_security_policy.txt: 8 チャンク
  remote_work_rules.txt: 9 チャンク
チャンク合計: 43 件

ベクトルストア保存完了: .../vector_store.json
  チャンク数: 43, 語彙数: 837
```

**Step 2: 質問する（ask）**

```bash
python3 ask.py '有給休暇は何日もらえますか？'
```

出力例（APIキー未設定・stub モード）:
```
質問: 有給休暇は何日もらえますか？

【回答（抜粋 + 出典）】

─── 1位 (スコア: 0.385) ───
出典: employee_handbook.txt  (chunk #2)
2.2 有給休暇
入社後6ヶ月が経過した時点で10日の有給休暇が付与されます。その後、勤続年数に応じて
最大20日まで付与日数が増加します。有給休暇の取得は原則として事前申請（3営業日前）が必要です。...

─── 2位 (スコア: 0.243) ───
出典: employee_handbook.txt  (chunk #1)
...（以下略）
```

---

## サンプル質問集

```bash
# 有給・休暇
python3 ask.py '有給休暇は何日もらえますか？'
python3 ask.py '育児休業はいつまで取れますか？'

# テレワーク
python3 ask.py 'テレワークは週何日まで可能ですか？'
python3 ask.py '在宅勤務中の通信費は支給されますか？'

# セキュリティ
python3 ask.py 'パスワードの最低文字数は何文字ですか？'
python3 ask.py 'パソコンを紛失した場合はどうすればよいですか？'

# 経費
python3 ask.py '書籍代は経費申請できますか？'
python3 ask.py '海外出張のホテル代の上限はいくらですか？'

# IT ヘルプデスク
python3 ask.py '社内Wifiに繋がらない場合はどうすれば良いですか？'
python3 ask.py 'パスワードをリセットしたい'
```

---

## LLM 要約を有効にする（任意）

APIキーを設定するとドキュメント抜粋をもとに自然文の回答が生成されます：

```bash
# Anthropic Claude（推奨）
export ANTHROPIC_API_KEY=sk-ant-...
python3 ask.py 'テレワークは週何日まで？'

# または OpenAI
export OPENAI_API_KEY=sk-...
python3 ask.py 'テレワークは週何日まで？'
```

キーが未設定の場合は抜粋 + 出典のみ返します（完全にローカルで動作）。

---

## Web版（ローカル・依存ライブラリなし）

CLIだと非エンジニアの相手（受託営業・ここなら問い合わせ対応）には見せられないため、ブラウザで
実際に質問→回答をやり取りできる最小Webサーバーを同梱している
（`ask.py` の検索・要約ロジックをそのまま再利用、新規pip依存なし）。

```bash
cd mini-rag
python3 server.py
# または任意のポートを指定
python3 server.py --port 8765
```

起動後、ブラウザで `http://localhost:8000` を開くと入力欄付きのチャットUIが表示され、
そのまま質問を打ち込んで回答（出典つき）を受け取れる。`Ctrl+C` で停止。

API経由で直接叩く場合（動作確認・自動テスト向け）:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "有給休暇は何日もらえますか？"}'
```

`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` 未設定でもCLI同様stubモード（抜粋 + 出典）で動作する。

---

## source-lens（本番 RAG）との違い

| | mini-rag（このデモ） | source-lens（製品） |
|---|---|---|
| 目的 | RAG の仕組みの最小実証 | プロダクションRAGチャットボット |
| 埋め込み | TF-IDF（依存なし） | Voyage AI（高精度ベクトル） |
| ベクトルDB | ローカル JSON | Supabase pgvector |
| 精度 | 概念実証レベル | Hit@1 100%・回答品質94% |
| LLM | stub + 任意API | Claude 3.5 Sonnet |

mini-rag は「RAG をゼロから説明できる・動かして見せられる」ためのデモです。  
本番品質の RAG 実装は source-lens で提供しています。

---

## このデモで示せる技術

- **チャンク分割**（セマンティックな段落分割・オーバーラップ設計）
- **TF-IDF ベクトル化**（IDF 計算・L2 正規化・コサイン類似度）
- **出典追跡**（ファイル名・チャンク番号の保持と表示）
- **LLM の差し替え可能な設計**（環境変数でのプロバイダー切り替え）
- **ゼロ依存での完全ローカル動作**（パッケージインストール不要）
