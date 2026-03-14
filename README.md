# llm-test: Local LLM long-term memory chat

`Gemma 3 4B` を使ったローカルチャット向けに、コンテキストウィンドウを超えて
「ユーザー情報」と「感情状態」を保持する最小実装です。

## Features

- 長期記憶の分離: `profile` / `episodic` / `affective_state`
- 毎ターンの記憶抽出（LLM JSON出力）
- 埋め込み検索 + 感情一致 + 重要度 + 新鮮度で想起スコアリング
- 忘却（時間減衰）と古い記憶のアーカイブ
- `SQLite` 1ファイルで管理
- 返答トーンは「親密であたたかい会話」寄りに調整済み

## Requirements

- Python 3.10+
- Ollama
- Ollamaで以下モデルが利用可能
  - チャット: `gemma3:4b`
  - 埋め込み: `nomic-embed-text`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

モデル例:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Run

```bash
python chat.py --model gemma3:4b --embedding-model nomic-embed-text
```

低遅延で使う例:

```bash
python chat.py --model gemma3:4b --embedding-model nomic-embed-text --fast-mode --max-tokens 128 --num-ctx 1536 --memory-extract-interval 3
```

長文で詳しく返す例:

```bash
python chat.py --model gemma3:4b --embedding-model nomic-embed-text --long-form --max-tokens 768 --num-ctx 3072
```

## Run Web App (ChatGPT-like UI)

```bash
python webapp.py --model gemma3:4b --embedding-model nomic-embed-text --port 8000
```

低遅延で使う例:

```bash
python webapp.py --model gemma3:4b --embedding-model nomic-embed-text --port 8000 --fast-mode --max-tokens 128 --num-ctx 1536 --memory-extract-interval 3
```

長文で詳しく返す例:

```bash
python webapp.py --model gemma3:4b --embedding-model nomic-embed-text --port 8000 --long-form --max-tokens 768 --num-ctx 3072
```

ブラウザで `http://localhost:8000` を開くと、Webチャットを利用できます。

- 左サイドバーで `User ID` を切り替えるとユーザーごとに会話・記憶を分離
- `記憶を表示` で保存済みメモリを確認
- 見た目はChatGPT風のメッセージバブル + 下部コンポーザー
- チャットで `「〜を忘れて」` と送ると関連記憶をアーカイブ（想起対象から除外）

オプション:

- `--user-id`: ユーザー識別子（デフォルト: `default-user`）
- `--db-path`: 記憶DBファイル（デフォルト: `memory.db`）
- `--history-turns`: 直近会話の投入ターン数（デフォルト: `8`）
- `--fast-mode`: 感情判定/記憶抽出の追加LLM呼び出しを減らす
- `--max-tokens`: 1応答の最大生成トークン（小さいほど速い）
- `--num-ctx`: 生成時コンテキスト長（小さいほど速い）
- `--memory-extract-interval`: 記憶抽出をNターンごとに実行（大きいほど速い）
- `--long-form`: 回答を長文・詳細寄りにする（内部トークン上限も拡張）

チャット中コマンド:

- `/memory`: 現在想起される記憶を表示
- `/exit`: 終了

忘却リクエストの例:

- `Pythonのこと忘れて`
- `昨日の話を忘れて`
- `全部忘れて`

## Notes

- 感情推定・記憶抽出はLLM出力に依存するため、誤判定が起こり得ます。
- 実運用では、PIIフィルタや「忘れて」APIの実装を追加してください。
