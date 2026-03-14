# 実装ガイド（現在の llm-test）

このドキュメントは、現在の `llm-test` に実装されている
「ローカルLLMチャット + 長期記憶」の仕組みを、全体像から順に説明します。

## 1. 何を実装しているか

現在の実装は、以下を目的にしています。

- ローカルLLM（既定: `gemma3:4b`）でチャット
- 会話履歴と長期記憶を `SQLite` に保存
- コンテキストウィンドウを超えて、ユーザー情報や感情状態を再利用
- Web UI（ChatGPT風）とCLIの両方で利用

## 2. ファイル構成と役割

- `webapp.py`
  - FastAPIサーバー
  - Web UI配信（`/`）
  - API提供（`/api/chat`, `/api/history`, `/api/memories`）
- `chat.py`
  - CLIチャット実行
  - Web版と同様の記憶処理を端末で利用
- `memory.py`
  - DBスキーマ作成
  - 会話ログ・記憶の保存/更新/検索
  - 感情推定、記憶抽出、想起コンテキスト生成
- `static/index.html`, `static/styles.css`, `static/app.js`
  - Webフロントエンド（会話画面、送信、履歴表示、記憶表示）
- `README.md`
  - セットアップ・起動手順

## 3. データ保存設計（SQLite）

`memory.py` の `MemoryStore` が `memory.db` を管理します。

### 3.1 `turns` テーブル（会話履歴）

- `id`
- `user_id`
- `role` (`user` / `assistant`)
- `content`
- `created_at` (unix timestamp)

### 3.2 `memories` テーブル（長期記憶）

- `id`
- `user_id`
- `memory_type` (`profile` / `episodic` / `affective_state`)
- `content`
- `emotion`, `arousal`, `valence`
- `importance`
- `tags_json`
- `embedding_json`
- `created_at`, `updated_at`
- `last_accessed`, `access_count`
- `ttl_days`
- `archived`
- `source_turn_id`

## 4. チャット時の処理フロー

Web/CLIどちらも、概ね同じ流れです。

1. ユーザー入力を `turns` に保存
2. 感情推定（高速モード時はヒューリスティック優先）
3. 入力文を埋め込み化
4. 長期記憶を検索（類似度 + 重要度 + 鮮度 + 感情一致）
5. 想起結果をシステムプロンプトに注入してモデル生成
6. 生成文の反復除去クリーンアップ
7. アシスタント応答を `turns` に保存
8. 記憶抽出（高速モード時は軽量抽出）
9. `affective_state` を毎ターン保存
10. 一定ターンごとに忘却アーカイブ実行

## 5. 想起スコアの考え方

`recall()` 内で、主に以下を掛け合わせてランキングしています。

- 類似度（埋め込みコサイン、またはキーワード類似）
- 重要度 `importance`
- 鮮度（時間減衰）
- 直近アクセスのブースト
- 感情一致ブースト

これにより、単なる類似検索よりも「今の文脈で使うべき記憶」を優先します。

## 6. 重複抑制・忘却

### 6.1 重複抑制

- 同タイプ記憶で高類似（既定しきい値付近）の場合、`upsert` 更新
- 新規挿入を抑えて記憶の肥大化を減らす

### 6.2 忘却（アーカイブ）

- 古い/低重要/低参照の記憶を `archived = 1` に移行
- 想起対象から除外してノイズを減らす

## 7. 速度と品質のモード

### 7.1 高速化関連

- `--fast-mode`
  - 感情推定・記憶抽出で追加LLM呼び出しを減らす
- `--max-tokens`
  - 応答最大トークン
- `--num-ctx`
  - 生成時コンテキスト長
- `--memory-extract-interval`
  - 記憶抽出を N ターンごとに間引き

### 7.2 長文化

- `--long-form`
  - システム指示を長文説明寄りに変更
  - 生成トークン上限を実質的に拡張

## 8. 反復バグへの対策

モデルが同文反復を起こしやすいケースに対して、次を実装済みです。

- 生成パラメータ調整
  - `repeat_penalty`
  - `repeat_last_n`
  - `top_p`, `temperature`
- 出力後のテキストクリーンアップ
  - 連続重複行の除去
  - 明らかな長文ループの切り詰め

## 9. Web API 一覧

- `GET /`
  - Web UI（`static/index.html`）
- `POST /api/chat`
  - 入力を受けて応答返却、内部で記憶更新
- `GET /api/history?user_id=...&limit=...`
  - 会話履歴取得
- `GET /api/memories?user_id=...&limit=...`
  - 記憶一覧取得

## 10. 既知の制約

- 記憶抽出/感情推定はモデル品質の影響を受ける
- PII保護は最小限（実運用は追加対策推奨）
- 会話セッション管理は `user_id` ベースの簡易方式
- ストリーミング表示は未実装（レスポンスは一括返却）

## 11. 今後の拡張候補

- 明示的な「忘れて」API
- 記憶編集UI（重要度・タイプ修正）
- 感情推定の専用軽量モデル化
- ベクトルDB移行（大量データ対応）
- 応答ストリーミング（UX改善）

---

必要なら次に、このガイドを「非エンジニア向け版（図解寄り）」と
「実装者向け版（関数単位）」の2つに分けて作成できます。
