from __future__ import annotations

import argparse

from ollama import Client

from memory import MemoryEngine, MemoryRecord, MemoryStore, extract_forget_target


def system_prompt(long_form: bool) -> str:
    base = """あなたは親密であたたかい日本語チャットパートナーです。
以下のルールを守って回答してください。
- 長期記憶コンテキストがある場合は活用し、ユーザーに寄り添う。
- トーンはやさしく自然体。短い共感や相づちを入れてよい。
- ただし、べったりしすぎず、相手のペースを尊重する。
- 記憶は100%正しいとは限らないため、矛盾があればやわらかく確認する。
- 個人情報をむやみに再掲しない。"""
    if long_form:
        return base + "\n- 回答は詳しく、背景・理由・具体例まで含めて丁寧に説明する。"
    return base + "\n- 回答は読みやすく、基本は2〜6文で返す。"


def _clean_reply_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    deduped: list[str] = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    merged = "\n".join(deduped).strip()
    if not merged:
        return text.strip()

    # Guard against model repetition loops (same long fragment repeated many times).
    for size in (40, 30, 20):
        if len(merged) < size * 4:
            continue
        frag = merged[:size]
        if frag and merged.count(frag) >= 4:
            return merged[: max(120, size * 3)].strip()
    return merged


def run_chat(
    host: str,
    model: str,
    embedding_model: str,
    user_id: str,
    db_path: str,
    history_turns: int,
    fast_mode: bool,
    max_tokens: int,
    num_ctx: int,
    memory_extract_interval: int,
    long_form: bool,
) -> None:
    client = Client(host=host)
    store = MemoryStore(db_path=db_path)
    engine = MemoryEngine(client=client, chat_model=model, embedding_model=embedding_model, store=store)
    response_tokens = max(max_tokens, 512) if long_form else max_tokens

    print("Long-memory chat started. Type /exit to quit, /memory to inspect top memories.")
    turn_counter = 0

    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_text == "/memory":
            recalled = store.recall(
                user_id=user_id,
                query="user profile and current preferences",
                query_embedding=None,
                current_emotion="neutral",
                top_k=10,
            )
            if not recalled:
                print("Assistant> no memories.")
            else:
                print("Assistant> memories:")
                for r in recalled:
                    print(f"  - ({r['memory_type']}) {r['content']} [importance={r['importance']:.2f}]")
            continue

        user_turn_id = store.add_turn(user_id=user_id, role="user", content=user_text)
        forget_target = extract_forget_target(user_text)
        if forget_target is not None:
            if forget_target == "":
                assistant_text = "忘れる対象を教えてください。例: 「Pythonのこと忘れて」"
            elif forget_target == "__all__":
                archived = store.archive_all_memories(user_id=user_id)
                assistant_text = f"了解です。保存していた記憶を {archived} 件忘れました。"
            else:
                target_emb = engine.embed_text(forget_target)
                archived = store.archive_memories_by_query(
                    user_id=user_id,
                    query=forget_target,
                    query_embedding=target_emb,
                )
                if archived == 0:
                    assistant_text = f"「{forget_target}」に関連する記憶は見つかりませんでした。"
                else:
                    assistant_text = f"了解です。「{forget_target}」に関連する記憶を {archived} 件忘れました。"

            print(f"Assistant> {assistant_text}")
            store.add_turn(user_id=user_id, role="assistant", content=assistant_text)
            continue

        emotion, arousal, _ = engine.detect_emotion(user_text, use_llm=not fast_mode)
        query_emb = engine.embed_text(user_text)
        recalled = store.recall(
            user_id=user_id,
            query=user_text,
            query_embedding=query_emb,
            current_emotion=emotion,
            top_k=5,
        )
        memory_context = engine.build_memory_context(recalled)
        history = store.get_recent_turns(user_id=user_id, limit=history_turns)
        messages = [{"role": "system", "content": system_prompt(long_form) + "\n\n[Long-term memory]\n" + memory_context}] + history

        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "repeat_last_n": 128,
                "num_predict": response_tokens,
                "num_ctx": num_ctx,
            },
        )
        assistant_text = _clean_reply_text(response["message"]["content"])
        print(f"Assistant> {assistant_text}")

        store.add_turn(user_id=user_id, role="assistant", content=assistant_text)
        should_extract = (turn_counter % max(1, memory_extract_interval)) == 0
        extracted = (
            engine.extract_memories(
                user_text=user_text,
                assistant_text=assistant_text,
                use_llm=not fast_mode,
            )
            if should_extract
            else []
        )

        # Save user's affective state explicitly to preserve emotion across context-window boundaries.
        extracted.append(
            MemoryRecord(
                memory_type="affective_state",
                content=f"User emotional state around this turn: {emotion} (arousal={arousal:.2f})",
                emotion=emotion,
                arousal=arousal,
                valence=0.0,
                importance=0.5,
                ttl_days=14,
                tags=["affect"],
            )
        )

        for record in extracted:
            emb = engine.embed_text(record.content)
            store.upsert_memory(
                user_id=user_id,
                record=record,
                embedding=emb,
                source_turn_id=user_turn_id,
            )

        turn_counter += 1
        if turn_counter % 10 == 0:
            archived = store.decay_and_archive(user_id=user_id)
            if archived:
                print(f"Assistant> maintenance: archived {archived} stale memories.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local LLM chat with long-term affective memory")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--model", default="gemma3:4b", help="chat model name in Ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="embedding model name in Ollama")
    parser.add_argument("--user-id", default="default-user", help="logical user id")
    parser.add_argument("--db-path", default="memory.db", help="sqlite db path")
    parser.add_argument("--history-turns", type=int, default=8, help="recent turns to include")
    parser.add_argument("--fast-mode", action="store_true", help="reduce extra LLM calls for lower latency")
    parser.add_argument("--max-tokens", type=int, default=192, help="max generated tokens per response")
    parser.add_argument("--num-ctx", type=int, default=2048, help="context window used by generation")
    parser.add_argument("--long-form", action="store_true", help="generate longer, detailed responses")
    parser.add_argument(
        "--memory-extract-interval",
        type=int,
        default=2,
        help="extract long-term memory every N turns (higher is faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_chat(
        host=args.host,
        model=args.model,
        embedding_model=args.embedding_model,
        user_id=args.user_id,
        db_path=args.db_path,
        history_turns=args.history_turns,
        fast_mode=args.fast_mode,
        max_tokens=args.max_tokens,
        num_ctx=args.num_ctx,
        memory_extract_interval=args.memory_extract_interval,
        long_form=args.long_form,
    )
