from __future__ import annotations

import argparse
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ollama import Client
from pydantic import BaseModel, Field

from memory import MemoryEngine, MemoryRecord, MemoryStore


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

    for size in (40, 30, 20):
        if len(merged) < size * 4:
            continue
        frag = merged[:size]
        if frag and merged.count(frag) >= 4:
            return merged[: max(120, size * 3)].strip()
    return merged


class ChatRequest(BaseModel):
    user_id: str = Field(default="default-user")
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str


class ChatRuntime:
    def __init__(
        self,
        host: str,
        model: str,
        embedding_model: str,
        db_path: str,
        history_turns: int,
        fast_mode: bool,
        max_tokens: int,
        num_ctx: int,
        memory_extract_interval: int,
        long_form: bool,
    ) -> None:
        self.client = Client(host=host)
        self.store = MemoryStore(db_path=db_path)
        self.engine = MemoryEngine(
            client=self.client,
            chat_model=model,
            embedding_model=embedding_model,
            store=self.store,
        )
        self.model = model
        self.history_turns = history_turns
        self.fast_mode = fast_mode
        self.max_tokens = max(max_tokens, 512) if long_form else max_tokens
        self.num_ctx = num_ctx
        self.memory_extract_interval = max(1, memory_extract_interval)
        self.long_form = long_form
        self.lock = threading.Lock()
        self.turn_count_by_user: dict[str, int] = {}

    def process_message(self, user_id: str, user_text: str) -> str:
        with self.lock:
            user_turn_id = self.store.add_turn(user_id=user_id, role="user", content=user_text)
            emotion, arousal, _ = self.engine.detect_emotion(user_text, use_llm=not self.fast_mode)
            query_emb = self.engine.embed_text(user_text)
            recalled = self.store.recall(
                user_id=user_id,
                query=user_text,
                query_embedding=query_emb,
                current_emotion=emotion,
                top_k=5,
            )
            memory_context = self.engine.build_memory_context(recalled)
            history = self.store.get_recent_turns(user_id=user_id, limit=self.history_turns)
            messages = [{"role": "system", "content": system_prompt(self.long_form) + "\n\n[Long-term memory]\n" + memory_context}] + history

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "repeat_last_n": 128,
                    "num_predict": self.max_tokens,
                    "num_ctx": self.num_ctx,
                },
            )
            assistant_text = _clean_reply_text(response["message"]["content"])
            self.store.add_turn(user_id=user_id, role="assistant", content=assistant_text)

            turn_count = self.turn_count_by_user.get(user_id, 0)
            should_extract = (turn_count % self.memory_extract_interval) == 0
            extracted = (
                self.engine.extract_memories(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    use_llm=not self.fast_mode,
                )
                if should_extract
                else []
            )
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
                emb = self.engine.embed_text(record.content)
                self.store.upsert_memory(
                    user_id=user_id,
                    record=record,
                    embedding=emb,
                    source_turn_id=user_turn_id,
                )

            self.turn_count_by_user[user_id] = turn_count + 1
            if self.turn_count_by_user[user_id] % 10 == 0:
                self.store.decay_and_archive(user_id=user_id)
            return assistant_text


def create_app(runtime: ChatRuntime) -> FastAPI:
    app = FastAPI(title="Long Memory Chat")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse("static/index.html")

    @app.get("/api/history")
    def history(user_id: str = "default-user", limit: int = 100) -> dict:
        safe_limit = max(1, min(500, limit))
        return {"messages": runtime.store.get_turns(user_id=user_id, limit=safe_limit)}

    @app.get("/api/memories")
    def memories(user_id: str = "default-user", limit: int = 50) -> dict:
        safe_limit = max(1, min(200, limit))
        return {"memories": runtime.store.get_memories(user_id=user_id, limit=safe_limit)}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        text = payload.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="message is required")
        reply = runtime.process_message(user_id=payload.user_id, user_text=text)
        return ChatResponse(reply=reply)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web chat app with long-term memory")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--model", default="gemma3:4b", help="chat model in Ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="embedding model in Ollama")
    parser.add_argument("--db-path", default="memory.db", help="SQLite DB path")
    parser.add_argument("--history-turns", type=int, default=8, help="recent turns to include as short-term context")
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
    parser.add_argument("--port", type=int, default=8000, help="web server port")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    runtime = ChatRuntime(
        host=args.host,
        model=args.model,
        embedding_model=args.embedding_model,
        db_path=args.db_path,
        history_turns=args.history_turns,
        fast_mode=args.fast_mode,
        max_tokens=args.max_tokens,
        num_ctx=args.num_ctx,
        memory_extract_interval=args.memory_extract_interval,
        long_form=args.long_form,
    )
    app = create_app(runtime)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
