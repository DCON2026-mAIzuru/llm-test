from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any


def now_ts() -> int:
    return int(time.time())


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])
    raise ValueError("Invalid JSON response")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9ぁ-んァ-ヶ一-龠]{2,}", text.lower()))


def keyword_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = math.sqrt(len(ta) * len(tb))
    return inter / denom if denom else 0.0


@dataclass
class MemoryRecord:
    memory_type: str
    content: str
    emotion: str
    arousal: float
    valence: float
    importance: float
    ttl_days: int | None = None
    tags: list[str] | None = None


class MemoryStore:
    def __init__(self, db_path: str = "memory.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT NOT NULL,
                arousal REAL NOT NULL,
                valence REAL NOT NULL,
                importance REAL NOT NULL,
                tags_json TEXT,
                embedding_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_accessed INTEGER,
                access_count INTEGER NOT NULL DEFAULT 0,
                ttl_days INTEGER,
                archived INTEGER NOT NULL DEFAULT 0,
                source_turn_id INTEGER
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user_created ON memories(user_id, created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user_type ON memories(user_id, memory_type)"
        )
        self.conn.commit()

    def add_turn(self, user_id: str, role: str, content: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO turns(user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, now_ts()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_recent_turns(self, user_id: str, limit: int = 8) -> list[dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT role, content
            FROM turns
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = list(reversed(cur.fetchall()))
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_turns(self, user_id: str, limit: int = 100) -> list[dict[str, str | int]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, role, content, created_at
            FROM turns
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = list(reversed(cur.fetchall()))
        return [
            {
                "id": int(r["id"]),
                "role": str(r["role"]),
                "content": str(r["content"]),
                "created_at": int(r["created_at"]),
            }
            for r in rows
        ]

    def get_memories(self, user_id: str, limit: int = 50) -> list[dict[str, str | float | int]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, memory_type, content, emotion, importance, created_at, archived
            FROM memories
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [
            {
                "id": int(r["id"]),
                "memory_type": str(r["memory_type"]),
                "content": str(r["content"]),
                "emotion": str(r["emotion"]),
                "importance": float(r["importance"]),
                "created_at": int(r["created_at"]),
                "archived": int(r["archived"]),
            }
            for r in cur.fetchall()
        ]

    def _find_duplicate(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        embedding: list[float] | None,
        threshold: float = 0.92,
    ) -> int | None:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, content, embedding_json
            FROM memories
            WHERE user_id = ? AND memory_type = ? AND archived = 0
            """,
            (user_id, memory_type),
        )
        for row in cur.fetchall():
            if embedding:
                raw = row["embedding_json"]
                if raw:
                    sim = cosine_similarity(embedding, json.loads(raw))
                    if sim >= threshold:
                        return int(row["id"])
            else:
                sim = keyword_similarity(content, row["content"])
                if sim >= 0.8:
                    return int(row["id"])
        return None

    def upsert_memory(
        self,
        user_id: str,
        record: MemoryRecord,
        embedding: list[float] | None,
        source_turn_id: int | None,
    ) -> int:
        cur = self.conn.cursor()
        dup_id = self._find_duplicate(
            user_id=user_id,
            memory_type=record.memory_type,
            content=record.content,
            embedding=embedding,
        )
        ts = now_ts()
        tags_json = json.dumps(record.tags or [], ensure_ascii=False)
        emb_json = json.dumps(embedding) if embedding else None
        if dup_id is not None:
            cur.execute(
                """
                UPDATE memories
                SET content = ?, emotion = ?, arousal = ?, valence = ?, importance = MAX(importance, ?),
                    tags_json = ?, embedding_json = COALESCE(?, embedding_json), updated_at = ?, ttl_days = ?
                WHERE id = ?
                """,
                (
                    record.content,
                    record.emotion,
                    record.arousal,
                    record.valence,
                    record.importance,
                    tags_json,
                    emb_json,
                    ts,
                    record.ttl_days,
                    dup_id,
                ),
            )
            self.conn.commit()
            return dup_id

        cur.execute(
            """
            INSERT INTO memories(
                user_id, memory_type, content, emotion, arousal, valence, importance, tags_json,
                embedding_json, created_at, updated_at, ttl_days, source_turn_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                record.memory_type,
                record.content,
                record.emotion,
                float(record.arousal),
                float(record.valence),
                float(record.importance),
                tags_json,
                emb_json,
                ts,
                ts,
                record.ttl_days,
                source_turn_id,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recall(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None,
        current_emotion: str,
        top_k: int = 5,
    ) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM memories
            WHERE user_id = ? AND archived = 0
            """,
            (user_id,),
        )
        now = now_ts()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in cur.fetchall():
            ttl_days = row["ttl_days"]
            age_days = max(0.0, (now - row["created_at"]) / 86400.0)
            if ttl_days is not None and age_days > ttl_days:
                continue

            emb = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            if query_embedding and emb:
                sim = max(0.0, cosine_similarity(query_embedding, emb))
            else:
                sim = max(0.0, keyword_similarity(query, row["content"]))

            importance = float(row["importance"])
            freshness = math.exp(-age_days / 30.0)
            accessed = row["last_accessed"] or row["created_at"]
            hours_since_access = max(0.0, (now - accessed) / 3600.0)
            recency_boost = 1.2 if hours_since_access <= 24 else 1.0
            mood_match = 1.15 if row["emotion"] == current_emotion and current_emotion != "neutral" else 1.0

            score = max(sim, 0.05) * (0.4 + 0.6 * importance) * (0.5 + 0.5 * freshness) * recency_boost * mood_match
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [row for _, row in scored[:top_k] if _ > 0.08]
        for row in picked:
            cur.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, row["id"]),
            )
        self.conn.commit()
        return picked

    def decay_and_archive(self, user_id: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, importance, created_at, access_count
            FROM memories
            WHERE user_id = ? AND archived = 0
            """,
            (user_id,),
        )
        now = now_ts()
        archive_count = 0
        for row in cur.fetchall():
            age_days = max(0.0, (now - row["created_at"]) / 86400.0)
            decay = math.exp(-age_days / 60.0)
            effective = float(row["importance"]) * decay
            should_archive = effective < 0.15 and int(row["access_count"]) < 2 and age_days > 30
            if should_archive:
                cur.execute("UPDATE memories SET archived = 1 WHERE id = ?", (row["id"],))
                archive_count += 1
        self.conn.commit()
        return archive_count


class MemoryEngine:
    def __init__(self, client: Any, chat_model: str, embedding_model: str, store: MemoryStore) -> None:
        self.client = client
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.store = store

    def embed_text(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        try:
            result = self.client.embed(model=self.embedding_model, input=text)
            emb = result.get("embeddings", [])
            if emb and isinstance(emb[0], list):
                return emb[0]
        except Exception:
            pass
        try:
            result = self.client.embeddings(model=self.embedding_model, prompt=text)
            emb = result.get("embedding")
            if isinstance(emb, list):
                return emb
        except Exception:
            return None
        return None

    def _heuristic_emotion(self, text: str) -> tuple[str, float, float]:
        t = text.lower()
        if any(w in t for w in ["嬉しい", "最高", "楽しい", "happy", "やった"]):
            return "joy", 0.7, 0.8
        if any(w in t for w in ["悲しい", "つらい", "sad", "落ち込"]):
            return "sadness", 0.4, -0.7
        if any(w in t for w in ["怒", "イライラ", "angry", "むかつ"]):
            return "anger", 0.8, -0.7
        if any(w in t for w in ["不安", "怖", "anxious", "fear"]):
            return "fear", 0.7, -0.5
        if any(w in t for w in ["びっくり", "驚", "surprise"]):
            return "surprise", 0.8, 0.1
        return "neutral", 0.3, 0.0

    def detect_emotion(self, text: str, use_llm: bool = True) -> tuple[str, float, float]:
        if not use_llm:
            return self._heuristic_emotion(text)
        prompt = (
            "ユーザー発話の感情をJSONで返してください。"
            '形式: {"emotion":"joy|sadness|anger|fear|surprise|neutral","arousal":0-1,"valence":-1-1}'
        )
        try:
            res = self.client.chat(
                model=self.chat_model,
                options={"temperature": 0.0},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                format="json",
            )
            data = _safe_json_loads(res["message"]["content"])
            emotion = str(data.get("emotion", "neutral"))
            arousal = float(data.get("arousal", 0.3))
            valence = float(data.get("valence", 0.0))
            return emotion, max(0.0, min(1.0, arousal)), max(-1.0, min(1.0, valence))
        except Exception:
            return self._heuristic_emotion(text)

    def _heuristic_extract_memories(self, user_text: str) -> list[MemoryRecord]:
        text = user_text.strip()
        if not text:
            return []
        records: list[MemoryRecord] = []
        emotion, arousal, valence = self._heuristic_emotion(text)

        # Preserve lightweight episodic memory of the latest user event/request.
        records.append(
            MemoryRecord(
                memory_type="episodic",
                content=f"User recently said: {text[:180]}",
                emotion=emotion,
                arousal=arousal,
                valence=valence,
                importance=0.45,
                ttl_days=30,
                tags=["recent"],
            )
        )

        # Detect stable user preference/profile hints.
        markers = ["好き", "嫌い", "仕事", "住ん", "趣味", "目標", "欲しい", "苦手"]
        if any(m in text for m in markers):
            records.append(
                MemoryRecord(
                    memory_type="profile",
                    content=f"Possible user profile hint: {text[:180]}",
                    emotion=emotion,
                    arousal=arousal,
                    valence=valence,
                    importance=0.65,
                    ttl_days=None,
                    tags=["profile_hint"],
                )
            )
        return records[:3]

    def extract_memories(
        self,
        user_text: str,
        assistant_text: str,
        use_llm: bool = True,
    ) -> list[MemoryRecord]:
        if not use_llm:
            return self._heuristic_extract_memories(user_text)
        instruction = """
あなたは会話から長期記憶候補を抽出する。
出力はJSONのみ。形式:
{
  "memories":[
    {
      "memory_type":"profile|episodic|affective_state",
      "content":"短い事実文",
      "emotion":"joy|sadness|anger|fear|surprise|neutral",
      "arousal":0.0,
      "valence":0.0,
      "importance":0.0,
      "ttl_days":null,
      "tags":[]
    }
  ]
}
ルール:
- プライバシー性の高い詳細住所や電話番号は保存しない。
- 最大3件。
- あいまいな内容は保存しない。
"""
        try:
            res = self.client.chat(
                model=self.chat_model,
                options={"temperature": 0.0},
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"User: {user_text}\nAssistant: {assistant_text}"},
                ],
                format="json",
            )
            data = _safe_json_loads(res["message"]["content"])
            out: list[MemoryRecord] = []
            for item in data.get("memories", [])[:3]:
                mtype = str(item.get("memory_type", "episodic"))
                if mtype not in {"profile", "episodic", "affective_state"}:
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                out.append(
                    MemoryRecord(
                        memory_type=mtype,
                        content=content,
                        emotion=str(item.get("emotion", "neutral")),
                        arousal=max(0.0, min(1.0, float(item.get("arousal", 0.3)))),
                        valence=max(-1.0, min(1.0, float(item.get("valence", 0.0)))),
                        importance=max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                        ttl_days=item.get("ttl_days"),
                        tags=item.get("tags", []),
                    )
                )
            return out
        except Exception:
            return []

    def build_memory_context(self, recalled: list[sqlite3.Row]) -> str:
        if not recalled:
            return "No relevant long-term memory."
        lines = []
        for row in recalled:
            lines.append(
                f"- [id:{row['id']}] ({row['memory_type']}, emotion={row['emotion']}, importance={row['importance']:.2f}) {row['content']}"
            )
        return "\n".join(lines)
