from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


class Store:
    def __init__(self, path: str) -> None:
        self.path = path

    async def initialize(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS request_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    is_stream INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS request_events_created_at
                    ON request_events(created_at);
                CREATE TABLE IF NOT EXISTS verdict_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ladder_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    mutation TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    rule TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await db.commit()

    async def record(
        self,
        cache_key: str,
        provider: str,
        model: str,
        verdict: str,
        score: float,
        reason: str,
        status_code: int,
        is_stream: bool,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """INSERT INTO request_events
                   (cache_key, provider, model, verdict, score, reason, status_code, is_stream, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cache_key, provider, model, verdict, score, reason, status_code, int(is_stream), created_at),
            )
            await db.execute(
                """INSERT INTO verdict_cache (cache_key, provider, model, verdict, score, reason, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     verdict=excluded.verdict,
                     score=excluded.score,
                     reason=excluded.reason,
                     updated_at=excluded.updated_at""",
                (cache_key, provider, model, verdict, score, reason, created_at),
            )
            await db.commit()

    async def cached_verdict(self, cache_key: str) -> tuple[str, float, str] | None:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT verdict, score, reason FROM verdict_cache WHERE cache_key = ?",
                (cache_key,),
            ) as cursor:
                row = await cursor.fetchone()
        return (str(row[0]), float(row[1]), str(row[2])) if row else None

    async def record_ladder(self, attempts, provider_switches: int, budget_stops: list[str]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            ("attempt", item.step, item.mutation, item.provider, item.rule, item.verdict, created_at)
            for item in attempts
        ]
        rows.extend(
            ("provider_switch", 7, "provider_switch", "", "route_next_provider", "switched", created_at)
            for _ in range(provider_switches)
        )
        rows.extend(
            ("budget_stop", 0, "", "", reason, "stopped", created_at)
            for reason in budget_stops
        )
        if not rows:
            return
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                """INSERT INTO ladder_events
                   (event_type, step, mutation, provider, rule, verdict, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

    async def refusals(self, limit: int = 100) -> list[dict[str, object]]:
        limit = min(max(int(limit), 1), 500)
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                """SELECT provider, model, verdict, score, reason, status_code, is_stream, created_at
                   FROM request_events WHERE verdict = 'refusal'
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            {"provider": row[0], "model": row[1], "verdict": row[2], "score": row[3],
             "rule": row[4], "status_code": row[5], "stream": bool(row[6]), "created_at": row[7]}
            for row in rows
        ]

    async def stats(self, include_ladder: bool = True) -> dict[str, int | float | dict[str, int]]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN verdict = 'refusal' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN verdict = 'uncertain' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN verdict = 'unavailable' THEN 1 ELSE 0 END),
                          SUM(is_stream)
                   FROM request_events"""
            ) as cursor:
                row = await cursor.fetchone()
            if include_ladder:
                async with db.execute(
                    """SELECT event_type, mutation, COUNT(*) FROM ladder_events
                       GROUP BY event_type, mutation"""
                ) as cursor:
                    ladder_rows = await cursor.fetchall()
        total = int(row[0] or 0)
        refusals = int(row[1] or 0)
        result: dict[str, int | float | dict[str, int]] = {
            "requests": total,
            "refusals": refusals,
            "uncertain": int(row[2] or 0),
            "upstream_unavailable": int(row[3] or 0),
            "streamed": int(row[4] or 0),
            "refusal_rate": refusals / total if total else 0.0,
        }
        if include_ladder:
            counts = {(str(kind), str(mutation)): int(count) for kind, mutation, count in ladder_rows}
            result.update({
                "ladder_attempts": sum(count for (kind, _), count in counts.items() if kind == "attempt"),
                "mutations_fired": {
                    mutation: count for (kind, mutation), count in counts.items()
                    if kind == "attempt" and mutation
                },
                "provider_switches": sum(count for (kind, _), count in counts.items() if kind == "provider_switch"),
                "budget_stops": sum(count for (kind, _), count in counts.items() if kind == "budget_stop"),
            })
        return result
