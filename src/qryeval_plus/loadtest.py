"""Small dependency-light HTTP load generator for portfolio evidence."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from pathlib import Path
from typing import List

from qryeval_plus.statistics import percentile


async def run_load_test(
    *,
    url: str,
    questions: List[str],
    concurrency: int,
    requests: int,
    policy: str = "adaptive_rewrite",
    token: str | None = None,
    cache_bust: bool = False,
    cache_bust_label: str = "run",
):
    import httpx

    if concurrency <= 0 or requests <= 0 or not questions:
        raise ValueError("Load test requires positive concurrency/requests and questions.")
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"Authorization": "Bearer " + token} if token else {}
    latencies = []
    statuses = Counter()
    cache_hits = 0
    tokens = 0
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=240.0, headers=headers) as client:
            async def one(index):
                nonlocal cache_hits, tokens
                async with semaphore:
                    begin = time.monotonic()
                    response = await client.post(
                        url.rstrip("/") + "/v1/answer",
                        json={
                            "question": questions[index % len(questions)] + (
                                " [load-test-{}-{}]".format(cache_bust_label, index) if cache_bust else ""
                            ),
                            "policy": policy,
                        },
                    )
                    latencies.append(time.monotonic() - begin)
                    statuses[response.status_code] += 1
                    if response.status_code == 200:
                        payload = response.json()
                        cache_hits += int(bool(payload.get("cache_hit")))
                        tokens += int(payload.get("usage", {}).get("total_tokens", 0) or 0)

            await asyncio.gather(*(one(index) for index in range(requests)))
    except httpx.HTTPError as exc:
        raise RuntimeError("Load test could not reach {}: {}".format(url, exc)) from exc
    duration = time.monotonic() - started
    return {
        "url": url,
        "policy": policy,
        "requests": requests,
        "concurrency": concurrency,
        "cache_bust": cache_bust,
        "cache_bust_label": cache_bust_label if cache_bust else None,
        "duration_seconds": duration,
        "throughput_qps": requests / duration,
        "status_counts": dict(sorted(statuses.items())),
        "success_rate": statuses.get(200, 0) / requests,
        "rate_limited_rate": statuses.get(429, 0) / requests,
        "cache_hit_rate": cache_hits / max(1, statuses.get(200, 0)),
        "tokens_per_success": tokens / max(1, statuses.get(200, 0)),
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
    }


def read_questions(path: str | Path) -> List[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if ":" in line
    ]
