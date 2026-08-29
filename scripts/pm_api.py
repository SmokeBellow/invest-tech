"""Тонкий клиент для публичных API Polymarket (без ключей/аутентификации).

Gamma API (https://gamma-api.polymarket.com) — метаданные рынков: вопрос,
outcomes/outcomePrices, объём, теги, статус closed/active, clobTokenIds.

CLOB API (https://clob.polymarket.com) — история цены отдельного токена
(outcome) рынка: /prices-history.

Оба публичные, без API-ключа. Используются во всех pm_*.py скриптах, чтобы
не дублировать логику запросов/пагинации/ретраев.
"""
from __future__ import annotations

import time
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Категории (Gamma tag_id), в которых ищем ставки на редкие "хвостовые"
# события — подобраны вручную по смыслу (гео-события, катастрофы,
# макро-шоки), не по результату скана. Список фиксирован здесь, как
# LIVE_INSTRUMENTS для системы A/B — не додумывать/расширять молча.
CATEGORY_TAGS = {
    "geopolitics": 100265,
    "world": 101970,
    "weather": 84,
    "climate": 87,
    "natural_disaster": 101998,
    "earthquake": 100184,
    "recession": 100201,
    "economy": 100328,
    "war": 79,
    "nuclear": 1289,
}

_session = requests.Session()


def _get(url: str, params: dict, retries: int = 3, timeout: int = 20) -> object:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_exc}")


def fetch_markets(tag_id: int, closed: bool, limit: int = 20, offset: int = 0,
                   order: str = "volume", ascending: bool = False) -> list:
    """Одна страница /markets для заданного тега."""
    params = {
        "tag_id": tag_id,
        "closed": str(closed).lower(),
        "active": "true" if not closed else "false",
        "limit": limit,
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    return _get(f"{GAMMA_BASE}/markets", params)


def fetch_markets_paginated(tag_id: int, closed: bool, max_markets: int,
                             page_size: int = 100) -> list:
    """Собирает до max_markets рынков по тегу, постранично (order=volume desc)."""
    out = []
    offset = 0
    while len(out) < max_markets:
        page = fetch_markets(tag_id, closed, limit=page_size, offset=offset)
        if not page:
            break
        out.extend(page)
        offset += page_size
        if len(page) < page_size:
            break
    return out[:max_markets]


def fetch_price_history(token_id: str, interval: str = "max",
                         fidelity: int = 1440) -> list:
    """Дневная история цены (close-подобная точка p за интервал t) для
    одного outcome-токена. fidelity=1440 минут = 1 точка в сутки."""
    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    data = _get(f"{CLOB_BASE}/prices-history", params)
    return data.get("history", [])
