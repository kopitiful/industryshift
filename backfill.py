#!/usr/bin/env python3
"""Backfill 4 weeks of historical snapshots so trend deltas work immediately.
Run once after initial setup: python3 backfill.py && python3 compute.py
"""

import json
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from compute import (
    EU_BENCHMARK, EU_SECTORS, OUT_PATH, US_BENCHMARK, US_SECTORS,
    _get_series, normalize, rs_score, technical_score, volume_score,
)


def scores_as_of(sectors: dict, benchmark: str, close: pd.DataFrame, vol: pd.DataFrame, cutoff: str) -> dict:
    ts = pd.Timestamp(cutoff)
    c = close[close.index <= ts]
    v = vol[vol.index <= ts]
    if len(c) < 65:
        return {}
    bench = _get_series(c, benchmark)
    rs_raw, vol_raw, tech_raw = [], [], []
    for ticker in sectors:
        rs_raw.append(rs_score(_get_series(c, ticker), bench))
        vol_raw.append(volume_score(_get_series(v, ticker)))
        tech_raw.append(technical_score(_get_series(c, ticker)))
    rs_n, vol_n, tech_n = normalize(rs_raw), normalize(vol_raw), normalize(tech_raw)
    return {
        ticker: {
            "s": round(0.35 * rs_n[i] + 0.50 * vol_n[i] + 0.15 * tech_n[i], 2),
            "r": rs_n[i],
            "v": vol_n[i],
            "t": tech_n[i],
        }
        for i, ticker in enumerate(sectors)
    }


def past_fridays(n: int) -> list[str]:
    today = date.today()
    days_back = (today.weekday() - 4) % 7 or 7
    last_friday = today - timedelta(days=days_back)
    return [(last_friday - timedelta(weeks=i)).isoformat() for i in range(n - 1, -1, -1)]


def main():
    fridays = past_fridays(4)
    print(f"Backfilling: {', '.join(fridays)}")

    with open(OUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    existing_dates = {h["date"] for h in data.get("history", [])}

    all_tickers = list(US_SECTORS) + [US_BENCHMARK] + list(EU_SECTORS) + [EU_BENCHMARK]
    print("Downloading price data (1y)…")
    raw = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({all_tickers[0]: raw["Close"]})
    vol   = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame({all_tickers[0]: raw["Volume"]})

    new_snaps = []
    for friday in fridays:
        print(f"  {friday}: computing…")
        us = scores_as_of(US_SECTORS, US_BENCHMARK, close, vol, friday)
        eu = scores_as_of(EU_SECTORS, EU_BENCHMARK, close, vol, friday)
        if us and eu:
            new_snaps.append({"date": friday, "us": us, "eu": eu})

    # Merge: new_snaps override old entries for same date
    by_date = {h["date"]: h for h in data.get("history", [])}
    for s in new_snaps:
        by_date[s["date"]] = s
    combined = sorted(by_date.values(), key=lambda x: x["date"])[-8:]
    data["history"] = combined

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nHistory: {[h['date'] for h in combined]}")
    print("Done. Now run: python3 compute.py")


if __name__ == "__main__":
    main()
