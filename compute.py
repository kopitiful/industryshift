#!/usr/bin/env python3
"""Weekly sector rotation scores for US and Europe."""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

US_SECTORS = {
    "XLK": "Technology",
    "XLC": "Communication",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLY": "Consumer Discret.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}
US_BENCHMARK = "SPY"

EU_SECTORS = {
    "EXV3.DE": "Technology",
    "EXV4.DE": "Banks",
    "EXH7.DE": "Health Care",
    "EXV6.DE": "Energy",
    "EXV1.DE": "Automobiles",
    "EXI3.DE": "Basic Resources",
    "EXV5.DE": "Telecom",
    "EXV2.DE": "Insurance",
    "EXH1.DE": "Utilities",
    "EXH8.DE": "Food & Beverage",
    "EXH4.DE": "Industrials",
}
EU_BENCHMARK = "EXW1.DE"

OUT_PATH = "docs/data/sectors.json"
HISTORY_MAX = 8  # weeks to retain


def _get_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker in df.columns:
        return df[ticker].dropna()
    return pd.Series(dtype=float)


def rs_score(s: pd.Series, b: pd.Series) -> float:
    common = s.index.intersection(b.index)
    if len(common) < 65:
        return 0.0
    s, b = s.loc[common], b.loc[common]
    r1m = (s.iloc[-1] / s.iloc[-21] - 1) - (b.iloc[-1] / b.iloc[-21] - 1)
    r3m = (s.iloc[-1] / s.iloc[-63] - 1) - (b.iloc[-1] / b.iloc[-63] - 1)
    return float(np.mean([r1m, r3m]) * 100)


def volume_score(vol: pd.Series) -> float:
    if len(vol) < 25:
        return 0.0
    recent = vol.iloc[-5:].sum()
    baseline = vol.iloc[-25:-5].sum() / 4
    return float((recent / baseline - 1) * 100) if baseline > 0 else 0.0


def technical_score(s: pd.Series) -> float:
    if len(s) < 50:
        return 0.0
    price = s.iloc[-1]
    score = 0.0
    if price > s.iloc[-50:].mean():
        score += 33
    if len(s) >= 200 and price > s.iloc[-200:].mean():
        score += 33
    high52w = s.iloc[-252:].max() if len(s) >= 252 else s.max()
    score += (price / high52w) * 34
    return score


def normalize(values: list) -> list:
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [5.0] * len(values)
    return ((arr - mn) / (mx - mn) * 10).round(2).tolist()


def load_history() -> list:
    if not os.path.exists(OUT_PATH):
        return []
    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("history", [])
    except Exception:
        return []


def prev_scores(history: list, weeks_ago: int, region: str) -> dict:
    """Returns {ticker: score} from N weeks ago, or {} if not enough history."""
    if len(history) < weeks_ago:
        return {}
    return history[-weeks_ago][region]


def compute(sectors: dict, benchmark: str, region: str, history: list) -> list:
    p1 = prev_scores(history, 1, region)
    p2 = prev_scores(history, 2, region)
    p4 = prev_scores(history, 4, region)

    tickers = list(sectors.keys()) + [benchmark]
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        vol = raw["Volume"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})
        vol = pd.DataFrame({tickers[0]: raw["Volume"]})

    bench_close = _get_series(close, benchmark)

    rs_raw, vol_raw, tech_raw = [], [], []
    for ticker in sectors:
        s = _get_series(close, ticker)
        v = _get_series(vol, ticker)
        rs_raw.append(rs_score(s, bench_close))
        vol_raw.append(volume_score(v))
        tech_raw.append(technical_score(s))

    rs_n = normalize(rs_raw)
    vol_n = normalize(vol_raw)
    tech_n = normalize(tech_raw)

    results = []
    for i, (ticker, name) in enumerate(sectors.items()):
        combined = round(0.4 * rs_n[i] + 0.3 * vol_n[i] + 0.3 * tech_n[i], 2)
        sc = combined

        d1 = round(sc - p1[ticker], 2) if ticker in p1 else None
        d2 = round(sc - p2[ticker], 2) if ticker in p2 else None
        d4 = round(sc - p4[ticker], 2) if ticker in p4 else None

        results.append({
            "ticker": ticker,
            "name": name,
            "score": sc,
            "rs": rs_n[i],
            "volume": vol_n[i],
            "technical": tech_n[i],
            "rs_pct": round(rs_raw[i], 2),
            "delta_1w": d1,
            "delta_2w": d2,
            "delta_4w": d4,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def main():
    history = load_history()

    print("Computing US sectors...")
    us = compute(US_SECTORS, US_BENCHMARK, "us", history)

    print("Computing EU sectors...")
    eu = compute(EU_SECTORS, EU_BENCHMARK, "eu", history)

    # Append snapshot to history AFTER computing (so deltas use old data)
    snapshot = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "us": {s["ticker"]: s["score"] for s in us},
        "eu": {s["ticker"]: s["score"] for s in eu},
    }
    history.append(snapshot)
    history = history[-HISTORY_MAX:]

    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "us": us,
        "eu": eu,
        "history": history,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTop 3 US:  " + " | ".join(f"{s['name']} {s['score']}" for s in us[:3]))
    print(f"Top 3 EU:  " + " | ".join(f"{s['name']} {s['score']}" for s in eu[:3]))
    print(f"History:   {len(history)} snapshots stored")
    print("\nDone → docs/data/sectors.json")


if __name__ == "__main__":
    main()
