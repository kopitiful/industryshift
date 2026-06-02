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

TECH_SUBSECTORS = {
    "SMH":  "Semiconductors",
    "IGV":  "Software",
    "WCLD": "Cloud",
    "CIBR": "Cybersecurity",
}

OUT_PATH = "docs/data/sectors.json"
HISTORY_MAX = 8  # weeks to retain


def _get_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker in df.columns:
        return df[ticker].dropna()
    return pd.Series(dtype=float)


def price_return_1w(s: pd.Series) -> float:
    """Actual 5-day price return of the ETF in %."""
    if len(s) < 6:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-6] - 1) * 100)


def rs_score(s: pd.Series, b: pd.Series) -> float:
    """Short-term excess return: 1W + 3W vs benchmark."""
    common = s.index.intersection(b.index)
    if len(common) < 16:
        return 0.0
    s, b = s.loc[common], b.loc[common]
    r1w = (s.iloc[-1] / s.iloc[-6]  - 1) - (b.iloc[-1] / b.iloc[-6]  - 1)
    r3w = (s.iloc[-1] / s.iloc[-16] - 1) - (b.iloc[-1] / b.iloc[-16] - 1)
    return float(np.mean([r1w, r3w]) * 100)


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


def prev_snap(history: list, weeks_ago: int, region: str, today: str) -> dict:
    """Returns {ticker: {s, r, v, t}} from N weeks ago, excluding today."""
    past = [h for h in history if h["date"] != today]
    if len(past) < weeks_ago:
        return {}
    snap = past[-weeks_ago][region]
    # Support old format {ticker: score} and new format {ticker: {s,r,v,t}}
    result = {}
    for ticker, val in snap.items():
        if isinstance(val, dict):
            result[ticker] = val
        else:
            result[ticker] = {"s": val, "r": None, "v": None, "t": None}
    return result


def compute(sectors: dict, benchmark: str, region: str, history: list, today: str) -> list:
    p1 = prev_snap(history, 1, region, today)
    p2 = prev_snap(history, 2, region, today)
    p4 = prev_snap(history, 4, region, today)

    tickers = list(sectors.keys()) + [benchmark]
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        vol = raw["Volume"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})
        vol = pd.DataFrame({tickers[0]: raw["Volume"]})

    bench_close = _get_series(close, benchmark)

    rs_raw, vol_raw, tech_raw, price1w_raw = [], [], [], []
    for ticker in sectors:
        s = _get_series(close, ticker)
        v = _get_series(vol, ticker)
        rs_raw.append(rs_score(s, bench_close))
        vol_raw.append(volume_score(v))
        tech_raw.append(technical_score(s))
        price1w_raw.append(price_return_1w(s))

    rs_n = normalize(rs_raw)
    vol_n = normalize(vol_raw)
    tech_n = normalize(tech_raw)

    results = []
    for i, (ticker, name) in enumerate(sectors.items()):
        combined = round(0.35 * rs_n[i] + 0.50 * vol_n[i] + 0.15 * tech_n[i], 2)
        sc = combined

        def _d(prev: dict, key: str, cur: float):
            v = prev.get(ticker, {}).get(key)
            return round(cur - v, 2) if v is not None else None

        d1 = _d(p1, "s", sc)
        d2 = _d(p2, "s", sc)
        d4 = _d(p4, "s", sc)

        results.append({
            "ticker": ticker,
            "name": name,
            "score": sc,
            "rs": rs_n[i],
            "volume": vol_n[i],
            "technical": tech_n[i],
            "rs_pct": round(rs_raw[i], 2),
            "price_1w": round(price1w_raw[i], 2),
            "delta_1w": d1,
            "delta_2w": d2,
            "delta_4w": d4,
            "delta_rs":   _d(p1, "r", rs_n[i]),
            "delta_vol":  _d(p1, "v", vol_n[i]),
            "delta_tech": _d(p1, "t", tech_n[i]),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def compute_subsectors(sectors: dict, benchmark: str) -> list:
    """Compute scores for sub-sectors, normalized within their own group."""
    tickers = list(sectors.keys()) + [benchmark]
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close, vol = raw["Close"], raw["Volume"]
    else:
        close = pd.DataFrame({tickers[0]: raw["Close"]})
        vol   = pd.DataFrame({tickers[0]: raw["Volume"]})

    bench_close = _get_series(close, benchmark)
    rs_raw, vol_raw, tech_raw, price1w_raw = [], [], [], []
    for ticker in sectors:
        s = _get_series(close, ticker)
        v = _get_series(vol, ticker)
        rs_raw.append(rs_score(s, bench_close))
        vol_raw.append(volume_score(v))
        tech_raw.append(technical_score(s))
        price1w_raw.append(price_return_1w(s))

    rs_n, vol_n, tech_n = normalize(rs_raw), normalize(vol_raw), normalize(tech_raw)
    results = []
    for i, (ticker, name) in enumerate(sectors.items()):
        sc = round(0.35 * rs_n[i] + 0.50 * vol_n[i] + 0.15 * tech_n[i], 2)
        results.append({
            "ticker": ticker,
            "name": name,
            "score": sc,
            "rs": rs_n[i],
            "volume": vol_n[i],
            "technical": tech_n[i],
            "rs_pct": round(rs_raw[i], 2),
            "price_1w": round(price1w_raw[i], 2),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def main():
    history = load_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("Computing US sectors...")
    us = compute(US_SECTORS, US_BENCHMARK, "us", history, today)

    print("Computing EU sectors...")
    eu = compute(EU_SECTORS, EU_BENCHMARK, "eu", history, today)

    print("Computing Tech sub-sectors...")
    tech_sub = compute_subsectors(TECH_SUBSECTORS, US_BENCHMARK)

    # Replace today's snapshot if it exists, then append
    def _snap(sectors):
        return {s["ticker"]: {"s": s["score"], "r": s["rs"], "v": s["volume"], "t": s["technical"]} for s in sectors}

    snapshot = {
        "date": today,
        "us": _snap(us),
        "eu": _snap(eu),
    }
    history = [h for h in history if h["date"] != today]
    history.append(snapshot)
    history = history[-HISTORY_MAX:]

    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "us": us,
        "eu": eu,
        "tech_sub": tech_sub,
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
