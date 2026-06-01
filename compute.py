#!/usr/bin/env python3
"""Wöchentliche Sektor-Rotation-Scores für US und Europa."""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

US_SECTORS = {
    "XLK": "Technologie",
    "XLC": "Kommunikation",
    "XLF": "Finanzen",
    "XLE": "Energie",
    "XLV": "Gesundheit",
    "XLI": "Industrie",
    "XLB": "Rohstoffe",
    "XLY": "Konsum zyklisch",
    "XLP": "Konsum defensiv",
    "XLU": "Versorger",
    "XLRE": "Immobilien",
}
US_BENCHMARK = "SPY"

EU_SECTORS = {
    "EXV3.DE": "Technologie",
    "EXV4.DE": "Banken",
    "EXH7.DE": "Gesundheit",
    "EXV6.DE": "Energie",
    "EXV1.DE": "Automobil",
    "EXI3.DE": "Rohstoffe",
    "EXV5.DE": "Telekommunikation",
    "EXV2.DE": "Versicherung",
    "EXH1.DE": "Versorger",
    "EXH8.DE": "Lebensmittel",
    "EXH4.DE": "Industrie",
}
EU_BENCHMARK = "EXW1.DE"


def _get_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Gibt eine Preisserie aus einem DataFrame zurück, toleriert fehlende Ticker."""
    if ticker in df.columns:
        return df[ticker].dropna()
    return pd.Series(dtype=float)


def rs_score(s: pd.Series, b: pd.Series) -> float:
    """Überschussrendite des Sektors vs. Benchmark (1M + 3M Durchschnitt)."""
    common = s.index.intersection(b.index)
    if len(common) < 65:
        return 0.0
    s, b = s.loc[common], b.loc[common]
    r1m = (s.iloc[-1] / s.iloc[-21] - 1) - (b.iloc[-1] / b.iloc[-21] - 1)
    r3m = (s.iloc[-1] / s.iloc[-63] - 1) - (b.iloc[-1] / b.iloc[-63] - 1)
    return float(np.mean([r1m, r3m]) * 100)


def volume_score(vol: pd.Series) -> float:
    """Aktuelle Woche vs. 4-Wochen-Durchschnitt (in %)."""
    if len(vol) < 25:
        return 0.0
    recent = vol.iloc[-5:].sum()
    baseline = vol.iloc[-25:-5].sum() / 4
    return float((recent / baseline - 1) * 100) if baseline > 0 else 0.0


def technical_score(s: pd.Series) -> float:
    """0–100: MA50, MA200, Abstand zum 52W-Hoch."""
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
    """Skaliert eine Werteliste auf 0–10."""
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [5.0] * len(values)
    return ((arr - mn) / (mx - mn) * 10).round(2).tolist()


def compute(sectors: dict, benchmark: str) -> list:
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
        results.append({
            "ticker": ticker,
            "name": name,
            "score": combined,
            "rs": rs_n[i],
            "volume": vol_n[i],
            "technical": tech_n[i],
            "rs_pct": round(rs_raw[i], 2),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def main():
    print("Berechne US-Sektoren...")
    us = compute(US_SECTORS, US_BENCHMARK)

    print("Berechne EU-Sektoren...")
    eu = compute(EU_SECTORS, EU_BENCHMARK)

    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "us": us,
        "eu": eu,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/sectors.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTop 3 USA:    " + " | ".join(f"{s['name']} {s['score']}" for s in us[:3]))
    print(f"Top 3 Europa: " + " | ".join(f"{s['name']} {s['score']}" for s in eu[:3]))
    print("\nFertig → docs/data/sectors.json")


if __name__ == "__main__":
    main()
