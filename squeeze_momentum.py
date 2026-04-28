"""
Squeeze Momentum Module
------------------------
Implementasi Python yang PERSIS mengikuti Pine Script
"Squeeze Momentum Indicator [LazyBear]" dari TradingView.

Algoritma:
1. Bollinger Bands (BB): SMA ± mult * stdev
2. Keltner Channels (KC): SMA ± multKC * SMA(TR)
3. Squeeze: BB inside KC (sqzOn) atau BB outside KC (sqzOff)
4. Momentum value: linear regression dari (close - midline)
5. Klasifikasi: Rise/Fall + strong/weak + white (squeeze ON)

Output categories (8 total):
  Rise strong white, Rise weak white, Rise strong, Rise weak,
  Fall strong white, Fall weak white, Fall strong, Fall weak
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ============================
# Helper: Linear Regression Value (sama dengan linreg() Pine Script)
# ============================
def _linreg(series: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """
    Linear regression value — sama dengan linreg() di Pine Script.
    Menghitung predicted value dari linear regression pada window terakhir.
    """
    n = len(series)
    result = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = series[i - length + 1: i + 1]
        if np.any(np.isnan(window)):
            continue

        x = np.arange(length, dtype=float)
        y = window

        # Linear regression: y = a + b*x
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx == 0:
            result[i - offset] = y_mean
            continue

        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        b = ss_xy / ss_xx
        a = y_mean - b * x_mean

        # Predicted value at the end of the window (offset dari akhir)
        result[i - offset] = a + b * (length - 1 - offset)

    return result


# ============================
# Squeeze Momentum calculation — PERSIS Pine Script LazyBear
# ============================
def compute_squeeze_momentum(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
) -> pd.DataFrame:
    """
    Compute Squeeze Momentum Indicator.

    Parameters match LazyBear's Pine Script defaults:
    - BB Length: 20, BB MultFactor: 2.0
    - KC Length: 20, KC MultFactor: 1.5
    - Use TrueRange: True

    Returns DataFrame with columns:
    - squeeze_val: momentum histogram value
    - squeeze_on: True jika squeeze ON (BB inside KC)
    - squeeze_off: True jika squeeze OFF (BB outside KC)
    - bar_color: 'lime', 'green', 'red', atau 'maroon'
    - squeeze_label: klasifikasi lengkap (Rise strong white, dll)
    """
    df = df.copy()

    # Flatten multi-level columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    n = len(close)

    # ============================
    # Bollinger Bands
    # ============================
    # basis = SMA(close, length)
    # dev = mult * stdev(close, length)
    # Note: Pine Script stdev uses population stdev (ddof=0) by default
    basis = np.full(n, np.nan)
    bb_dev = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        window = close[i - bb_length + 1: i + 1]
        basis[i] = np.mean(window)
        bb_dev[i] = bb_mult * np.std(window, ddof=0)

    upper_bb = basis + bb_dev
    lower_bb = basis - bb_dev

    # ============================
    # Keltner Channels
    # ============================
    # ma = SMA(close, KC_length)
    # range = useTrueRange ? tr : (high - low)
    # rangema = SMA(range, KC_length)
    kc_ma = np.full(n, np.nan)
    for i in range(kc_length - 1, n):
        kc_ma[i] = np.mean(close[i - kc_length + 1: i + 1])

    # True Range
    if use_true_range:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
    else:
        tr = high - low

    # rangema = SMA(TR, KC_length)
    rangema = np.full(n, np.nan)
    for i in range(kc_length - 1, n):
        rangema[i] = np.mean(tr[i - kc_length + 1: i + 1])

    upper_kc = kc_ma + rangema * kc_mult
    lower_kc = kc_ma - rangema * kc_mult

    # ============================
    # Squeeze detection
    # ============================
    sqz_on = np.zeros(n, dtype=bool)
    sqz_off = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(lower_bb[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_kc[i]) or np.isnan(upper_kc[i]):
            continue
        sqz_on[i] = (lower_bb[i] > lower_kc[i]) and (upper_bb[i] < upper_kc[i])
        sqz_off[i] = (lower_bb[i] < lower_kc[i]) and (upper_bb[i] > upper_kc[i])

    # ============================
    # Momentum value (val)
    # val = linreg(close - avg(avg(highest(high, KC_len), lowest(low, KC_len)), sma(close, KC_len)), KC_len, 0)
    # ============================

    # highest(high, kc_length)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(kc_length - 1, n):
        highest_high[i] = np.max(high[i - kc_length + 1: i + 1])
        lowest_low[i] = np.min(low[i - kc_length + 1: i + 1])

    # midline = avg(avg(highest_high, lowest_low), sma(close, kc_length))
    # avg() in Pine = (a + b) / 2
    hl_avg = (highest_high + lowest_low) / 2.0
    midline = (hl_avg + kc_ma) / 2.0

    # source_for_linreg = close - midline
    source_linreg = close - midline

    # val = linreg(source_linreg, kc_length, 0)
    val = _linreg(source_linreg, kc_length, offset=0)

    # ============================
    # Classification
    # ============================
    bar_colors = np.full(n, "", dtype=object)
    labels = np.full(n, "", dtype=object)

    for i in range(1, n):
        if np.isnan(val[i]):
            continue

        prev_val = val[i - 1] if not np.isnan(val[i - 1]) else 0.0

        # Bar color (Pine Script logic)
        if val[i] > 0:
            if val[i] > prev_val:
                bar_colors[i] = "lime"      # bright green = strong rise
                direction = "Rise strong"
            else:
                bar_colors[i] = "green"      # dark green = weak rise
                direction = "Rise weak"
        else:
            if val[i] < prev_val:
                bar_colors[i] = "red"        # bright red = strong fall
                direction = "Fall strong"
            else:
                bar_colors[i] = "maroon"     # dark red = weak fall
                direction = "Fall weak"

        # Add "white" suffix if squeeze is ON
        if sqz_on[i]:
            labels[i] = f"{direction} white"
        else:
            labels[i] = direction

    # Build result
    df["squeeze_val"] = val
    df["squeeze_on"] = sqz_on
    df["squeeze_off"] = sqz_off
    df["bar_color"] = bar_colors
    df["squeeze_label"] = labels

    return df


def get_squeeze_label_at_bar(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
) -> str:
    """
    Convenience: compute Squeeze Momentum dan return label bar terakhir.
    Contoh output: "Rise strong white", "Fall weak", dll.
    """
    result = compute_squeeze_momentum(
        df,
        bb_length=bb_length,
        bb_mult=bb_mult,
        kc_length=kc_length,
        kc_mult=kc_mult,
        use_true_range=use_true_range,
    )

    if result.empty:
        return "Rise weak"

    label = result["squeeze_label"].iloc[-1]
    return label if label else "Rise weak"
