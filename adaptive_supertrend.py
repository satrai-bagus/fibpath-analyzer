"""
Adaptive SuperTrend Module
--------------------------
Implementasi Python yang PERSIS mengikuti Pine Script
"Machine Learning Adaptive SuperTrend [AlgoAlpha]" dari TradingView.

Algoritma:
1. Hitung ATR (Wilder's RMA)
2. K-Means clustering pada trailing ATR window → 3 centroid (high/mid/low vol)
3. Klasifikasi ATR bar saat ini → ambil centroid terdekat (assigned_centroid)
4. SuperTrend dihitung dengan: factor * assigned_centroid (BUKAN factor * ATR)
5. Direction: -1 = bullish (Long), 1 = bearish (Short)

Output: "Long" atau "Short" per bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================
# ATR menggunakan Wilder's RMA (sama dengan ta.atr di Pine Script)
# ============================
def _rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Moving Average (RMA) — sama dengan Pine Script ta.rma()."""
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return result

    # Seed dengan SMA
    result[period - 1] = np.mean(values[:period])

    # RMA: rma = (prev_rma * (period - 1) + current) / period
    alpha = 1.0 / period
    for i in range(period, n):
        result[i] = result[i - 1] * (1 - alpha) + values[i] * alpha

    return result


def _compute_atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10) -> np.ndarray:
    """ATR menggunakan RMA (Wilder's smoothing) — sama dengan ta.atr() Pine Script."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    return _rma(tr, period)


# ============================
# K-Means clustering — PERSIS seperti Pine Script
# ============================
def _kmeans_pine(
    atr_window: np.ndarray,
    init_high: float,
    init_mid: float,
    init_low: float,
    max_iter: int = 1000,
) -> tuple:
    """
    K-Means 3 cluster PERSIS seperti Pine Script.

    Parameters:
        atr_window: array ATR values dari training period
        init_high/mid/low: initial centroid guesses (dari linear interpolasi)

    Returns:
        (centroid_high, centroid_mid, centroid_low, cluster_sizes)
    """
    c_a = init_high  # high vol centroid
    c_b = init_mid   # mid vol centroid
    c_c = init_low   # low vol centroid

    for _ in range(max_iter):
        # Assignment step
        hv_vals = []
        mv_vals = []
        lv_vals = []

        for val in atr_window:
            d_a = abs(val - c_a)
            d_b = abs(val - c_b)
            d_c = abs(val - c_c)

            # Pine Script: strict < comparisons (pertama yang paling kecil menang)
            if d_a < d_b and d_a < d_c:
                hv_vals.append(val)
            elif d_b < d_a and d_b < d_c:
                mv_vals.append(val)
            elif d_c < d_a and d_c < d_b:
                lv_vals.append(val)
            # Kalau ada tie (jarak sama), Pine Script tidak assign ke manapun
            # (karena semua kondisi pakai strict <)

        # Update step
        new_a = np.mean(hv_vals) if hv_vals else c_a
        new_b = np.mean(mv_vals) if mv_vals else c_b
        new_c = np.mean(lv_vals) if lv_vals else c_c

        # Convergence check
        if new_a == c_a and new_b == c_b and new_c == c_c:
            break

        c_a = new_a
        c_b = new_b
        c_c = new_c

    sizes = (len(hv_vals), len(mv_vals), len(lv_vals))
    return c_a, c_b, c_c, sizes


# ============================
# SuperTrend — PERSIS seperti pine_supertrend() di Pine Script
# ============================
def _pine_supertrend_step(
    hl2: float,
    close_val: float,
    factor: float,
    atr_val: float,
    prev_lower: float,
    prev_upper: float,
    prev_close: float,
    prev_st: float,
    prev_dir: int,
    is_first: bool,
) -> tuple:
    """
    Satu step SuperTrend PERSIS seperti Pine Script.

    Pine Script direction convention:
        -1 = bullish (Long, harga di atas ST)
         1 = bearish (Short, harga di bawah ST)

    Returns: (supertrend_val, direction, lower_band, upper_band)
    """
    upper_band = hl2 + factor * atr_val
    lower_band = hl2 - factor * atr_val

    # Band adjustment (Pine Script logic)
    if not is_first:
        if lower_band > prev_lower or prev_close < prev_lower:
            pass  # keep new lower_band
        else:
            lower_band = prev_lower

        if upper_band < prev_upper or prev_close > prev_upper:
            pass  # keep new upper_band
        else:
            upper_band = prev_upper

    # Direction logic (Pine Script)
    if is_first:
        direction = 1  # start bearish
    elif prev_st == prev_upper:
        # Was bearish (ST was at upper band)
        direction = -1 if close_val > upper_band else 1
    else:
        # Was bullish (ST was at lower band)
        direction = 1 if close_val < lower_band else -1

    # SuperTrend value
    if direction == -1:
        st = lower_band  # bullish → ST at lower band
    else:
        st = upper_band  # bearish → ST at upper band

    return st, direction, lower_band, upper_band


# ============================
# MAIN: Adaptive SuperTrend (matching Pine Script exactly)
# ============================
def compute_adaptive_supertrend(
    df: pd.DataFrame,
    atr_length: int = 10,
    factor: float = 3.0,
    training_period: int = 100,
    highvol_percentile: float = 0.75,
    midvol_percentile: float = 0.50,
    lowvol_percentile: float = 0.25,
) -> pd.DataFrame:
    """
    Compute Adaptive SuperTrend — PERSIS seperti Pine Script AlgoAlpha.

    Key difference dari SuperTrend biasa:
    - Factor tetap 3.0
    - ATR diganti dengan CENTROID dari K-Means cluster terdekat
    - SuperTrend bands: hl2 ± factor * centroid (bukan hl2 ± factor * atr)
    """
    df = df.copy()

    # Flatten multi-level columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    n = len(close)

    # Step 1: ATR via RMA (Wilder's)
    atr = _compute_atr_rma(high, low, close, period=atr_length)

    # Step 2: Per-bar K-Means + SuperTrend
    supertrend = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)  # 1 = bearish default
    cluster_labels = np.full(n, "", dtype=object)
    assigned_centroids = np.full(n, np.nan)

    prev_lower = np.nan
    prev_upper = np.nan
    prev_st = np.nan
    prev_dir = 1
    first_valid = True

    for i in range(n):
        # Skip kalau ATR belum valid
        if np.isnan(atr[i]):
            continue

        # K-Means clustering (hanya jika cukup data training)
        if i >= training_period - 1:
            # Training window: ATR values dari [i - training_period + 1] sampai [i]
            start_idx = i - training_period + 1
            atr_window = atr[start_idx:i + 1]
            valid_atr = atr_window[~np.isnan(atr_window)]

            if len(valid_atr) > 0:
                # Initial guesses: linear interpolation (PERSIS Pine Script)
                # upper = ta.highest(volatility, training_data_period)
                # lower = ta.lowest(volatility, training_data_period)
                # high_vol = lower + (upper - lower) * highvol_percentile
                upper_atr = np.max(valid_atr)
                lower_atr = np.min(valid_atr)
                rng = upper_atr - lower_atr

                init_high = lower_atr + rng * highvol_percentile
                init_mid = lower_atr + rng * midvol_percentile
                init_low = lower_atr + rng * lowvol_percentile

                # Run K-Means
                c_high, c_mid, c_low, sizes = _kmeans_pine(
                    valid_atr, init_high, init_mid, init_low
                )

                # Classify current ATR → nearest centroid
                current_atr = atr[i]
                d_a = abs(current_atr - c_high)
                d_b = abs(current_atr - c_mid)
                d_c = abs(current_atr - c_low)

                distances = [d_a, d_b, d_c]
                centroids = [c_high, c_mid, c_low]
                labels = ["High", "Medium", "Low"]

                min_idx = int(np.argmin(distances))
                centroid_val = centroids[min_idx]
                cluster_labels[i] = labels[min_idx]
                assigned_centroids[i] = centroid_val
            else:
                centroid_val = atr[i]
                assigned_centroids[i] = centroid_val
        else:
            # Sebelum training period cukup, pakai ATR biasa
            centroid_val = atr[i]
            assigned_centroids[i] = centroid_val

        # SuperTrend step
        hl2 = (high[i] + low[i]) / 2.0
        prev_close_val = close[i - 1] if i > 0 else close[i]

        st, dir_val, lb, ub = _pine_supertrend_step(
            hl2=hl2,
            close_val=close[i],
            factor=factor,
            atr_val=centroid_val,  # KEY: pakai centroid, bukan raw ATR
            prev_lower=prev_lower if not np.isnan(prev_lower) else lb if 'lb' in dir() else 0,
            prev_upper=prev_upper if not np.isnan(prev_upper) else ub if 'ub' in dir() else 0,
            prev_close=prev_close_val,
            prev_st=prev_st if not np.isnan(prev_st) else 0,
            prev_dir=prev_dir,
            is_first=first_valid,
        )

        supertrend[i] = st
        direction[i] = dir_val
        prev_lower = lb
        prev_upper = ub
        prev_st = st
        prev_dir = dir_val
        first_valid = False

    # Pine Script: dir == -1 → bullish (Long), dir == 1 → bearish (Short)
    df["atr"] = atr
    df["volatility_cluster"] = cluster_labels
    df["assigned_centroid"] = assigned_centroids
    df["supertrend"] = supertrend
    df["trend_direction"] = direction
    df["trend_label"] = np.where(direction == -1, "Long", "Short")

    return df


def get_trend_at_bar(
    df: pd.DataFrame,
    atr_length: int = 10,
    factor: float = 3.0,
    training_period: int = 100,
    highvol_percentile: float = 0.75,
    midvol_percentile: float = 0.50,
    lowvol_percentile: float = 0.25,
) -> str:
    """
    Convenience: compute Adaptive SuperTrend dan return trend bar terakhir.
    "Long" atau "Short".
    """
    result = compute_adaptive_supertrend(
        df,
        atr_length=atr_length,
        factor=factor,
        training_period=training_period,
        highvol_percentile=highvol_percentile,
        midvol_percentile=midvol_percentile,
        lowvol_percentile=lowvol_percentile,
    )

    if result.empty:
        return "Long"

    return str(result["trend_label"].iloc[-1])
