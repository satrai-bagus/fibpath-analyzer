"""
Adaptive SuperTrend Module
--------------------------
Implementasi Python dari indikator "Machine Learning Adaptive SuperTrend"
oleh AlgoAlpha (TradingView).

Menggunakan K-Means clustering untuk mengkategorikan volatilitas ATR
ke dalam 3 cluster (High, Medium, Low), lalu mengadaptasi multiplier
SuperTrend secara dinamis.

Output: "Long" atau "Short" berdasarkan arah SuperTrend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ============================
# K-Means clustering (manual, seperti di Pine Script)
# ============================
def _kmeans_cluster_atr(
    atr_values: np.ndarray,
    n_clusters: int = 3,
    max_iter: int = 100,
    init_percentiles: tuple = (0.75, 0.50, 0.25),
) -> np.ndarray:
    """
    K-Means clustering sederhana untuk array ATR 1D.
    Return: array centroids (high, mid, low) yang sudah sorted descending.
    """
    if len(atr_values) < n_clusters:
        # Fallback: return evenly spaced centroids
        return np.array([np.max(atr_values), np.median(atr_values), np.min(atr_values)])

    # Initial centroids dari percentile
    centroids = np.array([
        np.percentile(atr_values, p * 100) for p in init_percentiles
    ], dtype=float)

    for _ in range(max_iter):
        # Assignment step: assign each point to nearest centroid
        distances = np.abs(atr_values[:, None] - centroids[None, :])  # (N, 3)
        labels = np.argmin(distances, axis=1)

        # Update step: recalculate centroids
        new_centroids = np.array([
            atr_values[labels == k].mean() if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    # Sort descending (high, mid, low)
    centroids = np.sort(centroids)[::-1]
    return centroids


# ============================
# SuperTrend core calculation
# ============================
def _compute_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    factor: float,
) -> tuple:
    """
    Compute SuperTrend bands and direction.

    Returns:
        supertrend: np.ndarray - SuperTrend line values
        direction: np.ndarray - 1 = bullish (Long), -1 = bearish (Short)
    """
    n = len(close)
    hl2 = (high + low) / 2.0

    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr

    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish

    supertrend[0] = upper_band[0]
    direction[0] = -1  # start bearish

    for i in range(1, n):
        # Adjust bands based on previous values
        if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            pass  # keep current lower_band
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            pass  # keep current upper_band
        else:
            upper_band[i] = upper_band[i - 1]

        # Determine direction
        if direction[i - 1] == 1:  # was bullish
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # was bearish
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    return supertrend, direction


# ============================
# ATR calculation
# ============================
def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10) -> np.ndarray:
    """Compute ATR using simple moving average (matching TradingView default)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # RMA (Wilder's smoothing) - same as TradingView's ta.atr
    atr = np.zeros(n)
    atr[:period] = np.nan

    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ============================
# MAIN: Adaptive SuperTrend
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
    Compute the Adaptive SuperTrend with K-Means clustering.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close
    atr_length : ATR period (default 10, matching user's TradingView setting)
    factor : Base SuperTrend multiplier (default 3.0)
    training_period : Lookback for K-Means training data (default 100)
    highvol_percentile : Initial centroid guess for high volatility
    midvol_percentile : Initial centroid guess for mid volatility
    lowvol_percentile : Initial centroid guess for low volatility

    Returns
    -------
    DataFrame with added columns:
        - 'atr': ATR values
        - 'volatility_cluster': 'High', 'Medium', 'Low'
        - 'adaptive_factor': The adapted multiplier
        - 'supertrend': SuperTrend line value
        - 'trend_direction': 1 (bullish/Long) or -1 (bearish/Short)
        - 'trend_label': 'Long' or 'Short'
    """
    df = df.copy()

    # Flatten multi-level columns if needed (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values
    n = len(close)

    # Step 1: Compute ATR
    atr = _compute_atr(high, low, close, period=atr_length)

    # Step 2: For each bar, run K-Means on trailing ATR window
    # and determine adaptive factor
    adaptive_factor = np.full(n, factor)
    cluster_labels = np.full(n, "", dtype=object)

    for i in range(training_period + atr_length, n):
        # Get training window of valid ATR values
        start_idx = max(0, i - training_period)
        atr_window = atr[start_idx:i + 1]
        valid_atr = atr_window[~np.isnan(atr_window)]

        if len(valid_atr) < 10:
            continue

        # Run K-Means
        centroids = _kmeans_cluster_atr(
            valid_atr,
            init_percentiles=(highvol_percentile, midvol_percentile, lowvol_percentile),
        )

        # Classify current ATR
        current_atr = atr[i]
        if np.isnan(current_atr):
            continue

        distances = np.abs(current_atr - centroids)
        cluster_idx = np.argmin(distances)

        # Map cluster to factor adjustment
        # High volatility -> lower factor (tighter, catch reversals faster)
        # Low volatility -> higher factor (wider, avoid false signals)
        if cluster_idx == 0:  # High volatility
            cluster_labels[i] = "High"
            adaptive_factor[i] = factor * 0.75  # tighter
        elif cluster_idx == 1:  # Medium volatility
            cluster_labels[i] = "Medium"
            adaptive_factor[i] = factor * 1.0  # normal
        else:  # Low volatility
            cluster_labels[i] = "Low"
            adaptive_factor[i] = factor * 1.25  # wider

    # Step 3: Compute SuperTrend with adaptive factor
    # We need to compute bar-by-bar since factor changes
    hl2 = (high + low) / 2.0
    upper_band = hl2 + adaptive_factor * atr
    lower_band = hl2 - adaptive_factor * atr

    # Handle NaN in ATR
    upper_band = np.where(np.isnan(atr), np.nan, upper_band)
    lower_band = np.where(np.isnan(atr), np.nan, lower_band)

    supertrend = np.zeros(n)
    direction = np.ones(n)

    # Find first valid index
    first_valid = atr_length
    if first_valid < n:
        supertrend[first_valid] = upper_band[first_valid]
        direction[first_valid] = -1

    for i in range(first_valid + 1, n):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Adjust bands
        if not np.isnan(lower_band[i - 1]):
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass
            else:
                lower_band[i] = lower_band[i - 1]

        if not np.isnan(upper_band[i - 1]):
            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

        # Direction logic
        if direction[i - 1] == 1:  # was bullish
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # was bearish
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    # Build result
    df["atr"] = atr
    df["volatility_cluster"] = cluster_labels
    df["adaptive_factor"] = adaptive_factor
    df["supertrend"] = supertrend
    df["trend_direction"] = direction
    df["trend_label"] = np.where(direction == 1, "Long", "Short")

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
    Convenience function: compute Adaptive SuperTrend and return
    the trend at the LAST bar as 'Long' or 'Short'.

    Parameters match the user's TradingView settings:
    AlgoAlpha Adaptive SuperTrend 10 3 100 0.75 0.5 0.25
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
        return "Long"  # fallback

    return str(result["trend_label"].iloc[-1])
