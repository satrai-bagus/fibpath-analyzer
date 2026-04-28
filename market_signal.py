"""
Market Signal Module
--------------------
Ekstraksi logika kalkulasi sinyal trading dari Analisa.ipynb.
Menghitung Score, Last TR, Raw Position, dan Final Position
secara otomatis dari data market (yfinance) berdasarkan
Ticker, Tanggal, dan Jam.
"""
from __future__ import annotations

import warnings
from datetime import datetime, time, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================
# PARAMETER UTAMA (SAMA PERSIS DENGAN NOTEBOOK)
# ============================
MIN_SCORE_FOR_TRADE = 1        # minimal |score| biar mau entry
ADX_TREND_THRESHOLD = 20.0     # ADX di atas ini dianggap ada trend
ADX_NO_TRADE_THRESHOLD = 15.0  # ADX di bawah ini: NO TRADE
EXTREME_TR_MULT = 1.6          # jika True Range bar terakhir > 1.6 * ATR → NO TRADE
WICK_RATIO = 2                 # wick > ratio * range candle → dihindari


# ============================
# Download data
# ============================
def fetch_data(ticker_symbol: str, start_date, end_date, interval: str = "1h") -> pd.DataFrame:
    stock_data = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    return stock_data


# ============================
# Buang candle yang sedang berjalan jika end time = jam sekarang
# ============================
def drop_incomplete_bar_if_live(df: pd.DataFrame, interval: str, end_dt: datetime) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    else:
        end_dt = end_dt.astimezone(timezone.utc)

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")

    now_utc = datetime.now(timezone.utc)

    def floor_to_interval_start(dt: datetime, itv: str) -> datetime:
        itv = itv.lower().strip()
        if itv.endswith("h"):
            n = int(itv[:-1])
            base = dt.replace(minute=0, second=0, microsecond=0)
            floored_hour = base.hour - (base.hour % n)
            return base.replace(hour=floored_hour)
        if itv.endswith("m"):
            n = int(itv[:-1])
            base = dt.replace(second=0, microsecond=0)
            floored_min = base.minute - (base.minute % n)
            return base.replace(minute=floored_min)
        if itv.endswith("d"):
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt

    current_bar_start = floor_to_interval_start(now_utc, interval)

    if end_dt >= current_bar_start:
        df = df[df.index < current_bar_start]

    return df


# ============================
# Indikator teknikal
# ============================
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.rolling(window=signal_window).mean()
    return macd, signal


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = tr.rolling(window=period, min_periods=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx


# ============================
# Hitung signal & score dari indikator
# ============================
def compute_signal_from_indicators(df: pd.DataFrame) -> Dict:
    """
    Hitung score, raw_position, final_position, last_tr dari DataFrame OHLC.
    Return dict dengan semua nilai yang dibutuhkan.
    """
    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Indikator utama
    ema_fast = compute_ema(close, span=21)
    ema_slow = compute_ema(close, span=50)
    macd, signal = compute_macd(close)
    rsi = compute_rsi(close, period=14)
    atr = compute_atr(high, low, close, period=14)
    adx = compute_adx(high, low, close, period=14)

    # Ambil nilai terakhir
    last_close = float(close.iloc[-1])
    ema_fast_last = float(ema_fast.iloc[-1])
    ema_slow_last = float(ema_slow.iloc[-1])
    macd_last = float(macd.dropna().iloc[-1]) if macd.dropna().size > 0 else np.nan
    signal_last = float(signal.dropna().iloc[-1]) if signal.dropna().size > 0 else np.nan
    rsi_last = float(rsi.dropna().iloc[-1]) if rsi.dropna().size > 0 else np.nan
    atr_last = float(atr.dropna().iloc[-1]) if atr.dropna().size > 0 else np.nan
    adx_last = float(adx.dropna().iloc[-1]) if adx.dropna().size > 0 else np.nan

    # ============================
    # Skoring bullish vs bearish
    # ============================
    score = 0

    # 1) Price vs EMA fast
    if last_close > ema_fast_last:
        score += 1
    else:
        score -= 1

    # 2) EMA fast vs EMA slow (trend)
    if ema_fast_last > ema_slow_last:
        score += 1
    else:
        score -= 1

    # 3) MACD vs Signal (momentum)
    if not np.isnan(macd_last) and not np.isnan(signal_last):
        if macd_last > signal_last:
            score += 1
        else:
            score -= 1

    # 4) MACD di atas / bawah nol
    if not np.isnan(macd_last):
        if macd_last > 0:
            score += 1
        else:
            score -= 1

    # 5) RSI (overbought/oversold soft)
    if not np.isnan(rsi_last):
        if rsi_last > 55:
            score += 1
        elif rsi_last < 45:
            score -= 1

    # 6) ADX (kekuatan trend)
    if not np.isnan(adx_last) and adx_last >= ADX_TREND_THRESHOLD:
        if ema_fast_last > ema_slow_last:
            score += 1
        elif ema_fast_last < ema_slow_last:
            score -= 1

    # ============================
    # Posisi awal (sebelum filter)
    # ============================
    if score >= MIN_SCORE_FOR_TRADE:
        raw_position = "Long"
    elif score <= -MIN_SCORE_FOR_TRADE:
        raw_position = "Short"
    else:
        raw_position = "No Trade"

    # ============================
    # HARD FILTER 1: ADX terlalu rendah → NO TRADE
    # ============================
    filter_reason = []
    position_after_filters = raw_position

    if not np.isnan(adx_last) and adx_last < ADX_NO_TRADE_THRESHOLD:
        position_after_filters = "No Trade"
        filter_reason.append(f"ADX < {ADX_NO_TRADE_THRESHOLD}")

    # ============================
    # HARD FILTER 2: Spike/TR ekstrem di candle terakhir → NO TRADE
    # ============================
    prev_close_val = float(close.iloc[-2]) if len(close) > 1 else float(last_close)
    tr1 = float(abs(high.iloc[-1] - low.iloc[-1]))
    tr2 = float(abs(high.iloc[-1] - prev_close_val))
    tr3 = float(abs(low.iloc[-1] - prev_close_val))
    last_tr = max(tr1, tr2, tr3)

    if not np.isnan(atr_last) and atr_last > 0 and last_tr > EXTREME_TR_MULT * atr_last:
        position_after_filters = "No Trade"
        filter_reason.append(f"Last TR > {EXTREME_TR_MULT} * ATR (spike)")

    return {
        "last_close": last_close,
        "ema_fast_last": ema_fast_last,
        "ema_slow_last": ema_slow_last,
        "macd_last": macd_last,
        "signal_last": signal_last,
        "rsi_last": rsi_last,
        "atr_last": atr_last,
        "adx_last": adx_last,
        "score": score,
        "raw_position": raw_position,
        "final_position": position_after_filters,
        "last_tr": float(last_tr),
        "filter_reason": "; ".join(filter_reason) if filter_reason else "-",
    }


# ============================
# FUNGSI UTAMA: Hitung market signal dari Ticker + Tanggal + Jam
# ============================
def compute_market_signal(
    ticker: str,
    target_date: datetime,
    target_hour: int,
    interval: str = "1h",
    lookback_days: int = 90,
) -> Dict:
    """
    Hitung Score, Last TR, Raw Position, dan Final Position
    secara otomatis dari data market.

    Parameters
    ----------
    ticker : str
        Ticker crypto/saham (misal "ETH-USD", "BTC-USD")
    target_date : datetime atau date
        Tanggal analisis
    target_hour : int
        Jam analisis (0-23, UTC)
    interval : str
        Interval candle (default "1h")
    lookback_days : int
        Berapa hari ke belakang untuk ambil data historis (default 90)

    Returns
    -------
    dict dengan keys:
        - score (int)
        - last_tr (float)
        - raw_position (str): "Long", "Short", atau "No Trade"
        - final_position (str): "Long", "Short", atau "No Trade"
        - last_close (float)
        - adx_last (float)
        - rsi_last (float)
        - atr_last (float)
        - filter_reason (str)
        - error (str or None)
    """
    try:
        start_time = time(target_hour)
        if hasattr(target_date, "hour"):
            target_dt = target_date
        else:
            target_dt = datetime.combine(target_date, start_time)

        # Pastikan UTC aware
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)

        # Ambil data historis
        historical_start = target_dt - timedelta(days=lookback_days)

        data = fetch_data(ticker, historical_start, target_dt, interval)

        if data is None or data.empty:
            return {
                "error": "Data kosong. Coba ubah tanggal/ticker.",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
            }

        df = data.dropna()

        # Buang candle berjalan jika LIVE
        df = drop_incomplete_bar_if_live(df, interval, target_dt)

        if df.empty:
            return {
                "error": "Setelah buang candle LIVE, data jadi kosong.",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
            }

        if len(df) < 60:
            return {
                "error": f"Data terlalu sedikit ({len(df)} bar, butuh minimal 60).",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
            }

        result = compute_signal_from_indicators(df)
        result["error"] = None
        return result

    except Exception as e:
        return {
            "error": f"Error saat mengambil data: {str(e)}",
            "score": 0, "last_tr": 0.0,
            "raw_position": "No Trade", "final_position": "No Trade",
        }
