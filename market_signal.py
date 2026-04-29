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

from adaptive_supertrend import get_trend_at_bar
from squeeze_momentum import get_squeeze_label_at_bar

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


def normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def ensure_utc_datetime(dt: datetime) -> datetime:
    if not isinstance(dt, datetime):
        dt = datetime.combine(dt, time(0))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def interval_to_timedelta(interval: str) -> timedelta:
    itv = interval.lower().strip()
    if itv.endswith("h"):
        return timedelta(hours=int(itv[:-1] or 1))
    if itv.endswith("m"):
        return timedelta(minutes=int(itv[:-1] or 1))
    if itv.endswith("d"):
        return timedelta(days=int(itv[:-1] or 1))
    return timedelta(hours=1)


def floor_to_interval_start(dt: datetime, interval: str) -> datetime:
    dt = ensure_utc_datetime(dt)
    itv = interval.lower().strip()
    if itv.endswith("h"):
        n = int(itv[:-1] or 1)
        base = dt.replace(minute=0, second=0, microsecond=0)
        floored_hour = base.hour - (base.hour % n)
        return base.replace(hour=floored_hour)
    if itv.endswith("m"):
        n = int(itv[:-1] or 1)
        base = dt.replace(second=0, microsecond=0)
        floored_min = base.minute - (base.minute % n)
        return base.replace(minute=floored_min)
    if itv.endswith("d"):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt


def analysis_download_end(target_dt: datetime, interval: str, padding_bars: int = 3) -> datetime:
    target_dt = ensure_utc_datetime(target_dt)
    padding = interval_to_timedelta(interval) * max(int(padding_bars), 1)
    return target_dt + padding


def expected_closed_bar_time(target_dt: datetime, interval: str) -> datetime:
    return floor_to_interval_start(target_dt, interval) - interval_to_timedelta(interval)


def filter_closed_bars_for_analysis(df: pd.DataFrame, interval: str, target_dt: datetime) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = normalize_ohlc_columns(df).copy()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    cutoff = floor_to_interval_start(target_dt, interval)
    live_cutoff = floor_to_interval_start(datetime.now(timezone.utc), interval)
    if cutoff >= live_cutoff:
        cutoff = live_cutoff

    return df[df.index < cutoff]


def latest_closed_bar_time(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    latest = df.index[-1]
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    return latest


def is_latest_bar_stale(df: pd.DataFrame, interval: str, target_dt: datetime) -> bool:
    latest = latest_closed_bar_time(df)
    if latest is None:
        return True
    return latest < pd.Timestamp(expected_closed_bar_time(target_dt, interval))


# ============================
# Buang candle yang sedang berjalan jika end time = jam sekarang
# ============================
def drop_incomplete_bar_if_live(df: pd.DataFrame, interval: str, end_dt: datetime) -> pd.DataFrame:
    return filter_closed_bars_for_analysis(df, interval, end_dt)


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
    df = normalize_ohlc_columns(df)
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

        target_dt = ensure_utc_datetime(target_dt)

        # Ambil data historis
        historical_start = target_dt - timedelta(days=lookback_days)
        download_end = analysis_download_end(target_dt, interval)

        data = fetch_data(ticker, historical_start, download_end, interval)

        if data is None or data.empty:
            return {
                "error": "Data kosong. Coba ubah tanggal/ticker.",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
                "trend": "Long",
                "squeeze_momentum": "Rise weak",
                "squeeze_momentum2": "Rise weak",
            }

        df = normalize_ohlc_columns(data.dropna())

        # Ambil hanya candle yang sudah close sebelum jam trade.
        # Ini sengaja memotong ulang setelah download diberi buffer supaya
        # aman di sekitar transisi hari 23:00-01:00 UTC.
        df = drop_incomplete_bar_if_live(df, interval, target_dt)

        if df.empty:
            return {
                "error": "Setelah buang candle LIVE, data jadi kosong.",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
                "trend": "Long",
                "squeeze_momentum": "Rise weak",
                "squeeze_momentum2": "Rise weak",
            }

        latest_bar = latest_closed_bar_time(df)
        expected_bar = expected_closed_bar_time(target_dt, interval)
        analyzed_bar_time = latest_bar.isoformat() if latest_bar is not None else None
        expected_bar_time = expected_bar.isoformat()

        if is_latest_bar_stale(df, interval, target_dt):
            return {
                "error": (
                    "Data yfinance belum memuat candle yang seharusnya dipakai. "
                    f"Expected {expected_bar_time}, terakhir tersedia {analyzed_bar_time}. "
                    "Coba refresh beberapa menit lagi atau pilih jam setelahnya."
                ),
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
                "trend": "Long",
                "squeeze_momentum": "Rise weak",
                "squeeze_momentum2": "Rise weak",
                "analyzed_bar_time_utc": analyzed_bar_time,
                "expected_bar_time_utc": expected_bar_time,
            }

        if len(df) < 60:
            return {
                "error": f"Data terlalu sedikit ({len(df)} bar, butuh minimal 60).",
                "score": 0, "last_tr": 0.0,
                "raw_position": "No Trade", "final_position": "No Trade",
                "trend": "Long",
                "squeeze_momentum": "Rise weak",
                "squeeze_momentum2": "Rise weak",
                "analyzed_bar_time_utc": analyzed_bar_time,
                "expected_bar_time_utc": expected_bar_time,
            }

        result = compute_signal_from_indicators(df)
        result["analyzed_bar_time_utc"] = analyzed_bar_time
        result["expected_bar_time_utc"] = expected_bar_time

        # Auto-compute Trend dari Adaptive SuperTrend (AlgoAlpha)
        try:
            trend = get_trend_at_bar(
                df,
                atr_length=10,
                factor=3.0,
                training_period=100,
                highvol_percentile=0.75,
                midvol_percentile=0.50,
                lowvol_percentile=0.25,
            )
            result["trend"] = trend
        except Exception:
            result["trend"] = "Long" if result.get("ema_fast_last", 0) > result.get("ema_slow_last", 0) else "Short"

        # Auto-compute Squeeze Momentum (LazyBear)
        # Default parameters: BB=20/2.0, KC=20/1.5
        try:
            sq_mom = get_squeeze_label_at_bar(df)
            result["squeeze_momentum"] = sq_mom
        except Exception:
            result["squeeze_momentum"] = "Rise weak"

        # Squeeze Momentum 2 mengikuti dataset: label Squeeze Momentum
        # dari candle sebelumnya. Jika target jam 13:00, ini memakai 12:00.
        try:
            prev_df = df.iloc[:-1]
            result["squeeze_momentum2"] = get_squeeze_label_at_bar(prev_df) if not prev_df.empty else sq_mom
        except Exception:
            result["squeeze_momentum2"] = "Rise weak"

        result["error"] = None
        return result

    except Exception as e:
        return {
            "error": f"Error saat mengambil data: {str(e)}",
            "score": 0, "last_tr": 0.0,
            "raw_position": "No Trade", "final_position": "No Trade",
            "trend": "Long",
            "squeeze_momentum": "Rise weak",
            "squeeze_momentum2": "Rise weak",
        }
