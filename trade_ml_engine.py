"""
Trade ML Engine
---------------
Pipeline machine learning untuk membuat trade plan per jam:

1. Ambil OHLC 1h dari yfinance.
2. Bangun fitur dari indikator yang sudah ada di proyek.
3. Buat label historis: apakah TP1 kena sebelum SL dalam horizon tertentu.
4. Latih classifier dan cari confidence threshold untuk target winrate.
5. Prediksi action, entry, stop loss, TP1, dan TP2 untuk jam analisis.

Catatan penting:
- Winrate 70% diperlakukan sebagai target evaluasi, bukan angka yang dipaksakan.
- Test set default dibatasi 2 hari terakhir yang masih punya label future bars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from adaptive_supertrend import compute_adaptive_supertrend
from market_signal import (
    ADX_NO_TRADE_THRESHOLD,
    ADX_TREND_THRESHOLD,
    EXTREME_TR_MULT,
    MIN_SCORE_FOR_TRADE,
    analysis_download_end,
    compute_adx,
    compute_atr,
    compute_ema,
    compute_macd,
    compute_rsi,
    drop_incomplete_bar_if_live,
    ensure_utc_datetime,
    expected_closed_bar_time,
    fetch_data,
    is_latest_bar_stale,
    latest_closed_bar_time,
)
from squeeze_momentum import compute_squeeze_momentum


MODEL_VERSION = "trade_ml_v1"

CATEGORICAL_FEATURES = [
    "trend",
    "squeeze_momentum",
    "squeeze_momentum2",
    "raw_position",
    "final_position",
    "volatility_cluster",
]

NUMERIC_FEATURES = [
    "hour_utc",
    "day_of_week",
    "score",
    "last_tr",
    "last_close",
    "atr",
    "adx",
    "rsi",
    "ema_gap_pct",
    "macd",
    "macd_signal",
    "range_pct",
    "last_tr_atr",
    "close_to_supertrend_pct",
    "return_1h",
    "return_3h",
    "return_6h",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class TradeModelConfig:
    interval: str = "1h"
    lookback_days: int = 729
    test_days: int = 2
    horizon_hours: int = 24
    sl_atr_mult: float = 1.4
    tp1_r: float = 0.8
    tp2_r: float = 1.6
    min_risk_pct: float = 0.002
    target_winrate: float = 0.70
    min_confident_trades: int = 5
    random_state: int = 42
    leverage: float = 75.0
    adaptive_risk: bool = True
    low_vol_risk_factor: float = 0.85
    medium_vol_risk_factor: float = 1.0
    high_vol_risk_factor: float = 1.30
    high_range_risk_factor: float = 1.15
    low_range_risk_factor: float = 0.90


@dataclass
class TradePrediction:
    ticker: str
    timestamp: Optional[pd.Timestamp]
    action: str
    side: str
    probability_win_tp1: float
    confidence_threshold: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk: float
    risk_atr_mult: float
    reward_r_tp1: float
    reward_r_tp2: float
    leverage: float
    stop_loss_pnl_pct_leveraged: float
    take_profit_1_pnl_pct_leveraged: float
    take_profit_2_pnl_pct_leveraged: float
    model_winrate_2d: float
    model_trades_2d: int
    reason: str


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_config(value: object) -> TradeModelConfig:
    if isinstance(value, TradeModelConfig):
        return value
    if isinstance(value, dict):
        valid_keys = {field.name for field in fields(TradeModelConfig)}
        clean = {key: item for key, item in value.items() if key in valid_keys}
        return TradeModelConfig(**clean)
    return TradeModelConfig()


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLC tidak lengkap: {missing}")

    df = df.dropna(subset=required).copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def download_training_ohlc(
    ticker: str,
    lookback_days: int = 729,
    interval: str = "1h",
) -> pd.DataFrame:
    days = int(max(30, min(lookback_days, 729)))
    data = yf.download(
        ticker,
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(f"Data kosong untuk ticker {ticker}.")
    return _normalize_ohlc(data)


def _score_positions(ind: pd.DataFrame) -> pd.DataFrame:
    out = ind.copy()
    score = pd.Series(0, index=out.index, dtype=float)

    score += np.where(out["last_close"] > out["ema_fast"], 1, -1)
    score += np.where(out["ema_fast"] > out["ema_slow"], 1, -1)

    macd_valid = out["macd"].notna() & out["macd_signal"].notna()
    score += np.where(macd_valid & (out["macd"] > out["macd_signal"]), 1, 0)
    score += np.where(macd_valid & (out["macd"] <= out["macd_signal"]), -1, 0)

    score += np.where(out["macd"].notna() & (out["macd"] > 0), 1, 0)
    score += np.where(out["macd"].notna() & (out["macd"] <= 0), -1, 0)

    score += np.where(out["rsi"].notna() & (out["rsi"] > 55), 1, 0)
    score += np.where(out["rsi"].notna() & (out["rsi"] < 45), -1, 0)

    trend_ok = out["adx"].notna() & (out["adx"] >= ADX_TREND_THRESHOLD)
    score += np.where(trend_ok & (out["ema_fast"] > out["ema_slow"]), 1, 0)
    score += np.where(trend_ok & (out["ema_fast"] < out["ema_slow"]), -1, 0)

    out["score"] = score.astype(int)
    out["raw_position"] = np.where(
        out["score"] >= MIN_SCORE_FOR_TRADE,
        "Long",
        np.where(out["score"] <= -MIN_SCORE_FOR_TRADE, "Short", "No Trade"),
    )

    final_position = pd.Series(out["raw_position"], index=out.index, dtype=object)
    final_position = final_position.mask(out["adx"].notna() & (out["adx"] < ADX_NO_TRADE_THRESHOLD), "No Trade")
    final_position = final_position.mask(
        out["atr"].notna()
        & (out["atr"] > 0)
        & (out["last_tr"] > EXTREME_TR_MULT * out["atr"]),
        "No Trade",
    )
    out["final_position"] = final_position
    return out


def _adaptive_risk_factor(
    config: TradeModelConfig,
    volatility_cluster: Optional[str] = None,
    last_tr_atr: Optional[float] = None,
) -> float:
    if not config.adaptive_risk:
        return 1.0

    cluster = str(volatility_cluster or "").strip().lower()
    if cluster == "high":
        factor = config.high_vol_risk_factor
    elif cluster == "low":
        factor = config.low_vol_risk_factor
    else:
        factor = config.medium_vol_risk_factor

    if last_tr_atr is not None and not np.isnan(float(last_tr_atr)):
        if float(last_tr_atr) >= 1.35:
            factor *= config.high_range_risk_factor
        elif float(last_tr_atr) <= 0.75:
            factor *= config.low_range_risk_factor

    return float(min(max(factor, 0.70), 1.75))


def _risk_levels(
    entry: float,
    atr: float,
    side: str,
    config: TradeModelConfig,
    volatility_cluster: Optional[str] = None,
    last_tr_atr: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    risk_atr_mult = config.sl_atr_mult * _adaptive_risk_factor(config, volatility_cluster, last_tr_atr)
    risk = max(float(atr) * risk_atr_mult, float(entry) * config.min_risk_pct)
    if side == "Long":
        stop_loss = entry - risk
        take_profit_1 = entry + risk * config.tp1_r
        take_profit_2 = entry + risk * config.tp2_r
    elif side == "Short":
        stop_loss = entry + risk
        take_profit_1 = entry - risk * config.tp1_r
        take_profit_2 = entry - risk * config.tp2_r
    else:
        stop_loss = np.nan
        take_profit_1 = np.nan
        take_profit_2 = np.nan
    return float(stop_loss), float(take_profit_1), float(take_profit_2), float(risk), float(risk_atr_mult)


def _leveraged_pnl_pct(entry: float, exit_price: float, side: str, leverage: float) -> float:
    if np.isnan(entry) or np.isnan(exit_price) or entry == 0:
        return np.nan
    if side == "Long":
        return float(((exit_price - entry) / entry) * leverage * 100.0)
    if side == "Short":
        return float(((entry - exit_price) / entry) * leverage * 100.0)
    return np.nan


def _evaluate_trade_path(
    high: np.ndarray,
    low: np.ndarray,
    i: int,
    side: str,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    horizon_hours: int,
) -> Dict[str, object]:
    hit_tp1 = False
    hit_tp2 = False
    bars_to_tp1 = np.nan
    bars_to_tp2 = np.nan
    first_exit = "TIMEOUT"

    for step, j in enumerate(range(i + 1, i + horizon_hours + 1), start=1):
        if side == "Long":
            sl_hit = low[j] <= stop_loss
            tp1_hit = high[j] >= take_profit_1
            tp2_hit = high[j] >= take_profit_2
        else:
            sl_hit = high[j] >= stop_loss
            tp1_hit = low[j] <= take_profit_1
            tp2_hit = low[j] <= take_profit_2

        if sl_hit and (tp1_hit or tp2_hit):
            if not hit_tp1:
                return {
                    "label_win_tp1": 0,
                    "label_win_tp2": 0,
                    "first_exit": "AMBIGUOUS_SL_FIRST",
                    "bars_to_event": step,
                    "bars_to_tp1": np.nan,
                    "bars_to_tp2": np.nan,
                }
            first_exit = "TP1_THEN_AMBIGUOUS"
            break

        if sl_hit:
            if not hit_tp1:
                return {
                    "label_win_tp1": 0,
                    "label_win_tp2": 0,
                    "first_exit": "SL",
                    "bars_to_event": step,
                    "bars_to_tp1": np.nan,
                    "bars_to_tp2": np.nan,
                }
            first_exit = "TP1_THEN_SL"
            break

        if tp2_hit:
            hit_tp1 = True
            hit_tp2 = True
            bars_to_tp1 = bars_to_tp1 if not np.isnan(bars_to_tp1) else float(step)
            bars_to_tp2 = float(step)
            first_exit = "TP2"
            break

        if tp1_hit and not hit_tp1:
            hit_tp1 = True
            bars_to_tp1 = float(step)
            first_exit = "TP1"

    return {
        "label_win_tp1": int(hit_tp1),
        "label_win_tp2": int(hit_tp2),
        "first_exit": first_exit,
        "bars_to_event": bars_to_tp2 if hit_tp2 else bars_to_tp1 if hit_tp1 else float(horizon_hours),
        "bars_to_tp1": bars_to_tp1,
        "bars_to_tp2": bars_to_tp2,
    }


def build_trade_dataset(
    ohlc: pd.DataFrame,
    ticker: str,
    config: Optional[TradeModelConfig] = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    config = config or TradeModelConfig()
    df = _normalize_ohlc(ohlc)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    prev_close = close.shift(1)

    ind = pd.DataFrame(index=df.index)
    ind["ticker"] = ticker
    ind["timestamp"] = df.index
    ind["hour_utc"] = df.index.hour
    ind["day_of_week"] = df.index.dayofweek
    ind["open"] = open_
    ind["high"] = high
    ind["low"] = low
    ind["last_close"] = close

    ind["ema_fast"] = compute_ema(close, span=21)
    ind["ema_slow"] = compute_ema(close, span=50)
    ind["macd"], ind["macd_signal"] = compute_macd(close)
    ind["rsi"] = compute_rsi(close, period=14)
    ind["atr"] = compute_atr(high, low, close, period=14)
    ind["adx"] = compute_adx(high, low, close, period=14)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ind["last_tr"] = tr

    supertrend_df = compute_adaptive_supertrend(df)
    squeeze_df = compute_squeeze_momentum(df)

    ind["trend"] = supertrend_df["trend_label"].astype(str)
    ind["volatility_cluster"] = supertrend_df["volatility_cluster"].replace("", "Unknown").astype(str)
    ind["supertrend"] = supertrend_df["supertrend"]
    ind["squeeze_momentum"] = squeeze_df["squeeze_label"].replace("", np.nan).ffill().fillna("Rise weak")
    ind["squeeze_momentum2"] = ind["squeeze_momentum"].shift(1).fillna("Rise weak")

    ind["ema_gap_pct"] = (ind["ema_fast"] - ind["ema_slow"]) / close.replace(0, np.nan)
    ind["range_pct"] = (high - low) / close.replace(0, np.nan)
    ind["last_tr_atr"] = ind["last_tr"] / ind["atr"].replace(0, np.nan)
    ind["close_to_supertrend_pct"] = (close - ind["supertrend"]) / close.replace(0, np.nan)
    ind["return_1h"] = close.pct_change(1)
    ind["return_3h"] = close.pct_change(3)
    ind["return_6h"] = close.pct_change(6)

    ind = _score_positions(ind)

    for col in CATEGORICAL_FEATURES:
        ind[col] = ind[col].fillna("Unknown").astype(str)

    label_cols = [
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "risk",
        "risk_atr_mult",
        "label_win_tp1",
        "label_win_tp2",
        "first_exit",
        "bars_to_event",
        "bars_to_tp1",
        "bars_to_tp2",
        "label_available",
    ]
    for col in label_cols:
        ind[col] = np.nan
    ind["first_exit"] = ""
    ind["label_available"] = False

    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    atr_arr = ind["atr"].values
    final_arr = ind["final_position"].values

    if include_labels:
        max_i = len(ind) - config.horizon_hours - 1
        for i in range(max_i + 1):
            side = str(final_arr[i])
            if side not in {"Long", "Short"}:
                continue
            if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
                continue

            entry = float(close_arr[i])
            stop_loss, take_profit_1, take_profit_2, risk, risk_atr_mult = _risk_levels(
                entry,
                float(atr_arr[i]),
                side,
                config,
                volatility_cluster=ind["volatility_cluster"].iloc[i],
                last_tr_atr=ind["last_tr_atr"].iloc[i],
            )
            path = _evaluate_trade_path(
                high_arr,
                low_arr,
                i,
                side,
                stop_loss,
                take_profit_1,
                take_profit_2,
                config.horizon_hours,
            )

            ind.iat[i, ind.columns.get_loc("entry")] = entry
            ind.iat[i, ind.columns.get_loc("stop_loss")] = stop_loss
            ind.iat[i, ind.columns.get_loc("take_profit_1")] = take_profit_1
            ind.iat[i, ind.columns.get_loc("take_profit_2")] = take_profit_2
            ind.iat[i, ind.columns.get_loc("risk")] = risk
            ind.iat[i, ind.columns.get_loc("risk_atr_mult")] = risk_atr_mult
            ind.iat[i, ind.columns.get_loc("label_win_tp1")] = path["label_win_tp1"]
            ind.iat[i, ind.columns.get_loc("label_win_tp2")] = path["label_win_tp2"]
            ind.iat[i, ind.columns.get_loc("first_exit")] = path["first_exit"]
            ind.iat[i, ind.columns.get_loc("bars_to_event")] = path["bars_to_event"]
            ind.iat[i, ind.columns.get_loc("bars_to_tp1")] = path["bars_to_tp1"]
            ind.iat[i, ind.columns.get_loc("bars_to_tp2")] = path["bars_to_tp2"]
            ind.iat[i, ind.columns.get_loc("label_available")] = True

    return ind.reset_index(drop=True)


def _build_pipeline(config: TradeModelConfig) -> Pipeline:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=9,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=config.random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def _positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if 1 in classes:
        pos_idx = classes.index(1)
    elif True in classes:
        pos_idx = classes.index(True)
    else:
        return np.zeros(len(X), dtype=float)
    return probs[:, pos_idx]


def _select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: TradeModelConfig,
) -> Tuple[float, Dict[str, float]]:
    min_trades = min(config.min_confident_trades, max(1, len(y_true)))
    candidates = np.round(np.arange(0.50, 0.91, 0.02), 2)
    rows = []
    for threshold in candidates:
        mask = probabilities >= threshold
        trades = int(mask.sum())
        wins = int(y_true[mask].sum()) if trades else 0
        winrate = float(wins / trades) if trades else 0.0
        rows.append((threshold, trades, wins, winrate))

    viable = [r for r in rows if r[1] >= min_trades and r[3] >= config.target_winrate]
    if viable:
        threshold, trades, wins, winrate = sorted(viable, key=lambda r: (r[1], r[3]), reverse=True)[0]
    else:
        with_trades = [r for r in rows if r[1] >= min_trades] or [r for r in rows if r[1] > 0]
        threshold, trades, wins, winrate = sorted(with_trades, key=lambda r: (r[3], r[1]), reverse=True)[0]

    return float(threshold), {
        "validation_trades": float(trades),
        "validation_wins": float(wins),
        "validation_winrate": float(winrate),
    }


def _split_train_validation(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_df) < 120:
        return train_df, train_df
    split_idx = max(int(len(train_df) * 0.8), len(train_df) - 240)
    split_idx = min(max(split_idx, 1), len(train_df) - 1)
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def _make_test_split(dataset: pd.DataFrame, config: TradeModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    rows = dataset[
        dataset["label_available"].astype(bool)
        & dataset["final_position"].isin(["Long", "Short"])
    ].copy()
    rows = rows.dropna(subset=MODEL_FEATURES + ["label_win_tp1"]).sort_values("timestamp").reset_index(drop=True)
    if rows.empty:
        raise ValueError("Dataset tidak punya baris trade berlabel.")

    max_ts = pd.Timestamp(rows["timestamp"].max())
    test_start = max_ts - pd.Timedelta(days=config.test_days)
    train_df = rows[rows["timestamp"] < test_start].copy()
    test_df = rows[rows["timestamp"] >= test_start].copy()

    if train_df.empty or test_df.empty:
        fallback_count = min(48, max(1, len(rows) // 5))
        train_df = rows.iloc[:-fallback_count].copy()
        test_df = rows.iloc[-fallback_count:].copy()
        test_start = pd.Timestamp(test_df["timestamp"].min())

    if train_df.empty or test_df.empty:
        raise ValueError("Data training/test tidak cukup setelah split 2 hari.")

    return train_df, test_df, test_start


def _evaluate_backtest(model: Pipeline, test_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = test_df.copy()
    out["pred_prob_win_tp1"] = _positive_probability(model, out[MODEL_FEATURES])
    out["pred_take_trade"] = out["pred_prob_win_tp1"] >= threshold
    out["pred_win"] = out["label_win_tp1"].astype(int)
    return out


def _summarize_backtest(backtest_df: pd.DataFrame, threshold: float, config: TradeModelConfig) -> Dict[str, float]:
    taken = backtest_df[backtest_df["pred_take_trade"].astype(bool)]
    total_setups = int(len(backtest_df))
    baseline_winrate = float(backtest_df["label_win_tp1"].astype(float).mean()) if total_setups else 0.0
    trades = int(len(taken))
    wins = int(taken["label_win_tp1"].astype(int).sum()) if trades else 0
    winrate = float(wins / trades) if trades else 0.0
    return {
        "threshold": float(threshold),
        "target_winrate": float(config.target_winrate),
        "test_days": float(config.test_days),
        "test_setups": float(total_setups),
        "test_trades": float(trades),
        "test_wins": float(wins),
        "test_winrate": float(winrate),
        "test_target_met": float(winrate >= config.target_winrate and trades > 0),
        "baseline_winrate": float(baseline_winrate),
        "coverage": float(trades / total_setups) if total_setups else 0.0,
        "expectancy_r_tp1": float(winrate * config.tp1_r - (1.0 - winrate)) if trades else 0.0,
        "sl_atr_mult": float(config.sl_atr_mult),
        "tp1_r": float(config.tp1_r),
        "tp2_r": float(config.tp2_r),
        "leverage": float(config.leverage),
    }


def train_trade_model(
    dataset: pd.DataFrame,
    config: Optional[TradeModelConfig] = None,
) -> Dict[str, object]:
    config = config or TradeModelConfig()
    train_df, test_df, test_start = _make_test_split(dataset, config)
    fit_df, val_df = _split_train_validation(train_df)

    y_fit = fit_df["label_win_tp1"].astype(int)
    if y_fit.nunique() < 2:
        raise ValueError("Training gagal: label hanya punya satu kelas. Tambah data historis.")

    threshold_model = _build_pipeline(config)
    threshold_model.fit(fit_df[MODEL_FEATURES], y_fit)
    val_prob = _positive_probability(threshold_model, val_df[MODEL_FEATURES])
    threshold, validation_summary = _select_threshold(val_df["label_win_tp1"].astype(int).values, val_prob, config)

    final_model = _build_pipeline(config)
    final_model.fit(train_df[MODEL_FEATURES], train_df["label_win_tp1"].astype(int))
    backtest_df = _evaluate_backtest(final_model, test_df, threshold)
    summary = _summarize_backtest(backtest_df, threshold, config)
    summary.update(validation_summary)
    summary["train_rows"] = float(len(train_df))
    summary["test_start"] = str(test_start)
    summary["model_version"] = MODEL_VERSION

    return {
        "model": final_model,
        "config": asdict(config),
        "summary": summary,
        "backtest": backtest_df,
        "feature_columns": MODEL_FEATURES,
    }


def save_trade_model(payload: Dict[str, object], model_path: str | Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)


def load_trade_model(model_path: str | Path) -> Dict[str, object]:
    return joblib.load(model_path)


def train_and_save_trade_model(
    ticker: str,
    model_path: str | Path = "trade_ml_model.pkl",
    dataset_csv: Optional[str | Path] = "trade_ml_dataset.csv",
    backtest_csv: Optional[str | Path] = "trade_ml_backtest.csv",
    config: Optional[TradeModelConfig] = None,
) -> Dict[str, object]:
    config = config or TradeModelConfig()
    ohlc = download_training_ohlc(ticker, lookback_days=config.lookback_days, interval=config.interval)
    dataset = build_trade_dataset(ohlc, ticker=ticker, config=config, include_labels=True)
    payload = train_trade_model(dataset, config=config)
    payload["ticker"] = ticker
    payload["dataset_rows"] = int(len(dataset))
    payload["dataset_label_rows"] = int(dataset["label_available"].astype(bool).sum())
    save_trade_model(payload, model_path)

    if dataset_csv is not None:
        Path(dataset_csv).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(dataset_csv, index=False)
    if backtest_csv is not None:
        Path(backtest_csv).parent.mkdir(parents=True, exist_ok=True)
        payload["backtest"].to_csv(backtest_csv, index=False)
    return payload


def _target_datetime(target_date: datetime, target_hour: int) -> datetime:
    start_time = time(int(target_hour))
    if hasattr(target_date, "hour"):
        target_dt = target_date
    else:
        target_dt = datetime.combine(target_date, start_time)
    return ensure_utc_datetime(target_dt)


def predict_trade_from_ohlc(
    payload: Dict[str, object],
    ohlc: pd.DataFrame,
    ticker: str,
) -> TradePrediction:
    config = _coerce_config(payload.get("config", {}))
    model: Pipeline = payload["model"]
    summary: Dict[str, float] = payload.get("summary", {})
    threshold = float(summary.get("threshold", 0.60))

    dataset = build_trade_dataset(ohlc, ticker=ticker, config=config, include_labels=False)
    if dataset.empty:
        raise ValueError("Tidak ada baris fitur untuk prediksi.")

    row = dataset.iloc[-1].copy()
    side = str(row.get("final_position", "No Trade"))
    entry = float(row.get("last_close", np.nan))
    atr = float(row.get("atr", np.nan))

    if side not in {"Long", "Short"} or np.isnan(entry) or np.isnan(atr) or atr <= 0:
        score = row.get("score", np.nan)
        adx = row.get("adx", np.nan)
        ema_fast = row.get("ema_fast", np.nan)
        ema_slow = row.get("ema_slow", np.nan)
        trend = row.get("trend", "Unknown")
        raw_position = row.get("raw_position", "Unknown")
        if side not in {"Long", "Short"}:
            reason = (
                f"ML skip karena final_position={side} "
                f"(raw={raw_position}, score={score}, trend={trend}, "
                f"ADX={float(adx):.2f} jika valid, EMA fast={float(ema_fast):.2f}, EMA slow={float(ema_slow):.2f})."
            )
        else:
            reason = "ML skip karena entry atau ATR belum valid."

        return TradePrediction(
            ticker=ticker,
            timestamp=row.get("timestamp"),
            action="No Trade",
            side=side,
            probability_win_tp1=0.0,
            confidence_threshold=threshold,
            entry=entry,
            stop_loss=np.nan,
            take_profit_1=np.nan,
            take_profit_2=np.nan,
            risk=np.nan,
            risk_atr_mult=np.nan,
            reward_r_tp1=config.tp1_r,
            reward_r_tp2=config.tp2_r,
            leverage=config.leverage,
            stop_loss_pnl_pct_leveraged=np.nan,
            take_profit_1_pnl_pct_leveraged=np.nan,
            take_profit_2_pnl_pct_leveraged=np.nan,
            model_winrate_2d=float(summary.get("test_winrate", 0.0)),
            model_trades_2d=int(summary.get("test_trades", 0)),
            reason=reason,
        )

    prob = float(_positive_probability(model, pd.DataFrame([row])[MODEL_FEATURES])[0])
    stop_loss, take_profit_1, take_profit_2, risk, risk_atr_mult = _risk_levels(
        entry,
        atr,
        side,
        config,
        volatility_cluster=row.get("volatility_cluster"),
        last_tr_atr=row.get("last_tr_atr"),
    )
    action = side if prob >= threshold else "No Trade"
    reason = "Probability lolos threshold." if action != "No Trade" else "Probability di bawah threshold model."

    return TradePrediction(
        ticker=ticker,
        timestamp=row.get("timestamp"),
        action=action,
        side=side,
        probability_win_tp1=prob,
        confidence_threshold=threshold,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk=risk,
        risk_atr_mult=risk_atr_mult,
        reward_r_tp1=config.tp1_r,
        reward_r_tp2=config.tp2_r,
        leverage=config.leverage,
        stop_loss_pnl_pct_leveraged=_leveraged_pnl_pct(entry, stop_loss, side, config.leverage),
        take_profit_1_pnl_pct_leveraged=_leveraged_pnl_pct(entry, take_profit_1, side, config.leverage),
        take_profit_2_pnl_pct_leveraged=_leveraged_pnl_pct(entry, take_profit_2, side, config.leverage),
        model_winrate_2d=float(summary.get("test_winrate", 0.0)),
        model_trades_2d=int(summary.get("test_trades", 0)),
        reason=reason,
    )


def _prediction_for_side(
    payload: Dict[str, object],
    row: pd.Series,
    ticker: str,
    side: str,
    reason_prefix: str,
) -> TradePrediction:
    config = _coerce_config(payload.get("config", {}))
    model: Pipeline = payload["model"]
    summary: Dict[str, float] = payload.get("summary", {})
    threshold = float(summary.get("threshold", 0.60))

    scenario = row.copy()
    score_magnitude = max(abs(float(scenario.get("score", 0.0))), 1.0)
    scenario["raw_position"] = side
    scenario["final_position"] = side
    scenario["score"] = score_magnitude if side == "Long" else -score_magnitude

    entry = float(scenario.get("last_close", np.nan))
    atr = float(scenario.get("atr", np.nan))
    if side not in {"Long", "Short"} or np.isnan(entry) or np.isnan(atr) or atr <= 0:
        return TradePrediction(
            ticker=ticker,
            timestamp=scenario.get("timestamp"),
            action="No Trade",
            side=side,
            probability_win_tp1=0.0,
            confidence_threshold=threshold,
            entry=entry,
            stop_loss=np.nan,
            take_profit_1=np.nan,
            take_profit_2=np.nan,
            risk=np.nan,
            risk_atr_mult=np.nan,
            reward_r_tp1=config.tp1_r,
            reward_r_tp2=config.tp2_r,
            leverage=config.leverage,
            stop_loss_pnl_pct_leveraged=np.nan,
            take_profit_1_pnl_pct_leveraged=np.nan,
            take_profit_2_pnl_pct_leveraged=np.nan,
            model_winrate_2d=float(summary.get("test_winrate", 0.0)),
            model_trades_2d=int(summary.get("test_trades", 0)),
            reason=f"{reason_prefix}: ATR belum valid.",
        )

    prob = float(_positive_probability(model, pd.DataFrame([scenario])[MODEL_FEATURES])[0])
    stop_loss, take_profit_1, take_profit_2, risk, risk_atr_mult = _risk_levels(
        entry,
        atr,
        side,
        config,
        volatility_cluster=scenario.get("volatility_cluster"),
        last_tr_atr=scenario.get("last_tr_atr"),
    )
    action = side if prob >= threshold else "No Trade"
    reason = (
        f"{reason_prefix}: confidence lolos threshold."
        if action != "No Trade"
        else f"{reason_prefix}: confidence di bawah threshold."
    )

    return TradePrediction(
        ticker=ticker,
        timestamp=scenario.get("timestamp"),
        action=action,
        side=side,
        probability_win_tp1=prob,
        confidence_threshold=threshold,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk=risk,
        risk_atr_mult=risk_atr_mult,
        reward_r_tp1=config.tp1_r,
        reward_r_tp2=config.tp2_r,
        leverage=config.leverage,
        stop_loss_pnl_pct_leveraged=_leveraged_pnl_pct(entry, stop_loss, side, config.leverage),
        take_profit_1_pnl_pct_leveraged=_leveraged_pnl_pct(entry, take_profit_1, side, config.leverage),
        take_profit_2_pnl_pct_leveraged=_leveraged_pnl_pct(entry, take_profit_2, side, config.leverage),
        model_winrate_2d=float(summary.get("test_winrate", 0.0)),
        model_trades_2d=int(summary.get("test_trades", 0)),
        reason=reason,
    )


def predict_trade_scenarios_from_ohlc(
    payload: Dict[str, object],
    ohlc: pd.DataFrame,
    ticker: str,
) -> Dict[str, TradePrediction]:
    config = _coerce_config(payload.get("config", {}))
    dataset = build_trade_dataset(ohlc, ticker=ticker, config=config, include_labels=False)
    if dataset.empty:
        raise ValueError("Tidak ada baris fitur untuk prediksi.")

    row = dataset.iloc[-1].copy()
    return {
        "Long": _prediction_for_side(payload, row, ticker, "Long", "Skenario Long"),
        "Short": _prediction_for_side(payload, row, ticker, "Short", "Skenario Short"),
    }


def analyze_trade_from_ohlc(
    payload: Dict[str, object],
    ohlc: pd.DataFrame,
    ticker: str,
) -> Tuple[TradePrediction, Dict[str, TradePrediction]]:
    primary = predict_trade_from_ohlc(payload, ohlc, ticker=ticker)
    scenarios = predict_trade_scenarios_from_ohlc(payload, ohlc, ticker=ticker)
    return primary, scenarios


def analyze_trade_for_hour(
    ticker: str,
    target_date: datetime,
    target_hour: int,
    model_path: str | Path = "trade_ml_model.pkl",
    lookback_days: int = 120,
    interval: str = "1h",
) -> Tuple[TradePrediction, Dict[str, TradePrediction]]:
    payload = load_trade_model(model_path)
    target_dt = _target_datetime(target_date, target_hour)
    historical_start = target_dt - timedelta(days=lookback_days)
    data = fetch_data(ticker, historical_start, analysis_download_end(target_dt, interval), interval)
    if data is None or data.empty:
        raise ValueError("Data market kosong untuk prediksi trade.")
    df = drop_incomplete_bar_if_live(_normalize_ohlc(data), interval, target_dt)
    if is_latest_bar_stale(df, interval, target_dt):
        latest = latest_closed_bar_time(df)
        expected = expected_closed_bar_time(target_dt, interval)
        raise ValueError(
            "Data yfinance belum memuat candle ML yang seharusnya dipakai "
            f"(expected {expected.isoformat()}, terakhir tersedia {latest})."
        )
    if len(df) < 120:
        raise ValueError(f"Data market terlalu sedikit ({len(df)} bar).")
    return analyze_trade_from_ohlc(payload, df, ticker=ticker)


def predict_trade_for_hour(
    ticker: str,
    target_date: datetime,
    target_hour: int,
    model_path: str | Path = "trade_ml_model.pkl",
    lookback_days: int = 120,
    interval: str = "1h",
) -> TradePrediction:
    payload = load_trade_model(model_path)
    target_dt = _target_datetime(target_date, target_hour)
    historical_start = target_dt - timedelta(days=lookback_days)
    data = fetch_data(ticker, historical_start, analysis_download_end(target_dt, interval), interval)
    if data is None or data.empty:
        raise ValueError("Data market kosong untuk prediksi trade.")
    df = drop_incomplete_bar_if_live(_normalize_ohlc(data), interval, target_dt)
    if is_latest_bar_stale(df, interval, target_dt):
        latest = latest_closed_bar_time(df)
        expected = expected_closed_bar_time(target_dt, interval)
        raise ValueError(
            "Data yfinance belum memuat candle ML yang seharusnya dipakai "
            f"(expected {expected.isoformat()}, terakhir tersedia {latest})."
        )
    if len(df) < 120:
        raise ValueError(f"Data market terlalu sedikit ({len(df)} bar).")
    return predict_trade_from_ohlc(payload, df, ticker=ticker)


def format_summary(summary: Dict[str, float]) -> str:
    return (
        f"test_winrate={summary.get('test_winrate', 0.0):.2%}, "
        f"test_trades={int(summary.get('test_trades', 0))}, "
        f"threshold={summary.get('threshold', 0.0):.2f}, "
        f"expectancy={summary.get('expectancy_r_tp1', 0.0):.2f}R, "
        f"SL={summary.get('sl_atr_mult', 0.0):.2f}ATR, "
        f"TP1={summary.get('tp1_r', 0.0):.2f}R, "
        f"TP2={summary.get('tp2_r', 0.0):.2f}R, "
        f"lev={summary.get('leverage', 0.0):.0f}x, "
        f"target_met={bool(summary.get('test_target_met', 0.0))}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset, train model, and backtest trade ML engine.")
    parser.add_argument("--ticker", default="ETH-USD")
    parser.add_argument("--lookback-days", type=int, default=729)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--sl-atr-mult", type=float, default=1.4)
    parser.add_argument("--tp1-r", type=float, default=0.8)
    parser.add_argument("--tp2-r", type=float, default=1.6)
    parser.add_argument("--leverage", type=float, default=75.0)
    parser.add_argument("--model-path", default="trade_ml_model.pkl")
    parser.add_argument("--dataset-csv", default="trade_ml_dataset.csv")
    parser.add_argument("--backtest-csv", default="trade_ml_backtest.csv")
    args = parser.parse_args()

    cfg = TradeModelConfig(
        lookback_days=args.lookback_days,
        test_days=args.test_days,
        horizon_hours=args.horizon_hours,
        sl_atr_mult=args.sl_atr_mult,
        tp1_r=args.tp1_r,
        tp2_r=args.tp2_r,
        leverage=args.leverage,
    )
    result = train_and_save_trade_model(
        ticker=args.ticker,
        model_path=args.model_path,
        dataset_csv=args.dataset_csv,
        backtest_csv=args.backtest_csv,
        config=cfg,
    )
    print(format_summary(result["summary"]))
    print(f"dataset_rows={result['dataset_rows']} label_rows={result['dataset_label_rows']}")
