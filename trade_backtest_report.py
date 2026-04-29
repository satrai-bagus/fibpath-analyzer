from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trade_ml_engine import (
    MODEL_FEATURES,
    _coerce_config,
    _evaluate_trade_path,
    _leveraged_pnl_pct,
    _positive_probability,
    _risk_levels,
    load_trade_model,
)


STRATEGIES = ["primary_layak", "scenario_layak", "best_every_hour"]


@dataclass
class BacktestConfig:
    initial_capital: float = 100.0
    stake_per_trade: float = 10.0
    cap_loss_at_margin: bool = True


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset harus punya kolom timestamp.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _scenario_row(row: pd.Series, side: str) -> pd.Series:
    scenario = row.copy()
    score_magnitude = max(abs(float(scenario.get("score", 0.0))), 1.0)
    scenario["raw_position"] = side
    scenario["final_position"] = side
    scenario["score"] = score_magnitude if side == "Long" else -score_magnitude
    return scenario


def _probability(payload: Dict[str, object], row: pd.Series) -> float:
    model = payload["model"]
    X = pd.DataFrame([row])[MODEL_FEATURES]
    return float(_positive_probability(model, X)[0])


def _pick_trade(payload: Dict[str, object], row: pd.Series, strategy: str, threshold: float) -> Optional[Dict[str, object]]:
    if strategy == "primary_layak":
        side = str(row.get("final_position", "No Trade"))
        if side not in {"Long", "Short"}:
            return None
        prob = _probability(payload, row)
        if prob < threshold:
            return None
        return {"side": side, "confidence": prob, "threshold": threshold, "source": "primary"}

    long_row = _scenario_row(row, "Long")
    short_row = _scenario_row(row, "Short")
    long_prob = _probability(payload, long_row)
    short_prob = _probability(payload, short_row)
    side = "Long" if long_prob >= short_prob else "Short"
    prob = max(long_prob, short_prob)

    if strategy == "scenario_layak" and prob < threshold:
        return None
    if strategy not in {"scenario_layak", "best_every_hour"}:
        raise ValueError(f"Strategi tidak dikenal: {strategy}")

    return {"side": side, "confidence": prob, "threshold": threshold, "source": "scenario"}


def _exit_for_tp1_policy(
    full_df: pd.DataFrame,
    i: int,
    side: str,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    horizon_hours: int,
) -> Dict[str, object]:
    high = full_df["high"].astype(float).values
    low = full_df["low"].astype(float).values
    close = full_df["last_close"].astype(float).values

    path = _evaluate_trade_path(
        high=high,
        low=low,
        i=i,
        side=side,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        horizon_hours=horizon_hours,
    )

    first_exit = str(path["first_exit"])
    if int(path["label_win_tp1"]) == 1:
        exit_price = take_profit_1
        exit_reason = "TP1"
    elif first_exit in {"SL", "AMBIGUOUS_SL_FIRST"}:
        exit_price = stop_loss
        exit_reason = "SL"
    else:
        exit_price = float(close[i + horizon_hours])
        exit_reason = "TIMEOUT"

    return {
        **path,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
    }


def _simulate_trade(
    payload: Dict[str, object],
    full_df: pd.DataFrame,
    i: int,
    decision: Dict[str, object],
    bt_config: BacktestConfig,
) -> Dict[str, object]:
    config = _coerce_config(payload.get("config", {}))
    row = full_df.iloc[i]
    side = str(decision["side"])
    entry = float(row["last_close"])
    atr = float(row["atr"])
    stop_loss, take_profit_1, take_profit_2, risk, risk_atr_mult = _risk_levels(
        entry,
        atr,
        side,
        config,
        volatility_cluster=row.get("volatility_cluster"),
        last_tr_atr=row.get("last_tr_atr"),
    )

    exit_info = _exit_for_tp1_policy(
        full_df=full_df,
        i=i,
        side=side,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        horizon_hours=config.horizon_hours,
    )

    pnl_pct = _leveraged_pnl_pct(entry, float(exit_info["exit_price"]), side, config.leverage)
    raw_pnl_pct = pnl_pct
    liquidation_capped = False
    if bt_config.cap_loss_at_margin and pnl_pct < -100.0:
        pnl_pct = -100.0
        liquidation_capped = True

    pnl_usdt = bt_config.stake_per_trade * pnl_pct / 100.0
    return {
        "timestamp": row["timestamp"],
        "date": row["timestamp"].date().isoformat(),
        "strategy": None,
        "side": side,
        "source": decision["source"],
        "confidence": float(decision["confidence"]),
        "threshold": float(decision["threshold"]),
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk": risk,
        "risk_atr_mult": risk_atr_mult,
        "first_exit": exit_info["first_exit"],
        "exit_reason": exit_info["exit_reason"],
        "exit_price": exit_info["exit_price"],
        "bars_to_event": exit_info["bars_to_event"],
        "hit_tp1": int(exit_info["label_win_tp1"]),
        "hit_tp2": int(exit_info["label_win_tp2"]),
        "raw_pnl_pct_x75": raw_pnl_pct,
        "pnl_pct_x75": pnl_pct,
        "stake_usdt": bt_config.stake_per_trade,
        "pnl_usdt": pnl_usdt,
        "liquidation_capped": liquidation_capped,
        "score": row.get("score"),
        "trend": row.get("trend"),
        "final_position": row.get("final_position"),
        "volatility_cluster": row.get("volatility_cluster"),
    }


def run_backtest(
    model_path: str | Path,
    dataset_csv: str | Path,
    start: str,
    end: str,
    bt_config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = load_trade_model(model_path)
    model_config = _coerce_config(payload.get("config", {}))
    threshold = float(payload.get("summary", {}).get("threshold", 0.60))
    df = _load_dataset(dataset_csv)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    max_i = len(df) - model_config.horizon_hours - 1
    detail_rows: List[Dict[str, object]] = []
    setup_count_by_strategy = {strategy: 0 for strategy in STRATEGIES}
    skip_count_by_strategy = {strategy: 0 for strategy in STRATEGIES}

    for i in range(max_i + 1):
        row = df.iloc[i]
        ts = row["timestamp"]
        if ts < start_ts or ts > end_ts:
            continue
        if pd.isna(row.get("last_close")) or pd.isna(row.get("atr")) or float(row.get("atr", 0.0)) <= 0:
            continue

        for strategy in STRATEGIES:
            setup_count_by_strategy[strategy] += 1
            decision = _pick_trade(payload, row, strategy, threshold)
            if decision is None:
                skip_count_by_strategy[strategy] += 1
                continue
            trade = _simulate_trade(payload, df, i, decision, bt_config)
            trade["strategy"] = strategy
            detail_rows.append(trade)

    detail_df = pd.DataFrame(detail_rows)
    summary_df = _make_summary(detail_df, setup_count_by_strategy, skip_count_by_strategy, bt_config)
    daily_df = _make_daily(detail_df)
    return summary_df, daily_df, detail_df


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def _make_summary(
    detail_df: pd.DataFrame,
    setup_count_by_strategy: Dict[str, int],
    skip_count_by_strategy: Dict[str, int],
    bt_config: BacktestConfig,
) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        sub = detail_df[detail_df["strategy"] == strategy].copy() if not detail_df.empty else pd.DataFrame()
        trades = int(len(sub))
        wins = int((sub["pnl_usdt"] > 0).sum()) if trades else 0
        losses = int((sub["pnl_usdt"] < 0).sum()) if trades else 0
        pnl = float(sub["pnl_usdt"].sum()) if trades else 0.0
        equity = bt_config.initial_capital + sub["pnl_usdt"].cumsum() if trades else pd.Series(dtype=float)
        days = max(1, len(set(sub["date"])) if trades else 0)
        rows.append(
            {
                "strategy": strategy,
                "setups": setup_count_by_strategy.get(strategy, 0),
                "skips": skip_count_by_strategy.get(strategy, 0),
                "trades": trades,
                "avg_trades_per_day": trades / days if days else 0.0,
                "wins": wins,
                "losses": losses,
                "winrate": wins / trades if trades else 0.0,
                "gross_profit_usdt": float(sub.loc[sub["pnl_usdt"] > 0, "pnl_usdt"].sum()) if trades else 0.0,
                "gross_loss_usdt": float(sub.loc[sub["pnl_usdt"] < 0, "pnl_usdt"].sum()) if trades else 0.0,
                "net_pnl_usdt": pnl,
                "ending_capital_usdt": bt_config.initial_capital + pnl,
                "return_pct_on_initial": pnl / bt_config.initial_capital if bt_config.initial_capital else 0.0,
                "max_drawdown_usdt": _max_drawdown(equity),
                "avg_confidence": float(sub["confidence"].mean()) if trades else 0.0,
                "liquidation_capped_count": int(sub["liquidation_capped"].sum()) if trades else 0,
            }
        )
    return pd.DataFrame(rows)


def _make_daily(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    grouped = detail_df.groupby(["strategy", "date"], as_index=False).agg(
        trades=("pnl_usdt", "size"),
        wins=("pnl_usdt", lambda s: int((s > 0).sum())),
        losses=("pnl_usdt", lambda s: int((s < 0).sum())),
        pnl_usdt=("pnl_usdt", "sum"),
        avg_confidence=("confidence", "mean"),
    )
    grouped["winrate"] = grouped["wins"] / grouped["trades"]
    return grouped.sort_values(["strategy", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest trade ML strategies over a historical date range.")
    parser.add_argument("--model-path", default="trade_ml_model.pkl")
    parser.add_argument("--dataset-csv", default="trade_ml_dataset.csv")
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--stake-per-trade", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs/backtests")
    args = parser.parse_args()

    bt_config = BacktestConfig(
        initial_capital=args.initial_capital,
        stake_per_trade=args.stake_per_trade,
    )
    summary_df, daily_df, detail_df = run_backtest(
        model_path=args.model_path,
        dataset_csv=args.dataset_csv,
        start=args.start,
        end=args.end,
        bt_config=bt_config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start}_to_{args.end}_capital_{args.initial_capital:g}_stake_{args.stake_per_trade:g}".replace("/", "-")
    summary_path = output_dir / f"summary_{tag}.csv"
    daily_path = output_dir / f"daily_{tag}.csv"
    detail_path = output_dir / f"detail_{tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("SUMMARY")
    print(summary_df.to_string(index=False))
    print()
    print(f"summary_csv={summary_path}")
    print(f"daily_csv={daily_path}")
    print(f"detail_csv={detail_path}")


if __name__ == "__main__":
    main()
