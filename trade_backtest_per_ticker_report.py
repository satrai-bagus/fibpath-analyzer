from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from trade_backtest_report import (
    BacktestConfig,
    STRATEGIES,
    _make_daily,
    _make_summary,
    _scenario_row,
    _simulate_trade,
)
from trade_ml_engine import MODEL_FEATURES, _coerce_config, _positive_probability, load_trade_model


DEFAULT_TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "SOL-USD",
    "DOGE-USD",
    "ADA-USD",
    "TRX-USD",
    "AVAX-USD",
    "LINK-USD",
]


def normalize_ticker(value: str) -> str:
    ticker = str(value or "").strip().upper().replace("/", "-")
    if not ticker:
        return ""
    if ticker.endswith("USDT") and "-" not in ticker:
        return f"{ticker[:-4]}-USD"
    if "-" not in ticker:
        return f"{ticker}-USD"
    return ticker


def ticker_slug(ticker: str) -> str:
    normalized = normalize_ticker(ticker)
    return "".join(ch if ch.isalnum() else "_" for ch in normalized).strip("_")


def parse_tickers(value: str) -> list[str]:
    if not value:
        return DEFAULT_TICKERS
    tickers = []
    seen = set()
    for item in value.replace(";", ",").replace("\n", ",").split(","):
        ticker = normalize_ticker(item)
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


def model_path_for(ticker: str, model_dir: Path) -> Path:
    return model_dir / f"{ticker_slug(ticker)}.pkl"


def dataset_path_for(ticker: str, dataset_dir: Path) -> Path:
    path = dataset_dir / f"{ticker_slug(ticker)}_dataset.csv"
    if path.exists():
        return path
    if normalize_ticker(ticker) == "ETH-USD" and Path("trade_ml_dataset.csv").exists():
        return Path("trade_ml_dataset.csv")
    return path


def month_periods(start: str, end: str) -> list[tuple[str, str, str]]:
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()
    periods = []
    cursor = start_ts.replace(day=1)
    while cursor <= end_ts:
        month_start = max(cursor, start_ts)
        month_end = min(cursor + pd.offsets.MonthEnd(0), end_ts)
        label = month_start.strftime("%Y-%m")
        if month_end != cursor + pd.offsets.MonthEnd(0):
            label = f"{label}_to_{month_end.strftime('%Y-%m-%d')}"
        periods.append((label, month_start.date().isoformat(), month_end.date().isoformat()))
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return periods


def requested_periods(start: str, end: str, include_monthly: bool) -> list[tuple[str, str, str]]:
    periods = [(f"combined_{start}_to_{end}", start, end)]
    if include_monthly:
        periods.extend(month_periods(start, end))
    return periods


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    peak = values.cummax()
    return float((values - peak).min())


def aggregate_summary(summary_df: pd.DataFrame, detail_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = summary_df.groupby(["period", "strategy"], as_index=False)
    for _, sub in grouped:
        period = sub["period"].iloc[0]
        strategy = sub["strategy"].iloc[0]
        tickers = int(sub["ticker"].nunique())
        initial_total = initial_capital * tickers
        net_pnl = float(sub["net_pnl_usdt"].sum())
        trades = int(sub["trades"].sum())
        wins = int(sub["wins"].sum())
        losses = int(sub["losses"].sum())
        setups = int(sub["setups"].sum())
        skips = int(sub["skips"].sum())
        days = max(1, int(sub["period_days"].max()))

        detail_sub = detail_df[(detail_df["period"] == period) & (detail_df["strategy"] == strategy)].copy()
        if detail_sub.empty:
            max_dd = 0.0
            avg_conf = 0.0
        else:
            detail_sub = detail_sub.sort_values(["timestamp", "ticker"])
            equity = initial_total + detail_sub["pnl_usdt"].cumsum()
            max_dd = max_drawdown(equity)
            avg_conf = float(detail_sub["confidence"].mean())

        rows.append(
            {
                "period": period,
                "strategy": strategy,
                "tickers": tickers,
                "initial_capital_total_usdt": initial_total,
                "setups": setups,
                "skips": skips,
                "trades": trades,
                "avg_trades_per_day": trades / days,
                "wins": wins,
                "losses": losses,
                "winrate": wins / trades if trades else 0.0,
                "gross_profit_usdt": float(sub["gross_profit_usdt"].sum()),
                "gross_loss_usdt": float(sub["gross_loss_usdt"].sum()),
                "net_pnl_usdt": net_pnl,
                "ending_capital_total_usdt": initial_total + net_pnl,
                "return_pct_on_initial": net_pnl / initial_total if initial_total else 0.0,
                "max_drawdown_usdt": max_dd,
                "avg_confidence": avg_conf,
                "liquidation_capped_count": int(sub["liquidation_capped_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset harus punya kolom timestamp.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _probabilities(payload: dict[str, object], rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    model = payload["model"]
    probs = _positive_probability(model, rows[MODEL_FEATURES])
    return pd.Series(probs, index=rows.index, dtype=float)


def _scenario_rows(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    scenario = rows.copy()
    score_magnitude = scenario["score"].astype(float).abs().clip(lower=1.0)
    scenario["raw_position"] = side
    scenario["final_position"] = side
    scenario["score"] = score_magnitude if side == "Long" else -score_magnitude
    return scenario


def run_backtest_fast(
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

    setup_count_by_strategy = {strategy: 0 for strategy in STRATEGIES}
    skip_count_by_strategy = {strategy: 0 for strategy in STRATEGIES}
    detail_rows = []

    if max_i < 0:
        empty = pd.DataFrame()
        return _make_summary(empty, setup_count_by_strategy, skip_count_by_strategy, bt_config), empty, empty

    candidate = df.iloc[: max_i + 1].copy()
    in_period = (candidate["timestamp"] >= start_ts) & (candidate["timestamp"] <= end_ts)
    valid = (
        in_period
        & candidate["last_close"].notna()
        & candidate["atr"].notna()
        & (candidate["atr"].astype(float) > 0)
    )
    valid_rows = candidate.loc[valid].copy()

    for strategy in STRATEGIES:
        setup_count_by_strategy[strategy] = int(len(valid_rows))

    if valid_rows.empty:
        empty = pd.DataFrame()
        summary = _make_summary(empty, setup_count_by_strategy, setup_count_by_strategy.copy(), bt_config)
        return summary, empty, empty

    primary_mask = valid_rows["final_position"].isin(["Long", "Short"])
    primary_rows = valid_rows.loc[primary_mask].copy()
    primary_probs = _probabilities(payload, primary_rows)
    for i, prob in primary_probs[primary_probs >= threshold].items():
        trade = _simulate_trade(
            payload,
            df,
            int(i),
            {
                "side": str(df.loc[i, "final_position"]),
                "confidence": float(prob),
                "threshold": threshold,
                "source": "primary",
            },
            bt_config,
        )
        trade["strategy"] = "primary_layak"
        detail_rows.append(trade)

    long_rows = _scenario_rows(valid_rows, "Long")
    short_rows = _scenario_rows(valid_rows, "Short")
    long_probs = _probabilities(payload, long_rows)
    short_probs = _probabilities(payload, short_rows)
    best_side = pd.Series("Long", index=valid_rows.index)
    best_side.loc[short_probs > long_probs] = "Short"
    best_prob = pd.concat([long_probs, short_probs], axis=1).max(axis=1)

    for strategy, mask in [
        ("scenario_layak", best_prob >= threshold),
        ("best_every_hour", pd.Series(True, index=valid_rows.index)),
    ]:
        for i, prob in best_prob.loc[mask].items():
            trade = _simulate_trade(
                payload,
                df,
                int(i),
                {
                    "side": str(best_side.loc[i]),
                    "confidence": float(prob),
                    "threshold": threshold,
                    "source": "scenario",
                },
                bt_config,
            )
            trade["strategy"] = strategy
            detail_rows.append(trade)

    detail_df = pd.DataFrame(detail_rows)
    trade_counts = detail_df["strategy"].value_counts().to_dict() if not detail_df.empty else {}
    for strategy in STRATEGIES:
        skip_count_by_strategy[strategy] = setup_count_by_strategy[strategy] - int(trade_counts.get(strategy, 0))

    summary_df = _make_summary(detail_df, setup_count_by_strategy, skip_count_by_strategy, bt_config)
    daily_df = _make_daily(detail_df)
    return summary_df, daily_df, detail_df


def run_per_ticker_backtests(
    tickers: Iterable[str],
    periods: list[tuple[str, str, str]],
    model_dir: Path,
    dataset_dir: Path,
    bt_config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    daily_frames = []
    detail_frames = []
    missing_rows = []

    for ticker in tickers:
        model_path = model_path_for(ticker, model_dir)
        dataset_path = dataset_path_for(ticker, dataset_dir)
        if not model_path.exists() or not dataset_path.exists():
            missing_rows.append(
                {
                    "ticker": ticker,
                    "model_path": str(model_path),
                    "dataset_path": str(dataset_path),
                    "model_exists": model_path.exists(),
                    "dataset_exists": dataset_path.exists(),
                }
            )
            continue

        for period_label, start, end in periods:
            summary_df, daily_df, detail_df = run_backtest_fast(
                model_path=model_path,
                dataset_csv=dataset_path,
                start=start,
                end=end,
                bt_config=bt_config,
            )
            period_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1

            summary_df.insert(0, "period", period_label)
            summary_df.insert(1, "ticker", ticker)
            summary_df["period_start"] = start
            summary_df["period_end"] = end
            summary_df["period_days"] = period_days
            summary_df["model_path"] = str(model_path)
            summary_frames.append(summary_df)

            if not daily_df.empty:
                daily_df.insert(0, "period", period_label)
                daily_df.insert(1, "ticker", ticker)
                daily_frames.append(daily_df)

            if not detail_df.empty:
                detail_df.insert(0, "period", period_label)
                detail_df.insert(1, "ticker", ticker)
                detail_frames.append(detail_df)

    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    missing_df = pd.DataFrame(missing_rows)
    return summary_all, daily_all, detail_all, missing_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest per-ticker Trade ML models over one or more periods.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2026-02-01")
    parser.add_argument("--end", default="2026-04-28")
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--stake-per-trade", type=float, default=10.0)
    parser.add_argument("--model-dir", default="models/trade_ml")
    parser.add_argument("--dataset-dir", default="outputs/trade_ml_datasets")
    parser.add_argument("--output-dir", default="outputs/backtests/per_ticker")
    parser.add_argument("--monthly", action="store_true")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)
    periods = requested_periods(args.start, args.end, include_monthly=args.monthly)
    bt_config = BacktestConfig(initial_capital=args.initial_capital, stake_per_trade=args.stake_per_trade)

    summary_df, daily_df, detail_df, missing_df = run_per_ticker_backtests(
        tickers=tickers,
        periods=periods,
        model_dir=Path(args.model_dir),
        dataset_dir=Path(args.dataset_dir),
        bt_config=bt_config,
    )
    aggregate_df = aggregate_summary(summary_df, detail_df, args.initial_capital)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.start}_to_{args.end}_capital_{args.initial_capital:g}_stake_{args.stake_per_trade:g}".replace("/", "-")
    summary_path = output_dir / f"summary_by_ticker_{tag}.csv"
    aggregate_path = output_dir / f"summary_aggregate_{tag}.csv"
    daily_path = output_dir / f"daily_by_ticker_{tag}.csv"
    detail_path = output_dir / f"detail_by_ticker_{tag}.csv"
    missing_path = output_dir / f"missing_inputs_{tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    print("AGGREGATE SUMMARY")
    print(aggregate_df.to_string(index=False))
    print()
    print(f"summary_by_ticker_csv={summary_path}")
    print(f"summary_aggregate_csv={aggregate_path}")
    print(f"daily_by_ticker_csv={daily_path}")
    print(f"detail_by_ticker_csv={detail_path}")
    print(f"missing_inputs_csv={missing_path}")


if __name__ == "__main__":
    main()
