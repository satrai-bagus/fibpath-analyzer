import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from market_signal import compute_market_signal
from telegram_notifier import notification_key, notify_once
from trade_ml_engine import (
    TradeModelConfig,
    TradePrediction,
    analyze_trade_for_hour,
    format_summary,
    load_trade_model,
    train_and_save_trade_model,
)


st.set_page_config(page_title="Fib Path Analyzer V2", layout="wide")

BASE_DIR = Path.cwd()
LEGACY_TRADE_MODEL_PATH = BASE_DIR / "trade_ml_model.pkl"
TRADE_MODEL_DIR = BASE_DIR / "models" / "trade_ml"
TRADE_DATASET_DIR = BASE_DIR / "outputs" / "trade_ml_datasets"
TRADE_BACKTEST_DIR = BASE_DIR / "outputs" / "trade_ml_backtests"
TELEGRAM_SENT_LOG_PATH = BASE_DIR / "outputs" / "telegram_sent_signals.json"
MEDIUM_CONFIDENCE = 0.56
STRONG_CONFIDENCE = 0.58
DEFAULT_WATCHLIST = [
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


def _config_value(name: str) -> str:
    env_value = os.getenv(name, "")
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    return str(secret_value or "")


@st.cache_resource
def load_trade_engine(model_path: Path, model_mtime: float):
    return load_trade_model(model_path)


def _fmt_price(value: float) -> str:
    return f"{value:.4f}" if pd.notna(value) else "-"


def _fmt_pct(value: float, signed: bool = False) -> str:
    if pd.isna(value):
        return "-"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.2f}%"


def _confidence_grade(probability: float) -> str:
    if probability >= STRONG_CONFIDENCE:
        return "Strong"
    if probability >= MEDIUM_CONFIDENCE:
        return "Medium"
    return "Weak / Avoid"


def _entry_decision(prediction: TradePrediction) -> str:
    if prediction.action != "No Trade" and prediction.probability_win_tp1 >= MEDIUM_CONFIDENCE:
        return prediction.action
    return "No Trade"


def _normalize_ticker_input(value: str) -> str:
    ticker = str(value or "").strip().upper().replace("/", "-")
    if not ticker:
        return ""
    if ticker.endswith("USDT") and "-" not in ticker:
        return f"{ticker[:-4]}-USD"
    if "-" not in ticker:
        return f"{ticker}-USD"
    return ticker


def _parse_watchlist(value: str) -> list[str]:
    tickers = []
    seen = set()
    normalized = value.replace(";", ",").replace("\n", ",")
    for item in normalized.split(","):
        ticker = _normalize_ticker_input(item)
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers[:20]


def _ticker_slug(ticker: str) -> str:
    normalized = _normalize_ticker_input(ticker)
    slug = "".join(ch if ch.isalnum() else "_" for ch in normalized).strip("_")
    return slug or "UNKNOWN"


def _trade_model_path(ticker: str) -> Path:
    return TRADE_MODEL_DIR / f"{_ticker_slug(ticker)}.pkl"


def _trade_dataset_path(ticker: str) -> Path:
    return TRADE_DATASET_DIR / f"{_ticker_slug(ticker)}_dataset.csv"


def _trade_backtest_path(ticker: str) -> Path:
    return TRADE_BACKTEST_DIR / f"{_ticker_slug(ticker)}_backtest.csv"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _enable_auto_refresh(interval_seconds: int) -> None:
    interval_ms = max(int(interval_seconds), 30) * 1000
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {interval_ms});
        </script>
        """,
        height=0,
    )


def _fmt_timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(value)


def _telegram_alert_plan(
    prediction: TradePrediction,
    include_below_threshold: bool,
) -> tuple[bool, str, str, str]:
    decision = _entry_decision(prediction)
    if decision != "No Trade":
        return True, decision, decision, "FibPath Primary Layak"

    if (
        include_below_threshold
        and prediction.side in {"Long", "Short"}
        and pd.notna(prediction.probability_win_tp1)
    ):
        return (
            True,
            f"Watch {prediction.side} (No Entry)",
            f"OBS_{prediction.side}",
            "FibPath Primary Observasi (Below Threshold)",
        )

    return False, "No Trade", "NO_TRADE", ""


def _build_primary_alert_message(
    ticker: str,
    prediction: TradePrediction,
    decision: str,
    title: str = "FibPath Primary Layak",
) -> str:
    grade = _confidence_grade(prediction.probability_win_tp1)
    return "\n".join(
        [
            title,
            f"Ticker: {ticker.upper()}",
            f"Candle: {_fmt_timestamp(prediction.timestamp)}",
            f"Action: {decision}",
            f"Grade: {grade}",
            f"Confidence: {prediction.probability_win_tp1:.2%}",
            f"Entry Threshold: {MEDIUM_CONFIDENCE:.2%}",
            f"Entry: {_fmt_price(prediction.entry)}",
            f"Stop Loss: {_fmt_price(prediction.stop_loss)}",
            f"TP1: {_fmt_price(prediction.take_profit_1)}",
            f"TP2: {_fmt_price(prediction.take_profit_2)}",
            f"Risk ATR: {prediction.risk_atr_mult:.2f}x" if pd.notna(prediction.risk_atr_mult) else "Risk ATR: -",
            f"SL PnL x{prediction.leverage:.0f}: {_fmt_pct(prediction.stop_loss_pnl_pct_leveraged)}",
            f"TP1 PnL x{prediction.leverage:.0f}: {_fmt_pct(prediction.take_profit_1_pnl_pct_leveraged, signed=True)}",
            f"TP2 PnL x{prediction.leverage:.0f}: {_fmt_pct(prediction.take_profit_2_pnl_pct_leveraged, signed=True)}",
            f"Reason: {prediction.reason}",
        ]
    )


def _maybe_send_primary_alert(
    ticker: str,
    prediction: TradePrediction,
    telegram_enabled: bool,
    bot_token: str,
    chat_id: str,
    include_below_threshold: bool = False,
) -> None:
    should_send, decision, key_action, title = _telegram_alert_plan(prediction, include_below_threshold)
    if not should_send:
        return

    if not telegram_enabled:
        st.caption("Telegram alert mati. Alert tidak dikirim ke Telegram.")
        return

    if not bot_token.strip() or not chat_id.strip():
        st.warning("Alert muncul, tapi Telegram token/chat ID belum diisi.")
        return

    key = notification_key(ticker, prediction.timestamp, key_action)
    message = _build_primary_alert_message(ticker, prediction, decision, title=title)
    sent, detail = notify_once(
        bot_token=bot_token,
        chat_id=chat_id,
        key=key,
        message=message,
        sent_log_path=TELEGRAM_SENT_LOG_PATH,
    )
    if sent:
        st.success(detail)
    else:
        st.caption(detail)


def _scenario_rows(scenarios: dict[str, TradePrediction]) -> list[dict[str, object]]:
    rows = []
    for side, pred in scenarios.items():
        rows.append(
            {
                "Side": side,
                "Mode": "Observasi Only",
                "Grade": _confidence_grade(pred.probability_win_tp1),
                "Confidence": f"{pred.probability_win_tp1:.2%}",
                "Entry": _fmt_price(pred.entry),
                "SL": _fmt_price(pred.stop_loss),
                "TP1": _fmt_price(pred.take_profit_1),
                "TP2": _fmt_price(pred.take_profit_2),
                "Risk ATR": f"{pred.risk_atr_mult:.2f}x" if pd.notna(pred.risk_atr_mult) else "-",
                "SL x75": _fmt_pct(pred.stop_loss_pnl_pct_leveraged),
                "TP1 x75": _fmt_pct(pred.take_profit_1_pnl_pct_leveraged, signed=True),
                "TP2 x75": _fmt_pct(pred.take_profit_2_pnl_pct_leveraged, signed=True),
            }
        )
    return rows


def _render_market_result(market_result: dict) -> None:
    st.subheader("Hasil Auto-Compute dari Market Data")
    st.caption(f"Candle dianalisis UTC: {market_result.get('analyzed_bar_time_utc', '-')}")

    row1 = st.columns(5)
    row1[0].metric("Trend", market_result.get("trend", "-"))
    row1[1].metric("Squeeze Mom", market_result.get("squeeze_momentum", "-"))
    row1[2].metric("Squeeze Mom 2", market_result.get("squeeze_momentum2", "-"))
    row1[3].metric("Score", f"{market_result.get('score', 0)}")
    row1[4].metric("Last TR", f"{market_result.get('last_tr', 0.0):.4f}")

    row2 = st.columns(2)
    row2[0].metric("Raw Position", market_result.get("raw_position", "-"))
    row2[1].metric("Final Position", market_result.get("final_position", "-"))

    with st.expander("Detail Indikator Market", expanded=False):
        detail_cols = st.columns(4)
        detail_cols[0].metric("Last Close", f"{market_result.get('last_close', 0):.2f}")
        detail_cols[1].metric("RSI", f"{market_result.get('rsi_last', 0):.2f}")
        detail_cols[2].metric("ADX", f"{market_result.get('adx_last', 0):.2f}")
        detail_cols[3].metric("ATR", f"{market_result.get('atr_last', 0):.4f}")

        detail_cols2 = st.columns(4)
        detail_cols2[0].metric("EMA Fast", f"{market_result.get('ema_fast_last', 0):.2f}")
        detail_cols2[1].metric("EMA Slow", f"{market_result.get('ema_slow_last', 0):.2f}")
        detail_cols2[2].metric("MACD", f"{market_result.get('macd_last', 0):.4f}")
        detail_cols2[3].metric("Filter", market_result.get("filter_reason", "-"))


def _render_trade_prediction(title: str, prediction: TradePrediction, note: str = "") -> None:
    st.subheader(title)
    if note:
        st.caption(note)

    plan_cols = st.columns(7)
    plan_cols[0].metric("Entry Decision", _entry_decision(prediction))
    plan_cols[1].metric("Grade", _confidence_grade(prediction.probability_win_tp1))
    plan_cols[2].metric("Prob TP1 before SL", f"{prediction.probability_win_tp1:.2%}")
    plan_cols[3].metric("Entry", _fmt_price(prediction.entry))
    plan_cols[4].metric("Stop Loss", _fmt_price(prediction.stop_loss))
    plan_cols[5].metric("Take Profit 1", _fmt_price(prediction.take_profit_1))
    plan_cols[6].metric("Take Profit 2", _fmt_price(prediction.take_profit_2))

    pnl_cols = st.columns(4)
    pnl_cols[0].metric("Risk ATR Mult", f"{prediction.risk_atr_mult:.2f}x" if pd.notna(prediction.risk_atr_mult) else "-")
    pnl_cols[1].metric("SL PnL x75", _fmt_pct(prediction.stop_loss_pnl_pct_leveraged))
    pnl_cols[2].metric("TP1 PnL x75", _fmt_pct(prediction.take_profit_1_pnl_pct_leveraged, signed=True))
    pnl_cols[3].metric("TP2 PnL x75", _fmt_pct(prediction.take_profit_2_pnl_pct_leveraged, signed=True))

    st.caption(
        f"Model test 2 hari: winrate {prediction.model_winrate_2d:.2%} "
        f"dari {prediction.model_trades_2d} trade | "
        f"threshold {prediction.confidence_threshold:.2f} | "
        f"leverage {prediction.leverage:.0f}x sebelum fee/funding | "
        f"candle ML UTC {prediction.timestamp} | {prediction.reason}"
    )


def _render_scenarios(scenarios: dict[str, TradePrediction]) -> None:
    st.subheader("Skenario Dua Arah")
    st.caption("Ini mode what-if/observasi. Entry default tetap hanya memakai Primary Layak.")
    st.dataframe(pd.DataFrame(_scenario_rows(scenarios)), use_container_width=True, hide_index=True)


def _send_primary_alert_detail(
    ticker: str,
    prediction: TradePrediction,
    telegram_enabled: bool,
    bot_token: str,
    chat_id: str,
    include_below_threshold: bool = False,
) -> str:
    should_send, decision, key_action, title = _telegram_alert_plan(prediction, include_below_threshold)
    if not should_send:
        return "-"
    if not telegram_enabled:
        return "Telegram off"
    if not bot_token.strip() or not chat_id.strip():
        return "Token/chat ID belum diisi"

    key = notification_key(ticker, prediction.timestamp, key_action)
    message = _build_primary_alert_message(ticker, prediction, decision, title=title)
    sent, detail = notify_once(
        bot_token=bot_token,
        chat_id=chat_id,
        key=key,
        message=message,
        sent_log_path=TELEGRAM_SENT_LOG_PATH,
    )
    return detail if sent else detail


def _watchlist_row(
    ticker: str,
    target_dt: datetime,
    telegram_enabled: bool,
    telegram_bot_token: str,
    telegram_chat_id: str,
    send_telegram_alert: bool,
    send_below_threshold_alerts: bool,
) -> dict[str, object]:
    model_path = _trade_model_path(ticker)
    base_row = {
        "Ticker": ticker,
        "Model": "Ready" if model_path.exists() else "Missing",
        "Decision": "No Trade",
        "Grade": "-",
        "Confidence": "-",
        "Entry": "-",
        "SL": "-",
        "TP1": "-",
        "TP2": "-",
        "Trend": "-",
        "Score": "-",
        "Final Position": "-",
        "Candle UTC": "-",
        "Alert Type": "-",
        "Telegram": "-",
        "Status": "OK",
        "_confidence_raw": 0.0,
    }

    market_result = compute_market_signal(
        ticker=ticker,
        target_date=target_dt,
        target_hour=target_dt.hour,
    )
    if market_result.get("error"):
        base_row["Status"] = market_result["error"]
        return base_row

    base_row.update(
        {
            "Trend": market_result.get("trend", "-"),
            "Score": market_result.get("score", "-"),
            "Final Position": market_result.get("final_position", "-"),
            "Candle UTC": market_result.get("analyzed_bar_time_utc", "-"),
        }
    )

    if not model_path.exists():
        base_row["Status"] = f"Model {ticker} belum dilatih"
        return base_row

    try:
        primary, _ = analyze_trade_for_hour(
            ticker=ticker,
            target_date=target_dt,
            target_hour=target_dt.hour,
            model_path=model_path,
        )
    except Exception as e:
        base_row["Status"] = str(e)
        return base_row

    decision = _entry_decision(primary)
    _, _, _, alert_title = _telegram_alert_plan(primary, send_below_threshold_alerts)
    base_row.update(
        {
            "Decision": decision,
            "Grade": _confidence_grade(primary.probability_win_tp1),
            "Confidence": f"{primary.probability_win_tp1:.2%}",
            "Entry": _fmt_price(primary.entry),
            "SL": _fmt_price(primary.stop_loss),
            "TP1": _fmt_price(primary.take_profit_1),
            "TP2": _fmt_price(primary.take_profit_2),
            "Candle UTC": _fmt_timestamp(primary.timestamp),
            "Alert Type": alert_title.replace("FibPath Primary ", "") if alert_title else "-",
            "Status": primary.reason,
            "_confidence_raw": primary.probability_win_tp1,
        }
    )

    if send_telegram_alert and (decision != "No Trade" or send_below_threshold_alerts):
        base_row["Telegram"] = _send_primary_alert_detail(
            ticker=ticker,
            prediction=primary,
            telegram_enabled=telegram_enabled,
            bot_token=telegram_bot_token,
            chat_id=telegram_chat_id,
            include_below_threshold=send_below_threshold_alerts,
        )

    return base_row


def render_watchlist_scan(
    tickers: list[str],
    target_dt: datetime,
    telegram_enabled: bool,
    telegram_bot_token: str,
    telegram_chat_id: str,
    send_telegram_alert: bool,
    send_below_threshold_alerts: bool,
) -> None:
    st.divider()
    st.header("Scan Watchlist Large Cap")
    st.caption(f"Scan {len(tickers)} ticker pada target UTC {target_dt.strftime('%Y-%m-%d %H:%M')}. Telegram hanya dikirim untuk Primary Layak.")

    if not tickers:
        st.warning("Watchlist kosong.")
        return

    progress = st.progress(0, text="Mulai scan watchlist...")
    rows = []
    for idx, ticker in enumerate(tickers, start=1):
        progress.progress(idx / len(tickers), text=f"Scan {ticker} ({idx}/{len(tickers)})")
        rows.append(
            _watchlist_row(
                ticker=ticker,
                target_dt=target_dt,
                telegram_enabled=telegram_enabled,
                telegram_bot_token=telegram_bot_token,
                telegram_chat_id=telegram_chat_id,
                send_telegram_alert=send_telegram_alert,
                send_below_threshold_alerts=send_below_threshold_alerts,
            )
        )
    progress.empty()

    df = pd.DataFrame(rows).sort_values("_confidence_raw", ascending=False)
    show_df = df.drop(columns=["_confidence_raw"], errors="ignore")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    signals = show_df[show_df["Decision"] != "No Trade"]
    if signals.empty:
        st.info("Belum ada Primary Layak dari watchlist saat ini.")
    else:
        st.success(f"Ada {len(signals)} Primary Layak dari watchlist.")


def run_analysis(
    ticker: str,
    target_dt: datetime,
    title: str,
    telegram_enabled: bool = False,
    telegram_bot_token: str = "",
    telegram_chat_id: str = "",
    send_telegram_alert: bool = False,
    send_below_threshold_alerts: bool = False,
) -> None:
    ticker = _normalize_ticker_input(ticker)
    model_path = _trade_model_path(ticker)
    st.divider()
    st.header(title)
    st.caption(f"Target waktu UTC: {target_dt.strftime('%Y-%m-%d %H:%M')}")

    with st.spinner("Mengambil data market dan menghitung indikator..."):
        market_result = compute_market_signal(
            ticker=ticker,
            target_date=target_dt,
            target_hour=target_dt.hour,
        )

    if market_result.get("error"):
        st.error(f"Market data error: {market_result['error']}")
        return

    _render_market_result(market_result)

    if not model_path.exists():
        st.info(f"Train Trade ML untuk {ticker} dari sidebar. Model yang dicari: {_relative_path(model_path)}")
        return

    try:
        with st.spinner("Menghitung trade plan ML..."):
            primary, scenarios = analyze_trade_for_hour(
                ticker=ticker,
                target_date=target_dt,
                target_hour=target_dt.hour,
                model_path=model_path,
            )
    except Exception as e:
        st.warning(f"Trade ML belum bisa membuat plan: {e}")
        return

    _render_trade_prediction(
        "Primary Layak",
        primary,
        note=f"Default entry hanya jika primary bukan No Trade dan confidence minimal {MEDIUM_CONFIDENCE:.0%}. Strong mulai {STRONG_CONFIDENCE:.0%}.",
    )

    if _entry_decision(primary) != "No Trade":
        st.success("Primary memenuhi syarat entry.")
        if send_telegram_alert:
            _maybe_send_primary_alert(
                ticker=ticker,
                prediction=primary,
                telegram_enabled=telegram_enabled,
                bot_token=telegram_bot_token,
                chat_id=telegram_chat_id,
                include_below_threshold=send_below_threshold_alerts,
            )
    else:
        st.info(f"Tidak ada entry primary saat ini. {primary.reason}")
        if send_telegram_alert and send_below_threshold_alerts:
            _maybe_send_primary_alert(
                ticker=ticker,
                prediction=primary,
                telegram_enabled=telegram_enabled,
                bot_token=telegram_bot_token,
                chat_id=telegram_chat_id,
                include_below_threshold=True,
            )

    _render_scenarios(scenarios)


st.title("Fib Path Analyzer Dashboard V2")
st.markdown("Begitu halaman dibuka, dashboard langsung menganalisis candle terbaru yang sudah close.")
st.caption("Jam trade memakai UTC sebagai waktu evaluasi. Contoh: jam 00:00 menganalisis candle 23:00 yang sudah close, jam 01:00 menganalisis candle 00:00.")


with st.sidebar:
    st.header("Trade ML SL/TP")
    trade_train_ticker_input = st.text_input("Ticker Training ML", value="ETH-USD")
    trade_train_ticker = _normalize_ticker_input(trade_train_ticker_input)
    trade_lookback_days = st.number_input(
        "Lookback Dataset (hari)",
        min_value=30,
        max_value=729,
        value=729,
        step=30,
    )
    trade_horizon_hours = st.number_input(
        "Horizon Label TP/SL (jam)",
        min_value=4,
        max_value=72,
        value=24,
        step=4,
    )
    st.caption("Default risk adaptif: SL dasar = 1.4 ATR, TP1 = 0.8R, TP2 = 1.6R, leverage display = 75x.")

    selected_model_path = _trade_model_path(trade_train_ticker)
    st.caption(f"Model aktif {trade_train_ticker}: {_relative_path(selected_model_path)}")
    if selected_model_path.exists():
        try:
            trade_payload_preview = load_trade_engine(selected_model_path, selected_model_path.stat().st_mtime)
            st.success(f"Model {trade_train_ticker} ditemukan")
            st.caption(format_summary(trade_payload_preview.get("summary", {})))
        except Exception as e:
            st.warning(f"Trade ML model gagal dibaca: {e}")
    else:
        st.info(f"Model {trade_train_ticker} belum ada")
        if LEGACY_TRADE_MODEL_PATH.exists():
            st.caption("Catatan: model global lama masih ada, tapi scan sekarang memakai model per ticker.")

    if st.button(f"Train Model {trade_train_ticker}"):
        with st.spinner(f"Mengambil data, membuat dataset, training, dan backtest {trade_train_ticker}..."):
            trade_config = TradeModelConfig(
                lookback_days=int(trade_lookback_days),
                test_days=2,
                horizon_hours=int(trade_horizon_hours),
            )
            trade_payload = train_and_save_trade_model(
                ticker=trade_train_ticker,
                model_path=selected_model_path,
                dataset_csv=_trade_dataset_path(trade_train_ticker),
                backtest_csv=_trade_backtest_path(trade_train_ticker),
                config=trade_config,
            )
        load_trade_engine.clear()
        st.success(f"Model {trade_train_ticker} selesai dilatih")
        st.caption(format_summary(trade_payload.get("summary", {})))

    st.divider()
    st.header("Auto Update")
    auto_refresh_enabled = st.toggle("Auto refresh dashboard", value=True)
    auto_refresh_seconds = st.number_input(
        "Interval refresh (detik)",
        min_value=60,
        max_value=3600,
        value=300,
        step=60,
    )
    st.caption("Dashboard akan cek candle terbaru secara berkala saat tab Streamlit terbuka.")

    scan_watchlist_enabled = st.toggle("Scan watchlist large-cap", value=True)
    watchlist_text = st.text_area(
        "Watchlist ticker",
        value="\n".join(DEFAULT_WATCHLIST),
        height=180,
        help="Bisa tulis BTC, BTCUSDT, BTC-USD, dipisah koma atau baris baru.",
        disabled=not scan_watchlist_enabled,
    )
    watchlist_tickers = _parse_watchlist(watchlist_text) if scan_watchlist_enabled else []
    st.caption(f"{len(watchlist_tickers)} ticker aktif. Batas scan dashboard: 20 ticker.")
    ready_models = [ticker for ticker in watchlist_tickers if _trade_model_path(ticker).exists()]
    missing_models = [ticker for ticker in watchlist_tickers if not _trade_model_path(ticker).exists()]
    st.caption(f"Model per ticker siap: {len(ready_models)}/{len(watchlist_tickers)}")

    retrain_watchlist_models = st.checkbox("Retrain model yang sudah ada", value=False)
    if st.button("Train Watchlist Models", disabled=not watchlist_tickers):
        train_targets = watchlist_tickers if retrain_watchlist_models else missing_models
        if not train_targets:
            st.success("Semua model watchlist sudah ada.")
        else:
            trade_config = TradeModelConfig(
                lookback_days=int(trade_lookback_days),
                test_days=2,
                horizon_hours=int(trade_horizon_hours),
            )
            progress = st.progress(0, text="Mulai training watchlist...")
            train_rows = []
            for idx, train_ticker in enumerate(train_targets, start=1):
                progress.progress(idx / len(train_targets), text=f"Training {train_ticker} ({idx}/{len(train_targets)})")
                try:
                    payload = train_and_save_trade_model(
                        ticker=train_ticker,
                        model_path=_trade_model_path(train_ticker),
                        dataset_csv=_trade_dataset_path(train_ticker),
                        backtest_csv=_trade_backtest_path(train_ticker),
                        config=trade_config,
                    )
                    summary = payload.get("summary", {})
                    train_rows.append(
                        {
                            "Ticker": train_ticker,
                            "Status": "OK",
                            "Winrate 2D": f"{summary.get('test_winrate', 0.0):.2%}",
                            "Trades": int(summary.get("test_trades", 0)),
                            "Threshold": f"{summary.get('threshold', 0.0):.2f}",
                        }
                    )
                except Exception as e:
                    train_rows.append(
                        {
                            "Ticker": train_ticker,
                            "Status": str(e),
                            "Winrate 2D": "-",
                            "Trades": "-",
                            "Threshold": "-",
                        }
                    )
            progress.empty()
            load_trade_engine.clear()
            st.success(f"Training watchlist selesai: {len(train_targets)} ticker diproses")
            st.dataframe(pd.DataFrame(train_rows), use_container_width=True, hide_index=True)

    st.header("Telegram Alert")
    env_bot_token = _config_value("TELEGRAM_BOT_TOKEN")
    env_chat_id = _config_value("TELEGRAM_CHAT_ID")
    telegram_enabled = st.toggle(
        "Kirim Primary Layak",
        value=bool(env_bot_token and env_chat_id),
    )
    telegram_bot_token = st.text_input(
        "Bot Token",
        value=env_bot_token,
        type="password",
        disabled=not telegram_enabled,
    )
    telegram_chat_id = st.text_input(
        "Chat ID",
        value=env_chat_id,
        disabled=not telegram_enabled,
    )
    telegram_below_threshold = st.toggle(
        "Kirim juga observasi below threshold",
        value=True,
        disabled=not telegram_enabled,
        help="Mengirim Watch Long/Short walaupun belum layak entry. Labelnya observasi, bukan sinyal entry.",
    )
    st.caption("Alert dikirim sekali per ticker + candle + arah. Observasi below threshold bisa ramai saat watchlist aktif.")


if auto_refresh_enabled:
    _enable_auto_refresh(int(auto_refresh_seconds))


ticker_val = st.text_input("Ticker Detail", value="ETH-USD", help="Contoh: ETH-USD, BTC-USD, SOL-USD")
auto_now = st.toggle("Analisis otomatis candle terbaru saat halaman dibuka", value=True)
show_detail_now = st.toggle(
    "Tampilkan detail ticker pilihan",
    value=not scan_watchlist_enabled,
    help="Matikan jika ingin dashboard fokus ke tabel watchlist saja.",
)

if auto_now:
    now_utc = datetime.now(timezone.utc)
    if scan_watchlist_enabled:
        render_watchlist_scan(
            tickers=watchlist_tickers,
            target_dt=now_utc,
            telegram_enabled=telegram_enabled,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            send_telegram_alert=True,
            send_below_threshold_alerts=telegram_below_threshold,
        )

    if show_detail_now:
        run_analysis(
            ticker_val,
            now_utc,
            "Analisis Sekarang",
            telegram_enabled=telegram_enabled,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            send_telegram_alert=not scan_watchlist_enabled,
            send_below_threshold_alerts=telegram_below_threshold,
        )


with st.expander("Analisis jam tertentu", expanded=False):
    with st.form("manual_input_form"):
        m1, m2 = st.columns(2)
        with m1:
            date_val = st.date_input("Tanggal Analisis", value=datetime.now(timezone.utc).date())
        with m2:
            hour_val = st.slider("Jam Trade (UTC)", min_value=0, max_value=23, value=datetime.now(timezone.utc).hour)

        submitted = st.form_submit_button("Jalankan Analisis Jam Ini", use_container_width=True)

    if submitted:
        manual_dt = datetime.combine(date_val, datetime.min.time(), tzinfo=timezone.utc).replace(hour=int(hour_val))
        run_analysis(ticker_val, manual_dt, "Analisis Jam Tertentu")
