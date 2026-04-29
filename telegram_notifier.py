from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def load_sent_log(path: str | Path) -> dict[str, dict[str, object]]:
    log_path = Path(path)
    if not log_path.exists():
        return {}

    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    return data if isinstance(data, dict) else {}


def save_sent_log(path: str | Path, data: dict[str, dict[str, object]]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> tuple[bool, str]:
    token = bot_token.strip()
    target_chat = chat_id.strip()
    if not token or not target_chat:
        return False, "Token Telegram atau chat ID belum diisi."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urlencode(
        {
            "chat_id": target_chat,
            "text": message,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = Request(url, data=payload, method="POST")

    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return False, f"Telegram HTTP {exc.code}: {error_body}"
    except URLError as exc:
        return False, f"Telegram network error: {exc.reason}"
    except OSError as exc:
        return False, f"Telegram error: {exc}"

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        return False, "Telegram mengirim response yang tidak bisa dibaca."

    if result.get("ok") is True:
        return True, "Notifikasi Telegram terkirim."

    description = result.get("description") or "Telegram menolak request."
    return False, str(description)


def notification_key(ticker: str, candle_time: object, action: str) -> str:
    return f"{ticker.upper().strip()}|{candle_time}|{action.upper().strip()}"


def notify_once(
    bot_token: str,
    chat_id: str,
    key: str,
    message: str,
    sent_log_path: str | Path,
) -> tuple[bool, str]:
    sent_log = load_sent_log(sent_log_path)
    if key in sent_log:
        return False, "Notifikasi candle ini sudah pernah dikirim."

    sent, detail = send_telegram_message(bot_token, chat_id, message)
    if not sent:
        return False, detail

    sent_log[key] = {"message": message}
    save_sent_log(sent_log_path, sent_log)
    return True, detail
