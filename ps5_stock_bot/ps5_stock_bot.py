#!/usr/bin/env python3
"""PS5 stock checker for altex.ro — LOCAL-ONLY alerts, no email, no passwords.

On restock: Windows popup + sound + auto-opens the product page(s) in your browser.
Nothing leaves your PC (no SMTP, no accounts, no network calls except altex.ro).

Logic (fail-safe):
  - page contains "stoc epuizat"  -> OUT OF STOCK
  - page does NOT contain it, AND contains an add-to-cart signal -> IN STOCK
  - fetch error / timeout / unknown layout -> UNKNOWN (no alert, logged only)

State file (state.json) remembers last status per URL so you get
one alert on the OOS -> IN-STOCK transition, not every 5 minutes.

Usage:
  python ps5_stock_bot.py               # single check (for Task Scheduler, run every 5 min)
  python ps5_stock_bot.py --loop        # stay running, check every 5 min
  python ps5_stock_bot.py --test-alert  # popup + sound + open first product page (verifies alerts work)
"""
import ctypes
import json
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "config.json"
STATE_PATH = BASE / "state.json"
LOG_PATH = BASE / "stock_bot.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml",
}

OUT_OF_STOCK_MARKERS = [
    "stoc epuizat",
    "stoc epuizat definitiv",
    "indisponibil",
    "momentan indisponibil",
]

# Any of these suggests a buyable page (Altex uses "Adauga in cos").
IN_STOCK_MARKERS = [
    "adauga in cos",
    "adaugă în coș",
    "adauga-in-cos",
    "addtocart",
    "in stoc",
    "în stoc",
]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def fetch_html(url: str, timeout: int) -> str | None:
    last_err = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="ignore")
        except Exception as e:  # timeout, HTTP 403/429, network blip
            last_err = e
            log(f"  fetch attempt {attempt}/3 failed for {url[:60]}...: {e}")
            time.sleep(3 * attempt)
    log(f"  giving up after 3 attempts: {last_err}")
    return None


def classify(html: str | None) -> tuple[str, str]:
    """Return (status, detail). status in IN_STOCK / OUT_OF_STOCK / UNKNOWN."""
    if html is None:
        return "UNKNOWN", "fetch failed"
    low = html.lower()
    for marker in OUT_OF_STOCK_MARKERS:
        if marker in low:
            # "stoc epuizat" wins even if cart snippets exist elsewhere in page
            return "OUT_OF_STOCK", f"found '{marker}'"
    for marker in IN_STOCK_MARKERS:
        if marker in low:
            return "IN_STOCK", f"found '{marker}', no OOS marker"
    m = re.search(r"(\d[\d\.]*,\d{2})\s*lei", low)
    price = m.group(1) if m else "no price found"
    return "UNKNOWN", f"no stock markers ({price})"


def extract_price(html: str) -> str:
    m = re.search(r"(\d[\d\.]*,\d{2})\s*lei", html.lower())
    return m.group(1) + " lei" if m else "?"


def play_sound() -> None:
    try:
        import winsound

        for _ in range(3):
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            time.sleep(0.4)
    except Exception as e:
        log(f"  sound failed: {e}")


def show_toast(title: str, message: str) -> None:
    """Best-effort Windows balloon notification (no dependencies)."""
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$ni = New-Object System.Windows.Forms.NotifyIcon; "
        "$ni.Icon = [System.Drawing.SystemIcons]::Information; "
        f"$ni.BalloonTipTitle = '{title}'; "
        f"$ni.BalloonTipText = '{message}'; "
        "$ni.Visible = $true; $ni.ShowBalloonTip(10000); "
        "Start-Sleep -Seconds 11; $ni.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            timeout=20,
            check=False,
            capture_output=True,
        )
    except Exception as e:
        log(f"  toast failed: {e}")


def show_popup(title: str, message: str) -> None:
    """Blocking Windows message box — impossible to miss."""
    try:
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1000)  # MB_ICONINFORMATION | MB_TOPMOST
    except Exception as e:
        log(f"  popup failed: {e}")


def alert_user(cfg: dict, hits: list[dict]) -> None:
    alert_cfg = cfg.get("alert", {})
    names = "\n".join(f"- {h['name']} ({h['price']})" for h in hits)
    title = "PS5 IN STOCK!"
    short = f"{len(hits)} console(s): {hits[0]['name']}" + (" (+more)" if len(hits) > 1 else "")

    if alert_cfg.get("open_browser", True):
        for h in hits:
            try:
                webbrowser.open(h["url"])
                log(f"  opened browser: {h['url'][:60]}...")
            except Exception as e:
                log(f"  browser open failed: {e}")
            time.sleep(0.5)
    if alert_cfg.get("sound", True):
        play_sound()
    show_toast(title, short)  # best-effort, never blocks
    if alert_cfg.get("popup", True):
        show_popup(title, f"{names}\n\nPage(s) opened in your browser. Hurry, stock goes fast!")


def check_once(cfg: dict, state: dict) -> list[dict]:
    """Run one pass over all products. Returns list of newly-in-stock items."""
    newly_in_stock = []
    for p in cfg["products"]:
        name, url = p["name"], p["url"]
        log(f"Checking: {name}")
        html = fetch_html(url, cfg.get("request_timeout_seconds", 30))
        status, detail = classify(html)
        prev = state.get(url, {}).get("status", "UNKNOWN")
        price = extract_price(html) if html else "?"
        log(f"  -> {status} ({detail}) price={price} prev={prev}")
        state[url] = {
            "name": name,
            "status": status,
            "price": price,
            "last_check": datetime.now().isoformat(timespec="seconds"),
        }
        if status == "IN_STOCK" and prev != "IN_STOCK":
            newly_in_stock.append({**p, "price": price})
        time.sleep(2)  # be polite, avoid rate limiting
    save_state(state)
    return newly_in_stock


def main() -> int:
    cfg = load_config()
    if "--test-alert" in sys.argv:
        log("Test alert: popup + sound + browser.")
        alert_user(cfg, [{**cfg["products"][0], "price": "TEST"}])
        log("Test alert done. If you saw/heard it and a page opened, alerts work.")
        return 0

    state = load_state()
    loop = "--loop" in sys.argv
    interval = cfg.get("check_interval_seconds", 300)

    while True:
        try:
            hits = check_once(cfg, state)
        except Exception as e:
            log(f"ERROR during check pass: {e}")
            hits = []
        if hits:
            log(f"RESTOCK! {len(hits)} product(s) in stock — alerting.")
            alert_user(cfg, hits)
        else:
            log("No new stock. Sleeping." if loop else "Done — no new stock.")
        if not loop:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
