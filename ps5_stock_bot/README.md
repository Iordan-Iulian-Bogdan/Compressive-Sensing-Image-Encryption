# PS5 Stock Alert Bot (altex.ro) — LOCAL ONLY

Pops a Windows notification + sound + auto-opens the product page when any of
the 5 watched PS5 consoles leaves "stoc epuizat" state. No email, no passwords,
nothing leaves your PC (only fetches altex.ro pages).

## 1. Test alerts

```powershell
cd ps5_stock_bot
python ps5_stock_bot.py --test-alert
```

You should see a popup, hear a sound, and get the first product page in your browser.

## 2. Run every 5 minutes (Windows Task Scheduler)

IMPORTANT: the task must run in your own login session (not headless), otherwise
popups and browser windows can't reach you.

Option A — script does it (run PowerShell **as Administrator**):
```powershell
cd ps5_stock_bot
.\register_task.ps1
```

Option B — manual:
- Task Scheduler → Create Task → "Run only when user is logged on"
- Triggers: Daily, repeat every 5 min indefinitely
- Action: `pythonw.exe` args `C:\...\ps5_stock_bot\ps5_stock_bot.py`
- Start in: `C:\...\ps5_stock_bot`

Or keep a terminal open: `python ps5_stock_bot.py --loop`

## 3. How it works

- Fetches each product page, looks for `stoc epuizat` (out) vs `adauga in cos` / `in stoc` (in).
- Fetch failures = UNKNOWN → no alert (never false-alarms on timeouts).
- `state.json` tracks last status → alerts **once** per restock, not every 5 min.
- Log: `stock_bot.log`.

## Files

| File | Purpose |
|---|---|
| `ps5_stock_bot.py` | checker + local alerts (stdlib only) |
| `config.json` | product URLs, schedule, alert toggles |
| `register_task.ps1` | registers the 5-min scheduled task |
| `state.json` | auto-created, last known status |
| `stock_bot.log` | auto-created run log |
