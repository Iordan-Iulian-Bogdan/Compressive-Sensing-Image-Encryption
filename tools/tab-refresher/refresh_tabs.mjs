#!/usr/bin/env node
import process from "node:process";

const DEFAULTS = { port: 9222, delay: 10, cycles: 1, timeout: 30 };

function usage() {
  console.log(`refresh_tabs.mjs - slowly reload Chrome tabs, one at a time

Usage: node refresh_tabs.mjs [options]

Options:
  -m, --match <text>    Only tabs whose URL contains <text> (repeatable,
                        comma-separated also works). Default: all http(s) tabs.
  -d, --delay <sec>     Pause between tabs after page load (default ${DEFAULTS.delay})
  -c, --cycles <n>      Repeat the whole pass n times (default ${DEFAULTS.cycles})
  -p, --port <n>        Chrome debugging port (default ${DEFAULTS.port})
  -t, --timeout <sec>   Max wait for a page load (default ${DEFAULTS.timeout})
      --list            Print matching tabs and exit
      --no-activate     Do not bring each tab to the foreground while reloading
  -h, --help            Show this help

Examples:
  node refresh_tabs.mjs --match example.com --delay 15 --cycles 3
  node refresh_tabs.mjs --list -m example.com -m cdn.example.org
`);
}

function parseArgs(argv) {
  const opts = { ...DEFAULTS, matches: [], list: false, activate: true, help: false };
  const num = (raw, name) => {
    const v = Number(raw);
    if (!Number.isFinite(v) || v < 0) {
      console.error(`Invalid value for ${name}: ${raw}`);
      process.exit(2);
    }
    return v;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = (name) => {
      const v = argv[++i];
      if (v === undefined) {
        console.error(`Missing value for ${name}`);
        process.exit(2);
      }
      return v;
    };
    switch (a) {
      case "-m": case "--match":
        opts.matches.push(...next(a).split(",").map((s) => s.trim().toLowerCase()).filter(Boolean));
        break;
      case "-d": case "--delay": opts.delay = num(next(a), a); break;
      case "-c": case "--cycles": opts.cycles = Math.max(1, Math.trunc(num(next(a), a))); break;
      case "-p": case "--port": opts.port = Math.trunc(num(next(a), a)); break;
      case "-t": case "--timeout": opts.timeout = num(next(a), a); break;
      case "--list": opts.list = true; break;
      case "--no-activate": opts.activate = false; break;
      case "-h": case "--help": opts.help = true; break;
      default:
        console.error(`Unknown argument: ${a} (see --help)`);
        process.exit(2);
    }
  }
  if (opts.matches.length) opts.matches = [...new Set(opts.matches)];
  return opts;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const timeoutSignal = (ms) => (typeof AbortSignal !== "undefined" && AbortSignal.timeout
  ? AbortSignal.timeout(ms)
  : AbortSignal.abort());

async function fetchTabs(port) {
  let res;
  try {
    res = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: timeoutSignal(4000) });
  } catch (err) {
    failConnect(port, err);
  }
  if (!res.ok) failConnect(port, new Error(`HTTP ${res.status}`));
  const targets = await res.json();
  return targets.filter((t) => t.type === "page" && t.webSocketDebuggerUrl);
}

function failConnect(port, err) {
  console.error(`Cannot reach Chrome DevTools on 127.0.0.1:${port} (${err.message}).`);
  console.error(
    [
      "Start Chrome with remote debugging enabled:",
      "",
      "  chrome.exe --remote-debugging-port=" + port + ' --user-data-dir="%TEMP%\\chrome-debug-profile"',
      "",
      "Note: Chrome 136+ ignores --remote-debugging-port on the default profile,",
      "so a separate --user-data-dir is required (bookmarks re-sync, or copy your",
      "profile folder over).",
    ].join("\n")
  );
  process.exit(1);
}

function selectTabs(tabs, opts) {
  return tabs.filter((t) => {
    if (!/^https?:\/\//i.test(t.url)) return false;
    if (!opts.matches.length) return true;
    const url = t.url.toLowerCase();
    return opts.matches.some((m) => url.includes(m));
  });
}

let activeSocket = null;

function reloadTab(tab, opts) {
  return new Promise((resolve) => {
    let done = false;
    let ws;
    const finish = (status) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      activeSocket = null;
      try { ws?.close(); } catch { /* ignore */ }
      resolve(status);
    };
    const timer = setTimeout(() => finish("timeout"), opts.timeout * 1000);

    try {
      ws = new WebSocket(tab.webSocketDebuggerUrl);
    } catch (err) {
      finish(`unreachable (${err.message})`);
      return;
    }
    activeSocket = ws;

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({ id: 1, method: "Page.enable" }));
      if (opts.activate) ws.send(JSON.stringify({ id: 2, method: "Page.bringToFront" }));
      ws.send(JSON.stringify({ id: 3, method: "Page.reload", params: { ignoreCache: false } }));
    });
    ws.addEventListener("message", (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.method === "Page.loadEventFired") finish("loaded");
      if (msg.id === 3 && msg.error) finish(`error (${msg.error.message})`);
    });
    ws.addEventListener("error", () => finish("unreachable"));
    ws.addEventListener("close", (ev) => finish(ev && ev.code !== 1000 ? "unreachable" : "loaded"));
  });
}

function label(tab) {
  const title = (tab.title || "(untitled)").trim();
  return title.length > 60 ? `${title.slice(0, 57)}...` : title;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    usage();
    return;
  }

  const all = await fetchTabs(opts.port);
  const tabs = selectTabs(all, opts);

  if (opts.list) {
    if (!tabs.length) console.log("No matching tabs.");
    for (const t of tabs) console.log(`${label(t)}\n  ${t.url}`);
    console.log(`${tabs.length} matching tab(s).`);
    return;
  }

  if (!tabs.length) {
    console.log("No matching tabs to refresh.");
    return;
  }

  console.log(`${tabs.length} tab(s), ${opts.delay}s between tabs, ${opts.cycles} cycle(s).`);
  const counts = { loaded: 0, timeout: 0, failed: 0 };
  let stopped = false;

  const onSigint = () => {
    stopped = true;
    console.log("\nStopping after current tab...");
    try { activeSocket?.close(); } catch { /* ignore */ }
    process.exit(130);
  };
  process.on("SIGINT", onSigint);

  for (let c = 1; c <= opts.cycles && !stopped; c++) {
    if (opts.cycles > 1) console.log(`--- cycle ${c}/${opts.cycles} ---`);
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      const status = await reloadTab(tab, opts);
      const short = { loaded: "ok", timeout: "load timeout", unreachable: "unreachable" };
      const word = status.startsWith("error") ? status : short[status] ?? status;
      console.log(`[${i + 1}/${tabs.length}] ${word === "ok" ? "ok" : word}  ${label(tab)}`);
      if (status === "loaded") counts.loaded++;
      else if (status === "timeout") counts.timeout++;
      else counts.failed++;
      if (i < tabs.length - 1 && !stopped) await sleep(opts.delay * 1000);
    }
    if (c < opts.cycles && !stopped) await sleep(opts.delay * 1000);
  }

  process.removeListener("SIGINT", onSigint);
  console.log(
    `Done. ${counts.loaded} loaded, ${counts.timeout} timed out, ${counts.failed} unreachable.`
  );
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
