// Slow Tab Refresher - MV3 service worker.
//
// Strategy: reload exactly one tab, then schedule a one-shot chrome.alarms
// alarm for the next one. Alarms (unlike setTimeout) survive service-worker
// shutdown, so the run continues even if Chrome suspends this worker.

const ALARM_NAME = "slow-refresh-next";
const LOG_KEEP = 50;

async function getState() {
  const s = await chrome.storage.local.get("state");
  return (
    s.state || {
      running: false,
      match: "",
      tabIds: [],
      index: 0,
      cycle: 0,
      cycles: 1,
      delaySec: 10,
      log: [],
      note: "",
    }
  );
}

async function setState(patch) {
  const cur = await getState();
  const next = { ...cur, ...patch };
  await chrome.storage.local.set({ state: next });
  return next;
}

async function collectTabs(match) {
  const tabs = await chrome.tabs.query({});
  const m = (match || "").trim().toLowerCase();
  return tabs
    .filter(
      (t) =>
        t.id != null &&
        /^https?:\/\//i.test(t.url || "") &&
        (!m || (t.url || "").toLowerCase().includes(m))
    )
    .map((t) => t.id);
}

function scheduleNext(delaySec) {
  chrome.alarms.create(ALARM_NAME, {
    delayInMinutes: Math.max(delaySec, 1) / 60,
  });
}

async function startRun({ match, delaySec, cycles }) {
  await chrome.alarms.clear(ALARM_NAME);
  delaySec = Math.min(Math.max(Number(delaySec) || 10, 1), 3600);
  cycles = Math.min(Math.max(Math.trunc(Number(cycles)) || 1, 1), 100);
  const tabIds = await collectTabs(match);
  if (!tabIds.length) {
    return setState({
      running: false,
      match,
      tabIds: [],
      index: 0,
      cycle: 0,
      cycles,
      delaySec,
      log: [],
      note: "No matching tabs open.",
    });
  }
  await setState({
    running: true,
    match,
    tabIds,
    index: 0,
    cycle: 1,
    cycles,
    delaySec,
    log: [],
    note: "",
  });
  await reloadCurrent(); // first tab immediately, pacing starts after it
  return getState();
}

async function reloadCurrent() {
  const s = await getState();
  if (!s.running) return s;
  if (s.index >= s.tabIds.length) {
    return finishCycle(s);
  }
  const id = s.tabIds[s.index];
  let result = "ok";
  try {
    await chrome.tabs.reload(id);
  } catch (err) {
    result = "failed: " + (err && err.message ? err.message : String(err));
  }
  const log = [...s.log.slice(-(LOG_KEEP - 1)), { id, result, at: Date.now() }];
  await setState({ index: s.index + 1, log });
  scheduleNext(s.delaySec);
  return getState();
}

async function finishCycle(s) {
  if (s.cycle >= s.cycles) {
    await chrome.alarms.clear(ALARM_NAME);
    return setState({
      running: false,
      note: `Done: ${s.tabIds.length} tab(s) x ${s.cycles} cycle(s).`,
    });
  }
  const tabIds = await collectTabs(s.match); // re-collect: tabs may have opened/closed
  if (!tabIds.length) {
    await chrome.alarms.clear(ALARM_NAME);
    return setState({ running: false, note: "No matching tabs left; stopped." });
  }
  await setState({ tabIds, index: 0, cycle: s.cycle + 1 });
  return reloadCurrent();
}

async function stopRun() {
  await chrome.alarms.clear(ALARM_NAME);
  return setState({ running: false, note: "Stopped." });
}

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (!alarm || alarm.name !== ALARM_NAME) return;
  const s = await getState();
  if (!s.running) return;
  await reloadCurrent();
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    if (msg.type === "start") return startRun(msg);
    if (msg.type === "stop") return stopRun();
    return getState(); // "status" and anything else
  })().then(
    sendResponse,
    (err) => sendResponse({ error: String((err && err.message) || err) })
  );
  return true; // async response
});
