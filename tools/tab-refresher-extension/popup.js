const $ = (id) => document.getElementById(id);

async function send(type, extra) {
  return chrome.runtime.sendMessage({ type, ...(extra || {}) });
}

function render(s) {
  if (!s || s.error) {
    $("status").textContent = "Error: " + ((s && s.error) || "no response");
    return;
  }
  const total = s.tabIds.length;
  const done = Math.min(s.index, total);
  $("status").textContent = s.running
    ? `Running: cycle ${s.cycle}/${s.cycles}, tab ${done}/${total} (waits ${s.delaySec}s between reloads)`
    : s.note || "Idle.";
  $("progress").value = total ? done / total : 0;
  $("start").disabled = s.running;
  $("stop").disabled = !s.running;
  const log = $("log");
  log.innerHTML = "";
  for (const e of [...s.log].reverse().slice(0, 20)) {
    const li = document.createElement("li");
    li.textContent = `${new Date(e.at).toLocaleTimeString()} tab ${e.id}: ${e.result}`;
    log.appendChild(li);
  }
}

async function refresh() {
  render(await send("status"));
}

$("start").addEventListener("click", async () => {
  await send("start", {
    match: $("match").value,
    delaySec: Number($("delay").value),
    cycles: Number($("cycles").value),
  });
  refresh();
});

$("stop").addEventListener("click", async () => {
  await send("stop");
  refresh();
});

(async () => {
  const s = await send("status");
  if (s && !s.error) {
    if (s.match) $("match").value = s.match;
    if (s.delaySec) $("delay").value = s.delaySec;
    if (s.cycles) $("cycles").value = s.cycles;
  }
  render(s);
  setInterval(refresh, 1000);
})();
