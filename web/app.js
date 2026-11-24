// lhdebug dashboard entry point: owns the session state and wires transport -> panels.

import { createTransport } from './transport.js';
import { RoomView, rampCSS } from './scene.js';
import { DeviceCards, LogTail, renderFindings, renderIncidentDetail, showTip, hideTip } from './panels.js';
import { Timeline } from './timeline.js';
import { RangeChart } from './range.js';
import { METRICS, kindInfo, fmtClock, fmtDuration, h } from './util.js';

const $ = (id) => document.getElementById(id);
const MAX_INCIDENTS = 1000, MAX_LOG = 1500, HEATMAP_EVERY_MS = 4000;

const S = {
  session: null,
  conn: { state: 'connecting' },
  steamvr: null,               // null = unknown yet
  everState: false,
  lastStateAt: 0,              // performance.now() of the last state message
  serverNow: Date.now() / 1000,
  devices: [],
  bases: [],
  names: new Map(),            // serial -> display name
  incidents: [],               // newest first
  incidentIds: new Set(),
  log: [],                     // newest first
  findings: [],
  selected: null,
  recording: true,
  pausedAt: null,           // server time at which recording was stopped
  windowS: 900,             // rolling window: older incidents and log lines are dropped
  metric: 'dropout',
  heatSerial: '',
};

const nameOf = (serial) => S.names.get(serial) || String(serial ?? 'system').replace(/^LH[RB]-/, '');

// ------------------------------------------------------------------ views
const room = new RoomView($('room-view'), {
  onPickIncident: (inc) => selectIncident(inc),
  onHoverCell: (cell, e) => {
    if (!cell || !e) { hideTip(); return; }
    const m = METRICS[S.metric], v = lastHeat ? lastHeat.voxel : 0.2;
    showTip([
      h('strong', `${m.label}: ${m.fmt(cell[3])}`),
      h('div', `x ${((cell[0] + 0.5) * v).toFixed(1)}, y ${((cell[1] + 0.5) * v).toFixed(1)}, z ${((cell[2] + 0.5) * v).toFixed(1)} m`),
      S.metric === 'coverage' ? null : h('div.tip-sub', `${cell[4]} sample${cell[4] === 1 ? '' : 's'} in this cell`),
    ].filter(Boolean), e.clientX, e.clientY);
  },
});
const cards = new DeviceCards($('devices'), {
  onSelectDevice: (serial) => {
    const sel = $('heat-device');
    sel.value = sel.value === serial ? '' : serial;
    S.heatSerial = sel.value;
    logTail.setDevice(S.heatSerial || '*');
    loadHeatmap();
  },
});
const logTail = new LogTail($('log'), $('log-device'), $('log-kind'), { nameOf });
const timeline = new Timeline($('timeline'), { nameOf, onSelect: (inc) => selectIncident(inc) });
timeline.onWindowChange = (sec) => {
  const sel = $('tl-window');
  let custom = sel.querySelector('[data-custom]');
  if (!custom) { custom = h('option', { 'data-custom': '1' }); sel.append(custom); }
  custom.value = String(sec);
  custom.textContent = fmtDuration(Math.round(sec));
  sel.value = custom.value;
};

// ------------------------------------------------------------------ transport
const tx = await createTransport();
tx.onStatus = (st) => {
  const was = S.conn.state;
  S.conn = st;
  renderHeader();
  renderOverlay();
  if (st.state === 'open' && was !== 'open') backfill();
};
tx.onMessage = (msg) => {
  switch (msg.type) {
    case 'hello': applySession(msg.session); if (msg.session && S.steamvr == null) S.steamvr = !!msg.session.steamvr; renderHeader(); renderOverlay(); break;
    case 'session': applySession(msg.session); renderHeader(); drawTimeline(); break;
    case 'cleared': clearLocal(); break;
    case 'state': onState(msg); break;
    case 'incident': addIncidents([msg.incident]); break;
    case 'log': addLog([msg.event]); break;
    case 'findings': S.findings = msg.findings || []; drawFindings(); break;
    default: break;      // unknown message types are ignored so the protocol can grow
  }
};

async function backfill() {
  const get = (path) => tx.getJSON(path).catch(() => null);
  const [session, inc, log, findings] = await Promise.all([
    get('/api/session'), get('/api/incidents?limit=200'), get('/api/log?limit=300'), get('/api/findings')]);
  if (session) { applySession(session); if (S.steamvr == null) S.steamvr = !!session.steamvr; }
  if (inc?.incidents) addIncidents(inc.incidents);
  if (log?.events) addLog(log.events);
  if (findings?.findings) { S.findings = findings.findings; }
  drawFindings(); renderHeader(); renderOverlay();
  loadHeatmap();
}

function onState(msg) {
  S.everState = true;
  S.lastStateAt = performance.now();
  S.serverNow = msg.t || Date.now() / 1000;
  const steamvr = msg.steamvr !== false;
  const devices = msg.devices || [], bases = msg.bases || [];
  let namesChanged = false;
  for (const d of devices) {
    const n = d.label || d.role || d.model || d.serial;
    if (S.names.get(d.serial) !== n) { S.names.set(d.serial, n); namesChanged = true; }
  }
  for (const b of bases) {
    const n = b.channel != null ? `S-${b.channel}` : String(b.serial).replace(/^LHB-/, '');
    if (S.names.get(b.serial) !== n) { S.names.set(b.serial, n); namesChanged = true; }
  }
  const listChanged = devices.length !== S.devices.length || devices.some((d, i) => d.serial !== S.devices[i].serial);
  S.devices = devices; S.bases = bases;
  if (steamvr !== S.steamvr) { S.steamvr = steamvr; renderHeader(); }
  room.setState(devices, bases);
  cards.update(devices, bases);
  if (namesChanged || listChanged) { syncDeviceOptions(); logTail.refreshNames(); drawFindings(); drawTimeline(); }
  renderOverlay();
}

function addIncidents(list) {
  let changed = false;
  for (const inc of list) {
    if (!inc || inc.id == null) continue;
    if (S.incidentIds.has(inc.id)) {                       // re-sent (e.g. attribution arrived later): replace
      const i = S.incidents.findIndex((x) => x.id === inc.id);
      if (i >= 0) S.incidents[i] = inc;
      if (S.selected && S.selected.id === inc.id) { S.selected = inc; drawDetail(); }
    } else { S.incidentIds.add(inc.id); S.incidents.push(inc); }
    changed = true;
  }
  if (!changed) return;
  S.incidents.sort((a, b) => b.t_start - a.t_start);
  for (const dropped of S.incidents.splice(MAX_INCIDENTS)) S.incidentIds.delete(dropped.id);
  room.setIncidents(S.incidents.slice(0, 80));
  drawTimeline();
  if (!S.findings.length) drawFindings();
}

// ------------------------------------------------- recording controls
function applySession(session) {
  if (!session) return;
  S.session = session;
  S.recording = session.recording !== false;
  S.pausedAt = session.paused_at ?? null;
  room.setPlayArea(session.play_area || null);
  if (session.window_s) {
    S.windowS = Number(session.window_s);
    const sel = $('tl-window');
    if (![...sel.options].some((o) => Number(o.value) === S.windowS)) {
      sel.append(h('option', { value: String(S.windowS) }, fmtDuration(Math.round(S.windowS))));
    }
    sel.value = String(S.windowS);
    timeline.setWindow(S.windowS);
  }
  renderRecording();
}

function renderRecording() {
  const b = $('btn-rec');
  b.setAttribute('aria-pressed', String(S.recording));
  b.textContent = S.recording ? '\u25A0 Stop recording' : '\u25CF Resume recording';
  document.body.classList.toggle('is-stopped', !S.recording);
  const note = $('rec-note');
  note.hidden = S.recording;
  note.textContent = S.recording ? '' : `Recording stopped${S.pausedAt ? ' at ' + fmtClock(S.pausedAt) : ''}: the timeline is frozen and nothing new is being added. `;
}

function clearLocal() {
  S.incidents = []; S.incidentIds.clear(); S.log = []; S.findings = [];
  selectIncident(null);
  room.setIncidents([]);
  logTail.setEvents(S.log);
  drawFindings(); drawTimeline(); renderHeader(); loadHeatmap();
}

// Drop what has fallen out of the rolling window (the backend does the same).
function pruneWindow() {
  if (!S.recording || !S.everState) return;
  const cutoff = S.serverNow - S.windowS;
  const ni = S.incidents.length, nl = S.log.length;
  S.incidents = S.incidents.filter((i) => i.t_end >= cutoff);
  S.log = S.log.filter((e) => e.t >= cutoff);
  if (S.incidents.length !== ni) {
    S.incidentIds = new Set(S.incidents.map((i) => i.id));
    if (S.selected && !S.incidentIds.has(S.selected.id)) selectIncident(null);
    room.setIncidents(S.incidents.slice(0, 80));
  }
  if (S.log.length !== nl) logTail.setEvents(S.log);
}

async function control(body) {
  try { applySession(await tx.postJSON('/api/control', body)); renderHeader(); drawTimeline(); }
  catch (e) { console.error('control failed', e); }
}

function addLog(list) {
  if (!list.length) return;
  for (const e of list) if (e) S.log.push(e);
  S.log.sort((a, b) => b.t - a.t);
  S.log.length = Math.min(S.log.length, MAX_LOG);
  logTail.setEvents(S.log);
  drawTimeline();
}

// ------------------------------------------------------------------ header / overlay
function renderHeader() {
  const mode = S.session?.mode;
  const mp = $('mode-pill');
  mp.dataset.mode = mode || 'unknown';
  mp.textContent = mode ? ({ live: '● LIVE', sim: '◆ SIMULATION', replay: '▶ REPLAY' }[mode] || mode.toUpperCase()) : 'mode …';
  if (tx.kind === 'mock') mp.textContent += ' · mock data';

  const sp = $('steamvr-pill');
  const on = S.conn.state === 'open' ? S.steamvr : null;
  sp.dataset.on = on == null ? 'unknown' : String(on);
  sp.textContent = on == null ? 'SteamVR: unknown' : on ? '✔ SteamVR connected' : '✖ SteamVR not running';

  const cp = $('conn-pill');
  cp.dataset.conn = S.conn.state;
  cp.textContent = S.conn.state === 'open' ? '✔ Connected'
    : S.conn.state === 'connecting' ? '↻ Connecting…'
      : `✖ Disconnected · retrying${S.conn.retryIn ? ' in ' + Math.max(1, Math.round(S.conn.retryIn)) + ' s' : ''}`;

  const info = $('session-info');
  if (S.session) {
    const up = S.session.started ? fmtDuration(Math.max(0, Math.round(S.serverNow - S.session.started))) : null;
    info.textContent = [`session ${S.session.id}`, up && `running ${up}`, `${S.incidents.length} incidents`].filter(Boolean).join(' · ');
  } else info.textContent = '';
}

function renderOverlay() {
  const ov = $('overlay');
  let title = '', text = '';
  const stale = S.everState && performance.now() - S.lastStateAt > 3000;
  if (S.conn.state !== 'open') {
    title = S.everState ? 'Connection lost' : 'Connecting to lhdebug';
    text = S.everState
      ? 'The lhdebug service stopped answering. Reconnecting automatically; the panels show the last data received.'
      : 'Waiting for the lhdebug service on this machine. Start it if it is not running; this page reconnects by itself.';
  } else if (S.steamvr === false && !S.devices.length) {
    title = 'Waiting for SteamVR';
    text = 'SteamVR is not running or no headset is connected. Start SteamVR and turn on your trackers; this page will pick them up by itself.';
  } else if (!S.everState || stale) {
    title = stale ? 'No data from the sampler' : 'Waiting for the first pose';
    text = stale ? 'The connection is open but pose updates have stopped.' : 'Connected. Waiting for the sampler to send device poses.';
  }
  const key = title + '|' + text;
  if (ov.dataset.key === key) return;
  ov.dataset.key = key;
  ov.hidden = !title;
  $('overlay-title').textContent = title;
  $('overlay-text').textContent = text;
  document.body.classList.toggle('is-waiting', !!title);
  drawFindings();
}

// ------------------------------------------------------------------ panels
function drawFindings() {
  const waiting = S.conn.state !== 'open' || (S.steamvr === false && !S.devices.length) || !S.everState;
  renderFindings($('findings'), S.findings, { nameOf, waiting: waiting && !S.findings.length, incidentCount: S.incidents.length });
  $('diag-note').textContent = S.findings.length
    ? `${S.findings.length} finding${S.findings.length === 1 ? '' : 's'}, most likely first`
    : 'What is wrong, most likely first';
}

function drawTimeline() {
  if (drawTimeline.queued) return;
  drawTimeline.queued = true;
  requestAnimationFrame(() => {
    drawTimeline.queued = false;
    timeline.setData({
      lanes: S.devices.map((d) => ({ serial: d.serial, label: nameOf(d.serial) })),
      incidents: S.incidents, log: S.log,
      now: !S.recording && S.pausedAt ? S.pausedAt
        : S.conn.state === 'open' && S.everState ? S.serverNow : Date.now() / 1000,
      selectedId: S.selected ? S.selected.id : null,
    });
    timeline.render();
  });
}

function drawDetail() {
  renderIncidentDetail($('incident-detail'), S.selected, { nameOf, onClose: () => selectIncident(null) });
}

function selectIncident(inc) {
  S.selected = inc && S.selected && S.selected.id === inc.id ? null : inc;
  drawDetail();
  const s = S.selected;
  room.highlightIncident(s, s ? `#${s.id} ${kindInfo(s.kind).label} · ${nameOf(s.serial)} · ${fmtClock(s.t_start)}` : '');
  drawTimeline();
  if (s) $('room-panel').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function syncDeviceOptions() {
  const sel = $('heat-device'), cur = sel.value;
  sel.replaceChildren(h('option', { value: '' }, 'All devices'),
    ...S.devices.map((d) => h('option', { value: d.serial }, nameOf(d.serial))));
  sel.value = S.devices.some((d) => d.serial === cur) ? cur : '';
  if (sel.value !== S.heatSerial) { S.heatSerial = sel.value; loadHeatmap(); }
}

// ------------------------------------------------------------------ heatmap
let lastHeat = null, heatSeq = 0, sliceTouched = false;

async function loadHeatmap() {
  room.setSelectedDevice(S.heatSerial);
  if (S.conn.state !== 'open') return;
  const seq = ++heatSeq;
  const metric = S.metric;
  const q = new URLSearchParams({ metric });
  if (S.heatSerial && metric !== 'coverage') q.set('serial', S.heatSerial);
  let snap;
  try { snap = await tx.getJSON('/api/heatmap?' + q); } catch { snap = null; }
  if (seq !== heatSeq) return;                                  // a newer request superseded this one
  if (!snap || !Array.isArray(snap.cells)) { renderLegend(null, metric); return; }
  lastHeat = snap;
  room.setHeatmap(snap, { reverse: METRICS[metric].worst === 'low' });
  syncSliceRange();
  renderLegend(snap, metric);
}

function renderLegend(snap, metric) {
  const m = METRICS[metric], lg = $('legend');
  lg.hidden = false;
  const dev = S.heatSerial && metric !== 'coverage' ? nameOf(S.heatSerial) : 'all devices';
  $('legend-title').textContent = `${m.label} (${m.unit}) · ${metric === 'coverage' ? 'predicted from base poses' : dev}`;
  const empty = !snap || !snap.cells.length;
  $('legend-bar').style.background = rampCSS(m.worst === 'low');
  $('legend-bar').style.visibility = empty ? 'hidden' : '';
  $('legend-min').textContent = empty ? '' : m.fmt(snap.min);
  $('legend-max').textContent = empty ? '' : m.fmt(snap.max);
  $('legend-note').textContent = empty
    ? (snap ? 'No data for this selection yet. Cells appear as devices move through the room.' : 'Heatmap unavailable.')
    : `${m.desc}. ${m.worst ? 'Brighter and larger = worse' : 'Brighter and larger = more'}. ${snap.cells.length.toLocaleString()} cells of ${Math.round(snap.voxel * 100)} cm.`;
}

function syncSliceRange() {
  const lo = $('slice-lo'), hi = $('slice-hi');
  const top = Math.max(2.4, Math.ceil(room.heatHeightRange()[1] * 5) / 5);
  const step = lastHeat ? lastHeat.voxel : 0.2;
  if (Number(hi.max) !== top || Number(hi.step) !== step) {
    const wasAll = !sliceTouched || Number(hi.value) >= Number(hi.max);
    for (const el of [lo, hi]) { el.max = top; el.step = step; }
    if (wasAll) hi.value = top;
    applySlice();
  }
}

function applySlice() {
  const lo = $('slice-lo'), hi = $('slice-hi');
  const a = Number(lo.value), b = Number(hi.value), max = Number(hi.max);
  const all = a <= 0 && b >= max;
  $('slice-out').textContent = all ? 'all' : `${a.toFixed(1)}–${b.toFixed(1)} m`;
  room.setSlice(all ? -1e9 : a, all ? 1e9 : b);
}

// ------------------------------------------------------------------ controls
$('metric').addEventListener('change', (e) => {
  S.metric = e.target.value;
  $('heat-device').disabled = S.metric === 'coverage';
  loadHeatmap();
});
$('heat-device').addEventListener('change', (e) => { S.heatSerial = e.target.value; loadHeatmap(); });
$('slice-lo').addEventListener('input', () => {
  sliceTouched = true;
  if (Number($('slice-lo').value) > Number($('slice-hi').value)) $('slice-hi').value = $('slice-lo').value;
  applySlice();
});
$('slice-hi').addEventListener('input', () => {
  sliceTouched = true;
  if (Number($('slice-hi').value) < Number($('slice-lo').value)) $('slice-lo').value = $('slice-hi').value;
  applySlice();
});
$('btn-top').addEventListener('click', (e) => {
  const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
  e.currentTarget.setAttribute('aria-pressed', String(on));
  room.setTopDown(on);
});
for (const b of document.querySelectorAll('.toggle[data-layer]')) {
  b.addEventListener('click', () => {
    const on = b.getAttribute('aria-pressed') !== 'true';
    b.setAttribute('aria-pressed', String(on));
    room.setLayer(b.dataset.layer, on);
    if (b.dataset.layer === 'heatmap') $('legend').classList.toggle('is-off', !on);
  });
}
$('btn-reset').addEventListener('click', () => room.resetView());
$('tl-window').addEventListener('change', (e) => {
  const sec = Number(e.target.value);
  S.windowS = sec;
  timeline.setWindow(sec);
  control({ window_s: sec });
});
$('btn-rec').addEventListener('click', () => control({ action: S.recording ? 'stop' : 'resume' }));
{
  // Clear needs a second click within 3 s; a modal confirm is awkward on a VR overlay.
  const b = $('btn-clear');
  let armed = 0;
  const disarm = () => { clearTimeout(armed); armed = 0; b.textContent = 'Clear'; b.classList.remove('is-armed'); };
  b.addEventListener('click', () => {
    if (armed) { disarm(); control({ action: 'clear' }); return; }
    b.textContent = 'Click again to clear';
    b.classList.add('is-armed');
    armed = setTimeout(disarm, 3000);
  });
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.selected) selectIncident(null); });

// URL presets, handy for kiosk / overlay use: ?metric=coverage&top=1&slice=0-0.4
{
  const q = new URLSearchParams(location.search);
  if (q.get('metric') && METRICS[q.get('metric')]) { $('metric').value = S.metric = q.get('metric'); $('heat-device').disabled = S.metric === 'coverage'; }
  if (q.get('top') === '1') $('btn-top').click();
  const sl = /^([\d.]+)-([\d.]+)$/.exec(q.get('slice') || '');
  if (sl) { sliceTouched = true; $('slice-lo').value = sl[1]; $('slice-hi').value = sl[2]; applySlice(); }
  if (q.get('window')) { timeline.windowSec = Number(q.get('window')) || 300; $('tl-window').value = String(timeline.windowSec); }
}

// ------------------------------------------------------------------ timers
setInterval(loadHeatmap, HEATMAP_EVERY_MS);

// ------------------------------------------------------------- range tab
const rangeChart = new RangeChart($('range-view'), { nameOf });
let rangeOpen = false;
async function loadRange() {
  if (!rangeOpen || S.conn.state !== 'open') return;
  try { rangeChart.setData(await tx.getJSON('/api/range')); } catch { /* keep the last chart */ }
}
function showTab(range) {
  rangeOpen = range;
  $('tab-log').setAttribute('aria-selected', String(!range));
  $('tab-range').setAttribute('aria-selected', String(range));
  $('log').hidden = range;
  $('range-view').hidden = !range;
  document.querySelector('.log-panel').classList.toggle('show-range', range);
  if (range) { rangeChart.render(); loadRange(); }
}
$('tab-log').addEventListener('click', () => showTab(false));
$('tab-range').addEventListener('click', () => showTab(true));
$('range-view-select').addEventListener('change', (e) => rangeChart.setView(e.target.value));
setInterval(loadRange, 4000);
if (new URLSearchParams(location.search).get('tab') === 'range') showTab(true);
setInterval(() => { pruneWindow(); renderHeader(); renderOverlay(); drawTimeline(); }, 1000);

renderHeader();
renderOverlay();
drawFindings();
logTail.render();
drawTimeline();
tx.start();

// Debug handle for the console; not used by the app itself.
window.lhdebug = { state: S, room, selectIncident, transport: tx, showTab };
