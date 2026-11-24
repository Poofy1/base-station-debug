// Shared helpers: tracking-state derivation, formatting, small DOM utilities.

export const STATES = {
  ok:           { label: 'OK',           icon: '●', color: '#2fbf4a' },   // filled circle
  degraded:     { label: 'Degraded',     icon: '▲', color: '#fab219' },   // triangle
  lost:         { label: 'Lost',         icon: '✖', color: '#ff6b5e' },   // cross
  disconnected: { label: 'Disconnected', icon: '○', color: '#9a988f' },   // hollow circle
};

// OpenVR ETrackingResult names (model.TrackingResult).
const RESULT_NAMES = {
  1: 'uninitialized', 100: 'calibrating', 101: 'calibrating, out of range',
  200: 'running ok', 201: 'out of range', 300: 'rotation only',
};
export const resultName = (r) => RESULT_NAMES[r] || `result ${r}`;

/** Collapse the wire fields of a device into one of the four display states. */
export function trackingState(d) {
  if (!d.connected) return 'disconnected';
  if (!d.valid || d.result === 300 || d.result === 1) return 'lost';
  if (d.result !== 200 || d.ok === false) return 'degraded';
  if (d.incident || (d.ok_frac != null && d.ok_frac < 0.95)) return 'degraded';
  return 'ok';
}

// Incident kinds grouped by what they mean for the user; the group picks the colour,
// the short code keeps kinds apart without relying on colour.
export const KINDS = {
  dropout:        { label: 'Dropout',        code: 'DROP',  group: 'loss' },
  disconnect:     { label: 'Disconnect',     code: 'DISC',  group: 'loss' },
  jump:           { label: 'Jump',           code: 'JUMP',  group: 'quality' },
  jitter:         { label: 'Jitter',         code: 'JIT',   group: 'quality' },
  stale:          { label: 'Stale poses',    code: 'STALE', group: 'quality' },
  base_shift:     { label: 'Base shift',     code: 'BASE',  group: 'geometry' },
  universe_shift: { label: 'Universe shift', code: 'UNIV',  group: 'geometry' },
};
export const GROUPS = {
  loss:     { label: 'Tracking lost',    color: '#ff6b5e' },
  quality:  { label: 'Tracking quality', color: '#fab219' },
  geometry: { label: 'Room geometry',    color: '#9085e9' },
};
export const kindInfo = (k) => KINDS[k] || { label: String(k), code: String(k).slice(0, 5).toUpperCase(), group: 'quality' };

export const CAUSES = {
  occlusion: 'Occlusion', radio: 'Radio link', base_station: 'Base station',
  reflection: 'Reflection', imu: 'IMU', usb: 'USB', battery: 'Battery',
  calibration: 'Calibration', unknown: 'Unknown',
};
export const causeLabel = (c) => CAUSES[c] || String(c);

export const METRICS = {
  dropout:     { label: 'Dropout',     unit: '%',     fmt: (v) => (v * 100).toFixed(v < 0.1 ? 1 : 0) + ' %', worst: 'high',
                 desc: 'share of samples without a valid pose' },
  jitter:      { label: 'Jitter',      unit: 'mm',    fmt: (v) => v.toFixed(v < 10 ? 1 : 0) + ' mm', worst: 'high',
                 desc: 'high-frequency position noise (RMS)' },
  jumps:       { label: 'Jumps',       unit: 'count', fmt: (v) => fmtCount(v), worst: 'high',
                 desc: 'position discontinuities' },
  update_rate: { label: 'Update rate', unit: 'Hz',    fmt: (v) => v.toFixed(0) + ' Hz', worst: 'low',
                 desc: 'distinct poses per second' },
  samples:     { label: 'Samples',     unit: 'count', fmt: (v) => fmtCount(v), worst: null,
                 desc: 'how much data was collected here' },
  coverage:    { label: 'Coverage',    unit: 'bases', fmt: (v) => (Number.isInteger(v) ? v : v.toFixed(1)) + (v === 1 ? ' base' : ' bases'), worst: 'low',
                 desc: 'predicted base stations with line of sight' },
};

export function fmtCount(v) {
  if (v >= 10000) return (v / 1000).toFixed(0) + 'k';
  if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

export function fmtClock(t) {
  const d = new Date(t * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function fmtDuration(s) {
  if (s < 1) return Math.round(s * 1000) + ' ms';
  if (s < 60) return s.toFixed(1) + ' s';
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} m ${Math.round(s - m * 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} m`;
}

export function fmtAgo(s) {
  if (s < 1.5) return 'now';
  if (s < 90) return Math.round(s) + ' s ago';
  if (s < 5400) return Math.round(s / 60) + ' min ago';
  return (s / 3600).toFixed(1) + ' h ago';
}

export const pct = (v) => Math.round(v * 100) + ' %';

/** Tiny hyperscript: h('div.cls', {attrs}, ...children). Text is always set as text, never HTML. */
export function h(sel, attrs, ...kids) {
  const [tag, ...cls] = sel.split('.');
  const el = document.createElement(tag || 'div');
  if (cls.length) el.className = cls.join(' ');
  if (attrs != null && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    kids.unshift(attrs); attrs = null;
  }
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}

const SVGNS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}

const fx = (v, n) => (typeof v === 'number' ? v.toFixed(n) : '?');

/** One-line plain-language summary of a LogEvent (falls back to the raw line). */
export function logSummary(ev) {
  const d = ev.data || {};
  const seeing = (d.seeing || []).map((s) => s.slot).join(', ');
  switch (ev.kind) {
    case 'sob_add': return `Started seeing ${d.base ?? '?'}${seeing ? ` (now: ${seeing})` : ''}`;
    case 'sob_drop': return `Lost sight of ${d.base ?? '?'}${seeing ? ` (still: ${seeing})` : ' (no bases left)'}`;
    case 'tracking_reset': return `Tracking reset: ${String(d.reason ?? 'other').replace(/_/g, ' ')}${d.ms != null ? ` after ${d.ms} ms` : ''}`;
    case 'imu_off_scale': return 'IMU went off scale';
    case 'backfacing_hits': return `Rejected ${d.rejected ?? '?'} optical hits (${d.backfacing ?? 0} backfacing, ${d.non_clustered ?? 0} non-clustered)`;
    case 'tilt_update': return `Base ${d.base ?? '?'} tilt changed by ${fx(d.delta_deg, 2)}°`;
    case 'universe_tilt': return `Universe tilt for ${d.base ?? '?'}: pitch ${fx(d.pitch, 2)}°, roll ${fx(d.roll, 2)}°`;
    case 'base_calibrated': return `Base ${d.base ?? '?'} calibrated (pitch ${fx(d.pitch, 1)}°, roll ${fx(d.roll, 1)}°)`;
    case 'base_bootstrapped': return `Base ${d.base ?? '?'} bootstrapped at ${fx(d.distance, 2)} m (${d.mode ?? '?'})`;
    case 'base_secondary': return `Base ${d.base ?? '?'} added as secondary at ${fx(d.distance, 2)} m`;
    case 'base_relationship': return `Bases ${d.a ?? '?'} ↔ ${d.b ?? '?'}: ${fx(d.distance, 2)} m, ${fx(d.angle, 1)}°`;
    case 'base_add_failed': return `Could not add base ${d.base ?? '?'}: ${d.reason ?? ''}`;
    case 'receiver_connected': return `Receiver ${d.dongle ?? '?'} connected`;
    case 'receiver_disconnected': return `Receiver ${d.dongle ?? '?'} disconnected`;
    case 'wireless_disconnected': return `Wireless link dropped on dongle ${d.dongle ?? '?'}`;
    case 'hid_error': return `USB HID error on ${d.interface ?? '?'} interface`;
    case 'dongle_enumerated': return `${d.kind ?? 'Dongle'} ${d.dongle ?? ''} enumerated`;
    case 'device_standby': return 'Device went to standby';
    case 'imu_calibration': return `IMU calibration, accel bias ${fx(d.accel_bias, 3)}`;
    case 'sync_assert': return `Sync assert: ${d.text ?? ''}`;
    case 'universe_changed': return `Universe changed to ${d.universe ?? '?'}`;
    case 'server_start': return `vrserver started${d.version ? ' (' + d.version + ')' : ''}`;
    default: return ev.raw || d.text || String(ev.kind);
  }
}

/** Log kinds worth a tick on the timeline (anything Warning/Error is included as well). */
export const NOTABLE_LOG = new Set([
  'sob_drop', 'tracking_reset', 'imu_off_scale', 'backfacing_hits', 'tilt_update', 'universe_tilt',
  'base_add_failed', 'receiver_disconnected', 'wireless_disconnected', 'hid_error', 'device_standby',
  'sync_assert', 'universe_changed', 'server_start',
]);
export const isNotableLog = (ev) => ev.level === 'Warning' || ev.level === 'Error' || NOTABLE_LOG.has(ev.kind);

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
