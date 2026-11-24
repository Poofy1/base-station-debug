// DOM panels: device cards, diagnosis (findings), incident detail, log tail, tooltip.

import { STATES, GROUPS, trackingState, kindInfo, causeLabel, resultName, logSummary, fmtClock, fmtDuration, pct, h, clamp, dist3, rangeLevel } from './util.js';

// ---------------------------------------------------------------- tooltip
let tipEl = null;
export function showTip(content, x, y) {
  if (!tipEl) { tipEl = h('div.tip', { role: 'tooltip' }); document.body.append(tipEl); }
  tipEl.replaceChildren(...[].concat(content));
  tipEl.hidden = false;
  const r = tipEl.getBoundingClientRect();
  const px = clamp(x + 14, 6, window.innerWidth - r.width - 6);
  const py = y + 18 + r.height > window.innerHeight ? y - r.height - 10 : y + 18;
  tipEl.style.left = px + 'px';
  tipEl.style.top = Math.max(6, py) + 'px';
}
export function hideTip() { if (tipEl) tipEl.hidden = true; }

export const stateBadge = (state) => {
  const st = STATES[state];
  return h('span.state', { dataset: { state } }, h('span.icon', { 'aria-hidden': 'true' }, st.icon), st.label);
};

// ---------------------------------------------------------------- device cards
const SPARK_N = 600;            // 60 s at 10 Hz

export class DeviceCards {
  constructor(root, { onSelectDevice } = {}) {
    this.root = root;
    this.cards = new Map();
    this.empty = h('p.empty', 'No tracked devices yet.');
    root.append(this.empty);
    this.onSelectDevice = onSelectDevice || (() => {});
    this._lastDraw = 0;
  }

  /** Called for every state message; DOM text is refreshed at ~4 Hz, sparklines at ~2 Hz. */
  update(devices, bases) {
    const now = performance.now();
    const seen = new Set();
    for (const d of devices) {
      seen.add(d.serial);
      let c = this.cards.get(d.serial);
      if (!c) { c = this._make(d); this.cards.set(d.serial, c); this.root.append(c.el); }
      const state = trackingState(d);
      c.hist.push({ jit: state === 'ok' || state === 'degraded' ? d.jitter_mm : null, state });
      if (c.hist.length > SPARK_N) c.hist.shift();
      c.latest = d; c.state = state;
    }
    for (const [serial, c] of this.cards) if (!seen.has(serial)) { c.el.remove(); this.cards.delete(serial); }
    this.empty.hidden = this.cards.size > 0;
    if (now - this._lastDraw < 240) return;
    this._lastDraw = now;
    this._bases = bases || [];
    this._spark = !this._spark;
    const slots = (bases || []).map((b) => (b.channel != null ? `S-${b.channel}` : b.serial)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    for (const c of this.cards.values()) {
      this._render(c, slots);
      if (this._spark) this._drawSpark(c);
    }
  }

  _make(d) {
    const f = {};
    const metric = (key, label) => h('div.metric', h('dt', label), (f[key] = h('dd', '–')));
    const el = h('article.device-card', { tabindex: '0', title: 'Click to filter the heatmap and log to this device' },
      h('header',
        h('div.device-title', (f.label = h('h3', d.label || d.serial)), (f.sub = h('p.sub'))),
        (f.badge = h('span.state'))),
      (f.alert = h('p.device-alert', { hidden: true })),
      h('dl.metrics',
        metric('rate', 'Rate'), metric('jitter', 'Jitter'),
        metric('battery', 'Battery'), metric('dongle', 'Dongle'),
        h('div.metric.wide', h('dt', 'Bases'), (f.bases = h('dd.bases')))),
      h('div.spark', (f.canvas = h('canvas', { 'aria-hidden': 'true', title: 'Line: jitter over the last 60 s. Strip underneath: tracking state over the same time.' })), (f.sparkCap = h('p.spark-cap'))));
    el.addEventListener('click', () => this.onSelectDevice(d.serial));
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.onSelectDevice(d.serial); });
    return { el, f, hist: [], latest: d, state: null, shown: {} };
  }

  _render(c, slots) {
    const d = c.latest, f = c.f, st = STATES[c.state];
    const set = (key, node, text) => { if (c.shown[key] !== text) { c.shown[key] = text; node.textContent = text; } };
    set('label', f.label, d.label || d.serial);
    set('sub', f.sub, [d.model, d.serial].filter(Boolean).join(' · '));
    const badgeText = st.icon + ' ' + st.label;
    if (c.shown.badge !== badgeText) {
      c.shown.badge = badgeText;
      f.badge.dataset.state = c.state;
      f.badge.replaceChildren(h('span.icon', { 'aria-hidden': 'true' }, st.icon), st.label);
      c.el.dataset.state = c.state;
    }
    let alert = '';
    if (d.incident) alert = `▲ ${kindInfo(d.incident).label} in progress`;
    else if (c.state === 'lost') alert = `✖ No valid pose (${resultName(d.result)})`;
    else if (c.state === 'degraded') alert = d.result !== 200 ? `▲ ${resultName(d.result)}` : `▲ Only ${pct(d.ok_frac ?? 0)} of recent samples OK`;
    if (c.shown.alert !== alert) {
      c.shown.alert = alert;
      f.alert.hidden = true;
      c.el.classList.toggle('has-incident', !!d.incident);
    }
    // An open incident replaces the model line instead of adding a row, so every
    // card keeps the same height whatever its state.
    if (alert) { f.sub.textContent = alert; c.shown.sub = alert; }
    f.sub.classList.toggle('is-alert', !!alert);
    const tracked = d.connected;
    set('rate', f.rate, tracked && d.update_hz != null ? `${Math.round(d.update_hz)} Hz` : '–');
    set('jitter', f.jitter, tracked && d.ok && d.jitter_mm != null ? `${d.jitter_mm.toFixed(d.jitter_mm < 10 ? 1 : 0)} mm` : '–');
    let bat = d.class === 'hmd' ? 'wired' : '–';
    if (d.battery != null) {
      bat = pct(d.battery);
      if (d.charging) bat += ' § ⚡ charging';
      else if (d.battery < 0.2) bat += ' § ▲ low';
    }
    if (c.shown.battery !== bat) {
      c.shown.battery = bat;
      const [num, ...rest] = bat.split(' § ');
      f.battery.replaceChildren(...[
        d.battery != null ? h('span.batt', { dataset: { low: d.battery < 0.2 && !d.charging ? '1' : '0' } },
          h('span', { style: { width: Math.round(clamp(d.battery, 0, 1) * 100) + '%' } })) : null,
        num,
        rest.length ? h('span.batt-note', rest[0]) : null,
      ].filter((n) => n != null));
    }
    set('dongle', f.dongle, d.dongle || (d.class === 'hmd' ? 'tethered' : '–'));
    const seenSet = new Set(d.bases_seen || []);
    const all = [...new Set([...slots, ...seenSet])];
    const distOf = new Map();
    for (const b of this._bases || []) {
      if (b.valid === false || !d.connected) continue;
      distOf.set(b.channel != null ? `S-${b.channel}` : b.serial, dist3(d.pos, b.pos));
    }
    const basesKey = all.map((s) => (seenSet.has(s) ? '+' : '-') + s + (distOf.has(s) ? distOf.get(s).toFixed(1) : '')).join(' ');
    if (c.shown.bases !== basesKey) {
      c.shown.bases = basesKey;
      f.bases.replaceChildren(
        ...all.map((s) => {
          const m = distOf.get(s);
          const level = m == null ? 'ok' : rangeLevel(m);
          const where = m == null ? '' : `, ${m.toFixed(1)} m away${level === 'ok' ? '' : level === 'warn' ? ' (long range: expect more jitter)' : ' (at the edge of its reach)'}`;
          return h('span.base-chip', { dataset: { on: seenSet.has(s) ? '1' : '0', range: level }, title: `${s}: ${seenSet.has(s) ? 'visible' : 'not visible'}${where}` },
            s, m == null ? null : h('span.base-dist', `${m.toFixed(1)} m`));
        }),
        h('span.base-count', all.length ? `${seenSet.size}/${all.length}` : 'none'));
    }
  }

  _drawSpark(c) {
    const cv = c.f.canvas;
    const w = cv.clientWidth, hgt = cv.clientHeight;
    if (!w || !hgt) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, hgt);
    const strip = 6, plotH = hgt - strip - 4;
    let max = 2;
    for (const p of c.hist) if (p.jit != null && p.jit > max) max = p.jit;
    const x = (i) => w - (c.hist.length - 1 - i) * (w / (SPARK_N - 1));
    // baseline
    g.strokeStyle = '#383835'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, plotH + 0.5); g.lineTo(w, plotH + 0.5); g.stroke();
    // jitter line (gaps where there is no pose)
    g.strokeStyle = '#6da7ec'; g.lineWidth = 2; g.lineJoin = 'round';
    g.beginPath();
    let pen = false;
    c.hist.forEach((p, i) => {
      if (p.jit == null) { pen = false; return; }
      const y = plotH - 1 - (p.jit / max) * (plotH - 3);
      if (pen) g.lineTo(x(i), y); else { g.moveTo(x(i), y); pen = true; }
    });
    g.stroke();
    // tracking-state strip: runs of equal state
    let i = 0;
    while (i < c.hist.length) {
      let j = i;
      while (j + 1 < c.hist.length && c.hist[j + 1].state === c.hist[i].state) j++;
      const s = c.hist[i].state;
      g.fillStyle = s === 'ok' ? '#1f5a2a' : STATES[s].color;
      const x0 = x(i) - 0.5, x1 = x(j) + 0.5;
      g.fillRect(x0, hgt - strip, Math.max(1.5, x1 - x0), strip);
      i = j + 1;
    }
    const cap = `Jitter, last 60 s · peak ${max > 2 ? max.toFixed(1) : '< 2'} mm`;
    if (c.shown.cap !== cap) { c.shown.cap = cap; c.f.sparkCap.textContent = cap; }
  }
}

// ---------------------------------------------------------------- diagnosis
export function renderFindings(root, findings, { nameOf, waiting, incidentCount }) {
  root.replaceChildren();
  if (!findings || !findings.length) {
    root.append(h('div.empty-block',
      h('p.empty-title', waiting ? 'Nothing to diagnose yet' : (incidentCount ? 'No conclusion yet' : '● No tracking problems found so far')),
      h('p', waiting
        ? 'Findings appear here once SteamVR is running and devices are being tracked.'
        : (incidentCount
          ? `${incidentCount} incident${incidentCount === 1 ? '' : 's'} recorded, but not enough evidence to name a cause. Keep moving through the play space.`
          : 'Move every tracker through the whole play space, including the floor and the corners, so weak spots can show up.'))));
    return;
  }
  findings.forEach((f, i) => {
    const score = clamp(f.score ?? 0, 0, 1);
    const level = score >= 0.66 ? 'high' : score >= 0.33 ? 'medium' : 'low';
    const chips = [
      ...(f.devices || []).map((s) => h('span.chip', { title: s }, nameOf(s))),
      ...(f.bases || []).map((s) => h('span.chip.base', { title: s }, 'base ' + nameOf(s))),
    ];
    root.append(h('article.finding', { dataset: { level } },
      h('header',
        h('span.rank', { 'aria-label': `Rank ${i + 1}` }, String(i + 1)),
        h('h3', f.title)),
      h('div.finding-meta',
        h('span.cause', causeLabel(f.cause)),
        h('span.conf', { title: 'Confidence that this is a real cause of the incidents seen' },
          h('span.conf-bar', h('span', { style: { width: Math.round(score * 100) + '%' } })),
          h('span.conf-num', `${Math.round(score * 100)} % · ${level} confidence`)),
        f.incident_count ? h('span.count', `${f.incident_count} incident${f.incident_count === 1 ? '' : 's'}`) : null),
      f.detail ? h('p.detail', f.detail) : null,
      f.fix ? h('p.fix', h('strong', 'Try: '), f.fix) : null,
      chips.length ? h('p.affected', h('span.affected-label', 'Affects'), chips) : null));
  });
}

// ---------------------------------------------------------------- incident detail
export function renderIncidentDetail(root, inc, { nameOf, onClose }) {
  root.replaceChildren();
  root.hidden = !inc;
  if (!inc) return;
  const k = kindInfo(inc.kind);
  const causes = Object.entries(inc.causes || {}).sort((a, b) => b[1] - a[1]);
  const metrics = Object.entries(inc.metrics || {});
  const dur = (inc.t_end ?? inc.t_start) - inc.t_start;
  const fmtMetric = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 1 ? 3 : 2)) : String(v));
  root.append(
    h('header',
      h('div',
        h('p.eyebrow', `Incident #${inc.id}`),
        h('h3', h('span.kind-dot', { style: { background: GROUPS[k.group].color } }), `${k.label} · ${nameOf(inc.serial)}`)),
      h('button.icon-btn', { type: 'button', 'aria-label': 'Close incident detail', onclick: onClose }, '✕')),
    h('dl.kv',
      h('dt', 'When'), h('dd', `${fmtClock(inc.t_start)}, lasted ${fmtDuration(Math.max(0, dur))}`),
      h('dt', 'Where'), h('dd', inc.pos ? `x ${inc.pos[0].toFixed(2)}, y ${inc.pos[1].toFixed(2)}, z ${inc.pos[2].toFixed(2)} m` : 'unknown'),
      h('dt', 'Severity'), h('dd', h('span.conf-bar', h('span', { style: { width: Math.round(clamp(inc.severity ?? 0, 0, 1) * 100) + '%' } })), ' ' + pct(inc.severity ?? 0))),
    h('h4', 'Likely causes'),
    causes.length
      ? h('ul.causes', causes.map(([c, v]) => h('li',
        h('span.cause-name', causeLabel(c)),
        h('span.conf-bar', h('span', { style: { width: Math.round(clamp(v, 0, 1) * 100) + '%' } })),
        h('span.cause-num', pct(v)))))
      : h('p.muted', 'Not attributed yet.'),
    h('h4', 'Evidence'),
    (inc.evidence || []).length ? h('ul.evidence', inc.evidence.map((e) => h('li', e))) : h('p.muted', 'No evidence recorded.'),
    metrics.length ? h('h4', 'Metrics') : null,
    metrics.length ? h('dl.kv.mono', metrics.flatMap(([key, v]) => [h('dt', key), h('dd', fmtMetric(v))])) : null);
}

// ---------------------------------------------------------------- log tail
const LEVEL_ICON = { Info: '·', Warning: '▲', Error: '✖' };

export class LogTail {
  constructor(root, deviceSel, kindSel, { nameOf }) {
    this.root = root; this.deviceSel = deviceSel; this.kindSel = kindSel; this.nameOf = nameOf;
    this.events = [];
    this.kinds = new Set();
    this.serials = new Set();
    this._dirty = false;
    deviceSel.addEventListener('change', () => this.render());
    kindSel.addEventListener('change', () => this.render());
  }

  /** events: newest-first array owned by the app. */
  setEvents(events) {
    this.events = events;
    let optsChanged = false;
    for (const e of events) {
      if (!this.kinds.has(e.kind)) { this.kinds.add(e.kind); optsChanged = true; }
      const s = e.serial || '';
      if (!this.serials.has(s)) { this.serials.add(s); optsChanged = true; }
    }
    if (optsChanged) this._options();
    if (!this._dirty) { this._dirty = true; requestAnimationFrame(() => { this._dirty = false; this.render(); }); }
  }

  setDevice(serial) {
    if (![...this.deviceSel.options].some((o) => o.value === serial)) return;
    this.deviceSel.value = serial; this.render();
  }

  refreshNames() { this._options(); }

  _options() {
    const keep = (sel, opts) => {
      const cur = sel.value;
      sel.replaceChildren(...opts.map(([v, t]) => h('option', { value: v }, t)));
      sel.value = opts.some(([v]) => v === cur) ? cur : opts[0][0];
    };
    const devs = [...this.serials].filter(Boolean).sort((a, b) => this.nameOf(a).localeCompare(this.nameOf(b)));
    keep(this.deviceSel, [['*', 'All devices'], ...devs.map((s) => [s, this.nameOf(s)]), ...(this.serials.has('') ? [['', 'System / bases']] : [])]);
    keep(this.kindSel, [['*', 'All kinds'], ['!', 'Warnings and errors'], ...[...this.kinds].sort().map((k) => [k, k.replace(/_/g, ' ')])]);
  }

  render() {
    const dev = this.deviceSel.value, kind = this.kindSel.value;
    const rows = [];
    for (const e of this.events) {
      if (dev !== '*' && (e.serial || '') !== dev) continue;
      if (kind === '!' ? e.level === 'Info' : (kind !== '*' && e.kind !== kind)) continue;
      rows.push(h('li.log-row', { dataset: { level: e.level || 'Info' }, title: e.raw || '' },
        h('time', fmtClock(e.t)),
        h('span.lvl', { 'aria-label': e.level }, LEVEL_ICON[e.level] || '·'),
        h('span.who', e.serial ? this.nameOf(e.serial) : 'system'),
        h('span.msg', h('span.kind', String(e.kind).replace(/_/g, ' ')), logSummary(e))));
      if (rows.length >= 150) break;
    }
    this.root.replaceChildren(...(rows.length ? rows : [h('li.empty', this.events.length ? 'No log events match the filters.' : 'No lighthouse log events yet.')]));
  }
}
