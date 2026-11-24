// Timeline: one lane per device, incidents as marks, notable log events as ticks.
// Mark shape encodes the incident group so it does not depend on colour:
//   tall bar = tracking lost, low bar = tracking quality, diamond = room geometry.

import { GROUPS, kindInfo, causeLabel, isNotableLog, logSummary, fmtClock, fmtDuration, pct, h, svg, clamp } from './util.js';
import { showTip, hideTip } from './panels.js';

const LANE = 34, LABEL_W = 128, AXIS_H = 26, PAD_R = 14;
const LOG_COLOR = { Info: '#898781', Warning: '#fab219', Error: '#ff6b5e' };

export class Timeline {
  constructor(root, { nameOf, onSelect }) {
    this.root = root;
    this.nameOf = nameOf;
    this.onSelect = onSelect;
    this.windowSec = 300;
    this.lanes = [];            // [{serial, label}]
    this.incidents = [];
    this.log = [];
    this.selectedId = null;
    this.now = Date.now() / 1000;
    this.svg = svg('svg', { class: 'timeline-svg', role: 'img', 'aria-label': 'Incidents and log events per device over time' });
    root.append(this.svg);
    new ResizeObserver(() => this.render()).observe(root);
    root.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.shiftKey && Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;
      e.preventDefault();
      this.setWindow(clamp(this.windowSec * (e.deltaY > 0 ? 1.25 : 0.8), 20, 4 * 3600));
      this.onWindowChange?.(this.windowSec);
    }, { passive: false });
    root.addEventListener('pointerleave', hideTip);
  }

  setWindow(sec) { this.windowSec = sec; this.render(); }
  setData({ lanes, incidents, log, now, selectedId }) {
    if (lanes) this.lanes = lanes;
    if (incidents) this.incidents = incidents;
    if (log) this.log = log;
    if (now) this.now = now;
    if (selectedId !== undefined) this.selectedId = selectedId;
  }

  render() {
    const W = this.root.clientWidth;
    if (!W) return;
    const t1 = this.now, t0 = t1 - this.windowSec;
    const narrow = W < 520;
    const labelW = narrow ? 92 : LABEL_W;
    const plotW = Math.max(40, W - labelW - PAD_R);
    const X = (t) => labelW + ((t - t0) / (t1 - t0)) * plotW;

    // lanes: known devices first, then anything else that has events in the window
    const lanes = this.lanes.map((l) => ({ ...l }));
    const index = new Map(lanes.map((l, i) => [l.serial, i]));
    const laneOf = (serial) => {
      const key = serial && index.has(serial) ? serial : (serial && /^LHR-/.test(serial) ? serial : '');
      if (!index.has(key)) { index.set(key, lanes.length); lanes.push({ serial: key, label: key ? this.nameOf(key) : 'bases / system' }); }
      return index.get(key);
    };
    const incs = this.incidents.filter((i) => (i.t_end ?? i.t_start) >= t0 && i.t_start <= t1);
    const logs = this.log.filter((e) => e.t >= t0 && e.t <= t1 && isNotableLog(e));
    for (const i of incs) i._lane = laneOf(i.serial);
    for (const e of logs) e._lane = laneOf(e.serial);

    const H = Math.max(1, lanes.length) * LANE + AXIS_H + 4;
    const s = this.svg;
    s.setAttribute('width', W); s.setAttribute('height', H); s.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const out = [];

    // lanes
    lanes.forEach((l, i) => {
      const y = i * LANE;
      if (i % 2 === 0) out.push(svg('rect', { x: 0, y, width: W, height: LANE, class: 'tl-band' }));
      out.push(svg('text', { x: 10, y: y + LANE / 2 + 5, class: 'tl-lane' }, l.label.length > (narrow ? 10 : 14) ? l.label.slice(0, narrow ? 9 : 13) + '…' : l.label));
      out.push(svg('line', { x1: labelW, x2: W - PAD_R, y1: y + LANE - 9.5, y2: y + LANE - 9.5, class: 'tl-base' }));
    });
    if (!lanes.length) out.push(svg('text', { x: 10, y: 22, class: 'tl-lane' }, 'No devices yet'));

    // time axis
    const steps = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    const step = steps.find((v) => this.windowSec / v <= (narrow ? 4 : 8)) || 3600;
    const axisY = lanes.length * LANE;
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const x = X(t);
      out.push(svg('line', { x1: x, x2: x, y1: 0, y2: axisY + 5, class: 'tl-grid' }));
      if (x > labelW + 24 && x < W - 70) out.push(svg('text', { x, y: axisY + 20, class: 'tl-tick', 'text-anchor': 'middle' }, step < 60 ? fmtClock(t) : fmtClock(t).slice(0, 5)));
    }
    out.push(svg('text', { x: W - PAD_R, y: axisY + 20, class: 'tl-tick now', 'text-anchor': 'end' }, 'now'));
    out.push(svg('line', { x1: W - PAD_R, x2: W - PAD_R, y1: 0, y2: axisY + 5, class: 'tl-now' }));

    // log ticks (under the incident marks)
    for (const e of logs) {
      const x = X(e.t), y = e._lane * LANE + LANE - 9;
      const g = svg('g', { class: 'tl-log' },
        svg('rect', { x: x - 1, y, width: 2, height: 7, fill: LOG_COLOR[e.level] || LOG_COLOR.Info }),
        svg('rect', { x: x - 5, y: y - 3, width: 10, height: 12, fill: 'transparent' }));
      g.addEventListener('pointermove', (ev) => showTip([
        h('strong', `${String(e.kind).replace(/_/g, ' ')} · ${e.serial ? this.nameOf(e.serial) : 'system'}`),
        h('div', logSummary(e)),
        h('div.tip-sub', `${fmtClock(e.t)} · ${e.level}`)], ev.clientX, ev.clientY));
      g.addEventListener('pointerleave', hideTip);
      out.push(g);
    }

    // incident marks
    for (const inc of incs) {
      const k = kindInfo(inc.kind), color = GROUPS[k.group].color;
      const xa = Math.max(labelW, X(inc.t_start)), xb = Math.min(W - PAD_R, X(inc.t_end ?? inc.t_start));
      const w = Math.max(7, xb - xa), x = Math.min(xa, W - PAD_R - w);
      const top = inc._lane * LANE + 4, full = LANE - 15;
      const sel = inc.id === this.selectedId;
      let mark;
      if (k.group === 'geometry') {
        const cx = x + w / 2, cy = top + full / 2, r = full / 2;
        mark = svg('path', { d: `M${cx},${cy - r} L${cx + r},${cy} L${cx},${cy + r} L${cx - r},${cy} Z`, fill: color });
      } else {
        const hgt = k.group === 'loss' ? full : Math.round(full * 0.5);
        mark = svg('rect', { x, y: top + full - hgt, width: w, height: hgt, rx: 2.5, fill: color });
      }
      const g = svg('g', { class: 'tl-inc' + (sel ? ' selected' : ''), tabindex: 0, role: 'button', 'aria-label': `${k.label} on ${this.nameOf(inc.serial)} at ${fmtClock(inc.t_start)}` },
        sel ? svg('rect', { x: x - 4, y: top - 3, width: w + 8, height: full + 6, rx: 4, class: 'tl-sel' }) : null,
        mark,
        w > 46 && k.group !== 'geometry' ? svg('text', { x: x + w / 2, y: top + full - 4, 'text-anchor': 'middle', class: 'tl-code' }, k.code) : null,
        svg('rect', { x: x - 3, y: top - 2, width: w + 6, height: full + 4, fill: 'transparent' }));
      const top2 = Object.entries(inc.causes || {}).sort((a, b) => b[1] - a[1])[0];
      g.addEventListener('pointermove', (ev) => showTip([
        h('strong', `${k.label} · ${this.nameOf(inc.serial)}`),
        h('div', `${fmtClock(inc.t_start)} · ${fmtDuration(Math.max(0, (inc.t_end ?? inc.t_start) - inc.t_start))}`),
        top2 ? h('div', `Likely ${causeLabel(top2[0]).toLowerCase()} (${pct(top2[1])})`) : null,
        h('div.tip-sub', 'Click for details')].filter(Boolean), ev.clientX, ev.clientY));
      g.addEventListener('pointerleave', hideTip);
      g.addEventListener('click', () => this.onSelect(inc));
      g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); this.onSelect(inc); } });
      out.push(g);
    }
    s.replaceChildren(...out);
  }
}
