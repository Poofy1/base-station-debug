// Range chart: tracking quality against distance to the base stations.
// Data: GET /api/range (docs/protocol.md). One measure per view, one y-axis.
import { h, svg, RANGE_WARN_M } from './util.js';
import { showTip, hideTip } from './panels.js';

// Categorical slots in fixed order (validated set for dark surfaces); a base keeps
// its colour for the session, whatever else is on screen.
const SLOTS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];
const NEUTRAL = '#c9c8c0';
const M = { l: 46, r: 24, t: 8, b: 30 };

const VIEWS = {
  jitter: {
    title: 'Jitter by distance to each base station',
    unit: 'mm', value: (r) => r.jitter_mm, perBase: true,
    fmt: (v) => v.toFixed(v < 10 ? 2 : 1) + ' mm',
    hint: 'Position noise normally rises steadily with distance. A base whose line climbs faster than the others, or stops short, is weak, dirty or badly aimed.',
  },
  loss: {
    title: 'Tracking loss by distance to the nearest base station',
    unit: '%', value: (r) => (1 - r.ok_frac) * 100, perBase: false,
    fmt: (v) => v.toFixed(v < 10 ? 1 : 0) + ' %',
    hint: 'Share of time without a valid pose. Loss that only appears at long range means that part of the room is too far from the bases.',
  },
};

export class RangeChart {
  constructor(root, { nameOf } = {}) {
    this.root = root;
    this.nameOf = nameOf || ((s) => s);
    this.view = 'jitter';
    this.data = null;
    this.colorOf = new Map();           // base serial -> slot colour, assigned once
    this.root.addEventListener('pointerleave', () => { hideTip(); this._cross?.setAttribute('visibility', 'hidden'); });
    new ResizeObserver(() => this.render()).observe(root);
  }

  setView(v) { if (VIEWS[v]) { this.view = v; this.render(); } }
  setData(d) { this.data = d; this.render(); }

  _series() {
    const view = VIEWS[this.view], all = this.data?.series || {};
    const nearest = this.data?.nearest_key || 'nearest';
    const keys = Object.keys(all).filter((k) => (k === nearest) !== view.perBase).sort();
    return keys.map((key) => {
      if (key !== nearest && !this.colorOf.has(key)) this.colorOf.set(key, SLOTS[this.colorOf.size % SLOTS.length]);
      const pts = all[key].map((r) => ({ d: r.d, v: view.value(r), n: r.n })).filter((p) => p.v != null);
      return { key, name: key === nearest ? 'Nearest base' : this.nameOf(key), color: key === nearest ? NEUTRAL : this.colorOf.get(key), pts };
    }).filter((s) => s.pts.length).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }

  render() {
    const root = this.root, view = VIEWS[this.view];
    const W = root.clientWidth, H = root.clientHeight;
    if (!W || !H || root.hidden) return;
    const series = this._series();
    root.replaceChildren();
    const head = h('div.range-head', h('strong', `${view.title} (${view.unit})`),
      series.length > 1 ? h('span.range-legend', ...series.map((s) => h('span', h('i', { style: { background: s.color } }), s.name))) : null);
    root.append(head);
    if (!series.length) {
      root.append(h('p.range-empty', 'Not enough data yet. Move the trackers around the whole play space, including the far corners; each distance needs a few seconds of samples.'));
      return;
    }
    const ph = Math.max(60, H - head.offsetHeight - 22);
    const maxD = Math.max(7, Math.ceil(Math.max(...series.flatMap((s) => s.pts.map((p) => p.d))) + 0.5));
    const rawMax = Math.max(...series.flatMap((s) => s.pts.map((p) => p.v)), view.unit === '%' ? 5 : 0.2);
    const step = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50].find((v) => rawMax / v <= 4) || 50;
    const maxV = Math.ceil(rawMax / step) * step;
    const X = (d) => M.l + (d / maxD) * (W - M.l - M.r);
    const Y = (v) => M.t + (1 - v / maxV) * (ph - M.t - M.b);

    const g = svg('svg', { width: W, height: ph, role: 'img', 'aria-label': view.title });
    for (let v = 0; v <= maxV + 1e-9; v += step) {
      g.append(svg('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: 'rg-grid' }),
        svg('text', { x: M.l - 8, y: Y(v) + 4, 'text-anchor': 'end', class: 'rg-tick' }, String(+v.toFixed(2))));
    }
    for (let d = 0; d <= maxD; d += maxD > 9 ? 2 : 1) {
      g.append(svg('text', { x: X(d), y: ph - M.b + 16, 'text-anchor': 'middle', class: 'rg-tick' }, d + ' m'));
    }
    const warn = this.data?.warn_m || RANGE_WARN_M;
    if (warn < maxD) {
      g.append(svg('rect', { x: X(warn), y: M.t, width: X(maxD) - X(warn), height: Y(0) - M.t, class: 'rg-far' }),
        svg('text', { x: X(warn) + 6, y: M.t + 12, class: 'rg-note' }, `long range (> ${warn} m)`));
    }
    for (const s of series) {
      if (s.pts.length > 1) {
        g.append(svg('polyline', { points: s.pts.map((p) => `${X(p.d).toFixed(1)},${Y(p.v).toFixed(1)}`).join(' '), class: 'rg-line', stroke: s.color }));
      }
      for (const p of s.pts) g.append(svg('circle', { cx: X(p.d), cy: Y(p.v), r: 4, fill: s.color, class: 'rg-dot' }));
    }
    this._cross = svg('line', { y1: M.t, y2: Y(0), class: 'rg-cross', visibility: 'hidden' });
    g.append(this._cross);
    g.addEventListener('pointermove', (e) => {
      const rect = g.getBoundingClientRect();
      const d = ((e.clientX - rect.left - M.l) / (W - M.l - M.r)) * maxD;
      const bin = this.data.bin || 0.5;
      const centre = (Math.floor(d / bin) + 0.5) * bin;
      const rows = series.map((s) => ({ s, p: s.pts.find((p) => Math.abs(p.d - centre) < 1e-6) })).filter((r) => r.p);
      if (!rows.length || d < 0 || d > maxD) { hideTip(); this._cross.setAttribute('visibility', 'hidden'); return; }
      this._cross.setAttribute('x1', X(centre)); this._cross.setAttribute('x2', X(centre));
      this._cross.setAttribute('visibility', 'visible');
      showTip([
        h('strong', `${(centre - bin / 2).toFixed(1)}–${(centre + bin / 2).toFixed(1)} m from base`),
        ...rows.map(({ s, p }) => h('div.rg-tiprow', h('i', { style: { background: s.color } }), `${s.name}: `, h('b', view.fmt(p.v)), h('span.tip-sub', ` · ${Math.round(p.n / 10)} s of data`))),
      ], e.clientX, e.clientY);
    });
    root.append(g, h('p.range-hint', view.hint));
  }
}
