// Client-side stand-in for the lhdebug backend, enabled with `?mock=1`.
//
// Implements the same interface as RealTransport in transport.js and produces the
// full protocol from docs/protocol.md: hello, 10 Hz state, incident, log, findings
// over the "socket", and /api/session, /api/heatmap, /api/incidents, /api/log,
// /api/findings, /api/lighthousedb, /api/sessions over getJSON.
//
// Extra query switches (mock only):
//   steamvr=0   SteamVR never comes up (waiting state)
//   conn=down   the backend is unreachable (reconnect loop)
//   history=N   seconds of pre-simulated history (default 300)
//
// The simulated room has three deliberate problems so every panel has something to show:
//   * a cabinet in the back-right corner that hides the feet from the bases (occlusion dropouts)
//   * a mirror on the left wall (reflections: jitter + jumps for everything near it)
//   * a right-foot tracker with a weak radio link and a nearly flat battery (stale poses, disconnects)

const VOXEL = 0.2;
const DEG = Math.PI / 180;

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- small vector / quaternion helpers (wire order for quaternions is w, x, y, z) ---
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const r3 = (v) => v.map((x) => Math.round(x * 1000) / 1000);

/** Orthonormal basis for an object whose local -Z points along `fwd`. */
function basisLooking(fwd) {
  const z = norm([-fwd[0], -fwd[1], -fwd[2]]);
  let x = cross([0, 1, 0], z);
  if (Math.hypot(...x) < 1e-6) x = [1, 0, 0];
  x = norm(x);
  const y = cross(z, x);
  return { x, y, z };
}

function quatFromBasis({ x, y, z }) {
  // rotation matrix columns are x, y, z
  const m00 = x[0], m10 = x[1], m20 = x[2];
  const m01 = y[0], m11 = y[1], m21 = y[2];
  const m02 = z[0], m12 = z[1], m22 = z[2];
  const tr = m00 + m11 + m22;
  let w, qx, qy, qz;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = s / 4; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; qx = s / 4; qy = (m01 + m10) / s; qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; qx = (m01 + m10) / s; qy = s / 4; qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = s / 4;
  }
  return [w, qx, qy, qz].map((v) => Math.round(v * 10000) / 10000);
}

// --- the simulated room ---
const BASES = [
  { serial: 'LHB-4F091D9A', channel: 1, pos: [-2.15, 2.3, -2.15] },
  { serial: 'LHB-B3316995', channel: 5, pos: [2.15, 2.3, -2.15] },
  { serial: 'LHB-7C20E5D1', channel: 9, pos: [0.1, 2.4, 2.35] },
].map((b) => {
  const basis = basisLooking(norm(sub([0, 0.7, 0], b.pos)));
  return { ...b, basis, quat: quatFromBasis(basis), id: b.serial.slice(4), slot: `S-${b.channel}` };
});

const CABINET = { min: [1.0, 0, 1.15], max: [1.9, 0.95, 2.0] };   // occluder, back-right corner
const inFootShadow = (p) => p[0] > 0.55 && p[2] > 0.35 && p[1] < 0.5;
const nearMirror = (p) => p[0] < -1.15;

const DEVICES = [
  { serial: 'LHR-A81F3D02', class: 'hmd',        model: 'Valve Index',            role: 'head',       hz: 500, dongle: null,         battery: null },
  { serial: 'LHR-5E1C77B0', class: 'controller', model: 'Knuckles Left',          role: 'left_hand',  hz: 250, dongle: '7F3A21C9D0', battery: 0.74 },
  { serial: 'LHR-0D94E6A3', class: 'controller', model: 'Knuckles Right',         role: 'right_hand', hz: 250, dongle: 'C0118E52B7', battery: 0.69 },
  { serial: 'LHR-29C03CCA', class: 'tracker',    model: 'VIVE Tracker 3.0 MV',    role: 'waist',      hz: 250, dongle: '40E4AC9C61', battery: 0.82 },
  { serial: 'LHR-6B7710F4', class: 'tracker',    model: 'VIVE Tracker 3.0 MV',    role: 'left_foot',  hz: 250, dongle: '2A6D90F1E8', battery: 0.57, charging: true },
  { serial: 'LHR-E2385B19', class: 'tracker',    model: 'VIVE Tracker 3.0 MV',    role: 'right_foot', hz: 250, dongle: '9B55D3A7C4', battery: 0.13 },
];

function segmentHitsBox(a, b, box) {
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 3; i++) {
    const d = b[i] - a[i];
    if (Math.abs(d) < 1e-9) {
      if (a[i] < box.min[i] || a[i] > box.max[i]) return false;
    } else {
      let u = (box.min[i] - a[i]) / d, v = (box.max[i] - a[i]) / d;
      if (u > v) [u, v] = [v, u];
      t0 = Math.max(t0, u); t1 = Math.min(t1, v);
      if (t0 > t1) return false;
    }
  }
  return true;
}

/** Does `base` have line of sight to point p (inside its ~150x110 degree field of view)? */
function baseSees(base, p) {
  const d = sub(p, base.pos);
  const fz = -dot(d, base.basis.z);           // distance along the base's forward axis
  if (fz <= 0.05 || fz > 7) return false;
  if (Math.abs(dot(d, base.basis.x)) > fz * Math.tan(75 * DEG)) return false;
  if (Math.abs(dot(d, base.basis.y)) > fz * Math.tan(55 * DEG)) return false;
  return !segmentHitsBox(base.pos, p, CABINET);
}

export class MockTransport {
  constructor(query) {
    this.kind = 'mock';
    this.onMessage = () => {};
    this.onStatus = () => {};
    this.q = query || new URLSearchParams();
    this.steamvr = this.q.get('steamvr') !== '0';
    this.down = this.q.get('conn') === 'down';
    this.rand = mulberry32(20261124);
    this.incidents = [];          // oldest first
    this.log = [];                // oldest first
    this.cells = new Map();       // serial -> Map(cellKey -> stats)
    this.nextId = 1;
    this.sim = DEVICES.map((d) => ({
      info: d, battery: d.battery, episode: null, okFrac: 1, lastPos: null, lastT: 0,
      nextRadio: 0, seen: [], connectedAt: 0,
    }));
    this.started = Date.now() / 1000;
    this.session = {
      id: 'mock-' + new Date().toISOString().slice(0, 19).replace(/[-:T]/g, ''),
      mode: 'sim', started: this.started, steamvr: this.steamvr, voxel: VOXEL,
      recording: true, paused_at: null, window_s: 900,
      play_area: {
        rect: [[1.5, 0, -1.5], [-1.5, 0, -1.5], [-1.5, 0, 1.5], [1.5, 0, 1.5]],
        bounds: [[-1.8, 0, -1.7], [-1.7, 0, 1.8], [1.8, 0, 1.9], [1.9, 0, -1.6]], height: 2.4,
      },
    };
    this.emitting = false;
    this.baseShiftDone = false;
  }

  // ---------------------------------------------------------------- interface
  start() {
    if (this.down) { this._downLoop(500); return; }
    this.onStatus({ state: 'connecting' });
    setTimeout(() => {
      const history = Math.max(0, Number(this.q.get('history') ?? 300));
      const now = Date.now() / 1000;
      this.started = now - history;
      this.session.started = this.started;
      if (this.steamvr) {
        this._boot(this.started);
        for (let t = this.started; t < now - 0.05; t += 0.1) this._step(t);
      }
      // recording controls: while stopped, only state messages get through
      const emit = this.onMessage;
      this.onMessage = (m) => {
        if (!m) return;
        if (!this.session.recording && ['incident', 'log', 'findings'].includes(m.type)) return;
        if (m.type === 'state') m.recording = this.session.recording;
        emit(m);
      };
      this.emitting = true;
      this.onStatus({ state: 'open' });
      this.onMessage({ type: 'hello', session: { ...this.session } });
      this.onMessage({ type: 'findings', findings: this._findings() });
      setInterval(() => this.onMessage(this._step(Date.now() / 1000)), 100);
      setInterval(() => this.onMessage({ type: 'findings', findings: this._findings() }), 5000);
    }, 150);
  }

  async postJSON(path, body) {
    if (this.down) throw new Error('mock backend is down');
    if (path !== '/api/control') throw new Error(`${path}: HTTP 404`);
    const s = this.session;
    if (body.action === 'stop') { s.recording = false; s.paused_at = Date.now() / 1000; }
    if (body.action === 'resume') { s.recording = true; s.paused_at = null; }
    if (body.action === 'clear') {
      this.incidents = []; this.log = []; this.cells = new Map();
      if (s.paused_at != null) s.paused_at = Date.now() / 1000;
      this.onMessage({ type: 'cleared' });
    }
    if (body.window_s != null) s.window_s = Number(body.window_s);
    this.onMessage({ type: 'session', session: { ...s } });
    return { ...s };
  }

  async getJSON(path) {
    if (this.down) throw new Error('mock backend is down');
    const url = new URL(path, 'http://mock.invalid');
    const p = url.searchParams;
    await new Promise((r) => setTimeout(r, 30));
    switch (url.pathname) {
      case '/api/session': return { ...this.session };
      case '/api/heatmap': return this._heatmap(p.get('metric') || 'dropout', p.get('serial') || null);
      case '/api/incidents': return { incidents: this.incidents.slice(-Number(p.get('limit') || 200)).reverse() };
      case '/api/log': return { events: this.log.slice(-Number(p.get('limit') || 300)).reverse() };
      case '/api/range': {
        const rows = (k, top) => Array.from({ length: top }, (_, i) => ({
          d: 0.75 + i * 0.5, n: 400 - i * 30,
          jitter_mm: +(0.15 + 0.12 * k * (0.75 + i * 0.5) ** (k > 1.5 ? 1.6 : 1)).toFixed(3),
          ok_frac: +(1 - Math.max(0, i - 6) * 0.06).toFixed(3),
        }));
        const series = { nearest: rows(1, 10) };
        BASES.forEach((b, i) => { series[b.serial] = rows(i === 2 ? 1.8 : 1 + i * 0.1, 11 - i); });
        return { bin: 0.5, nearest_key: 'nearest', warn_m: 5, series };
      }
      case '/api/findings': return { findings: this._findings() };
      case '/api/lighthousedb': return {
        bases: Object.fromEntries(BASES.map((b) => [b.serial, { serial: b.serial, channel: b.channel }])),
        universes: [{ id: '1795500000', bases: BASES.map((b) => b.serial) }],
      };
      case '/api/sessions': return { sessions: [{ id: this.session.id, mode: 'sim', started: this.started, size: 0 }] };
      default: throw new Error(`${path}: HTTP 404`);
    }
  }

  _downLoop(delay) {
    this.onStatus({ state: 'connecting' });
    setTimeout(() => {
      this.onStatus({ state: 'closed', retryIn: delay / 1000 });
      setTimeout(() => this._downLoop(Math.min(5000, Math.round(delay * 1.7))), delay);
    }, 200);
  }

  // ---------------------------------------------------------------- simulation
  _emit(msg) { if (this.emitting) this.onMessage(msg); }

  _logEvent(t, kind, serial, data, level, text) {
    const d = new Date(t * 1000);
    const stamp = d.toISOString().replace('T', ' ').slice(0, 23);
    const ev = {
      t, kind, serial: serial || null, data: data || {}, level: level || 'Info',
      raw: `${stamp} - lighthouse: ${serial ? serial + ' ' : ''}${text || kind}`,
    };
    this.log.push(ev);
    if (this.log.length > 3000) this.log.splice(0, 500);
    this._emit({ type: 'log', event: ev });
  }

  _incident(inc) {
    inc.id = this.nextId++;
    inc.pos = r3(inc.pos);
    this.incidents.push(inc);
    if (this.incidents.length > 2000) this.incidents.splice(0, 500);
    this._emit({ type: 'incident', incident: inc });
  }

  _boot(t) {
    this._logEvent(t, 'server_start', null, { version: '2.8.6' }, 'Info', 'VR server 2.8.6 starting up');
    for (const d of DEVICES) {
      if (d.dongle) this._logEvent(t + 0.4, 'dongle_enumerated', d.serial, { dongle: d.dongle, kind: 'Watchman Dongle' }, 'Info', `Watchman Dongle ${d.dongle} enumerated`);
    }
    BASES.forEach((b, i) => {
      this._logEvent(t + 1.2 + i * 0.3, 'base_calibrated', null, { base: b.id, pitch: 31.2 + i * 2.1, roll: 0.4 * (i - 1) }, 'Info',
        `Base ${b.id} calibrated, pitch ${(31.2 + i * 2.1).toFixed(1)} roll ${(0.4 * (i - 1)).toFixed(1)}`);
    });
    this._logEvent(t + 2.4, 'base_relationship', null, { a: BASES[0].id, b: BASES[1].id, distance: 4.3, angle: 88.4 }, 'Info',
      `Base relationship ${BASES[0].id} <-> ${BASES[1].id}: 4.30 m, 88.4 deg`);
  }

  /** True (noise-free) pose of every device at time t: one person wandering the play space. */
  _truth(t) {
    const cx = 1.55 * Math.sin(0.113 * t + 0.4), cz = 1.5 * Math.sin(0.171 * t + 1.3);
    const vx = 1.55 * 0.113 * Math.cos(0.113 * t + 0.4), vz = 1.5 * 0.171 * Math.cos(0.171 * t + 1.3);
    const f = norm([vx, 0, vz]);                 // facing = walking direction
    const r = [-f[2], 0, f[0]];                  // right-hand side
    const step = 3.1 * t;
    const at = (side, fwd, y) => [cx + r[0] * side + f[0] * fwd, y, cz + r[2] * side + f[2] * fwd];
    const quat = quatFromBasis(basisLooking(f));
    return {
      head: { pos: at(0, 0.05, 1.66 + 0.015 * Math.sin(2 * step)), quat },
      waist: { pos: at(0, 0, 1.0 + 0.01 * Math.sin(2 * step)), quat },
      left_hand: { pos: at(-0.32, 0.22 + 0.18 * Math.sin(step + Math.PI), 1.05 + 0.2 * Math.sin(0.7 * t)), quat },
      right_hand: { pos: at(0.32, 0.22 + 0.18 * Math.sin(step), 1.08 + 0.22 * Math.sin(0.53 * t + 2)), quat },
      left_foot: { pos: at(-0.13, 0.2 * Math.sin(step), 0.09 + 0.09 * Math.max(0, Math.cos(step))), quat },
      right_foot: { pos: at(0.13, 0.2 * Math.sin(step + Math.PI), 0.09 + 0.09 * Math.max(0, Math.cos(step + Math.PI))), quat },
    };
  }

  _step(t) {
    const R = this.rand;
    if (!this.steamvr) return { type: 'state', t, steamvr: false, devices: [], bases: [] };
    const truth = this._truth(t);
    const devices = [];

    // one base station gets nudged once, a couple of minutes into the session
    if (!this.baseShiftDone && t - this.started > 140) {
      this.baseShiftDone = true;
      const b = BASES[2];
      this._logEvent(t, 'tilt_update', null, { base: b.id, delta_deg: 0.41 }, 'Warning', `Updating tilt for ${b.id}, delta 0.41 deg`);
      this._incident({
        kind: 'base_shift', serial: b.serial, t_start: t - 0.6, t_end: t, pos: b.pos.slice(), severity: 0.35,
        metrics: { shift_m: 0.011, tilt_deg: 0.41 },
        causes: { calibration: 0.7, base_station: 0.25 },
        evidence: [`Base ${b.id} (S-${b.channel}) reported a tilt change of 0.41°`, 'No device lost tracking at the same moment'],
      });
    }

    for (const s of this.sim) {
      const d = s.info;
      const tp = truth[d.role];
      const p = tp.pos;
      const dt = s.lastT ? Math.max(0.01, t - s.lastT) : 0.1;
      const speed = s.lastPos ? Math.hypot(...sub(p, s.lastPos)) / dt : 0;
      const isFoot = d.role.endsWith('foot');
      const mirror = nearMirror(p);
      if (s.battery != null) s.battery = Math.min(1, Math.max(0.02, s.battery + (d.charging ? 2e-5 : -4e-6)));

      // ---- start a problem episode?
      if (!s.episode) {
        if (isFoot && inFootShadow(p) && R() < 0.07) {
          s.episode = { kind: 'dropout', t0: t, until: t + 0.4 + R() * 2.6, pos: p.slice(), seenBefore: s.seen.slice() };
          const lost = s.seen[0] || BASES[0].slot;
          this._logEvent(t, 'sob_drop', d.serial, { base: lost, seeing: [] }, 'Info', `Dropped ${lost}, tracking from 0 bases`);
        } else if (mirror && R() < 0.022) {
          s.episode = { kind: 'jitter', t0: t, until: t + 1 + R() * 3, pos: p.slice(), peak: 0 };
          const rej = 20 + Math.floor(R() * 90);
          this._logEvent(t, 'backfacing_hits', d.serial, { rejected: rej, backfacing: Math.floor(rej * 0.7), non_clustered: Math.ceil(rej * 0.3) },
            'Warning', `Rejected ${rej} hits: backfacing or non-clustered`);
        } else if (mirror && R() < 0.006) {
          const jump = 0.04 + R() * 0.14;
          this._countJump(d.serial, p);
          this._incident({
            kind: 'jump', serial: d.serial, t_start: t - 0.05, t_end: t, pos: p.slice(), severity: Math.min(1, jump / 0.2),
            metrics: { jump_m: +jump.toFixed(3), speed_before: +speed.toFixed(2) },
            causes: { reflection: 0.55 + R() * 0.25, occlusion: 0.12, unknown: 0.08 },
            evidence: [
              `Position jumped ${(jump * 100).toFixed(0)} cm in one frame while the pose stayed "valid"`,
              'Driver rejected backfacing hits within 2 s of the jump',
              'Other incidents cluster within 0.5 m of this spot for 4 different devices',
            ],
          });
        } else if (d.role === 'right_foot' && t > s.nextRadio) {
          s.nextRadio = t + 35 + R() * 50;
          if (R() < 0.35) {
            s.episode = { kind: 'disconnect', t0: t, until: t + 1.5 + R() * 2.5, pos: p.slice() };
            this._logEvent(t, 'wireless_disconnected', d.serial, { dongle: d.dongle, device: d.serial }, 'Warning', `Wireless device disconnected from dongle ${d.dongle}`);
          } else {
            s.episode = { kind: 'stale', t0: t, until: t + 0.8 + R() * 1.8, pos: p.slice(), minHz: 250 };
          }
        }
      }

      // ---- current readings
      const ep = s.episode;
      let connected = true, valid = true, result = 200;
      let hz = d.hz * (0.985 + R() * 0.03);
      let jit = 0.25 + R() * 0.3 + speed * 0.25;
      if (mirror) jit += 0.9 + R() * 0.9;
      let seen = BASES.filter((b) => baseSees(b, p)).map((b) => b.slot);
      if (ep) {
        if (ep.kind === 'dropout') { valid = false; result = 300; hz = 0; seen = []; }
        if (ep.kind === 'disconnect') { connected = false; valid = false; result = 1; hz = 0; seen = []; }
        if (ep.kind === 'stale') { hz = 14 + R() * 30; ep.minHz = Math.min(ep.minHz, hz); }
        if (ep.kind === 'jitter') { jit = 3.2 + R() * 4.5; ep.peak = Math.max(ep.peak, jit); }
      }
      const ok = connected && valid && result === 200;
      s.okFrac += ((ok ? 1 : 0) - s.okFrac) * 0.04;
      s.seen = seen;
      this._accumulate(d.serial, p, ok, jit, hz);

      // reported pose: noisy when tracked, frozen at the last good one when not
      let pos = s.reported || p;
      if (ok) {
        const n = jit / 1000;
        pos = [p[0] + (R() - 0.5) * n * 2, p[1] + (R() - 0.5) * n * 2, p[2] + (R() - 0.5) * n * 2];
        s.reported = pos;
      }

      devices.push({
        serial: d.serial, class: d.class, model: d.model, role: d.role, label: d.role,
        connected, valid, result, ok,
        pos: r3(pos), quat: tp.quat,
        battery: s.battery == null ? null : +s.battery.toFixed(3), charging: d.battery == null ? null : !!d.charging,
        dongle: d.dongle,
        update_hz: +hz.toFixed(1), jitter_mm: +(ok ? jit : 0).toFixed(2), jitter_deg: +(ok ? jit * 0.08 : 0).toFixed(3),
        ok_frac: +s.okFrac.toFixed(3), speed: +speed.toFixed(2),
        bases_seen: seen,
        incident: ep ? ep.kind : null,
      });

      // ---- close the episode?
      if (ep && t >= ep.until) {
        s.episode = null;
        this._closeEpisode(s, ep, t);
      }
      s.lastPos = p; s.lastT = t;
    }

    return {
      type: 'state', t, steamvr: true, devices,
      bases: BASES.map((b) => ({ serial: b.serial, channel: b.channel, pos: b.pos, quat: b.quat, connected: true, valid: true })),
    };
  }

  _closeEpisode(s, ep, t) {
    const d = s.info, R = this.rand;
    const dur = t - ep.t0;
    const base = { kind: ep.kind, serial: d.serial, t_start: ep.t0, t_end: t, pos: ep.pos };
    if (ep.kind === 'dropout') {
      if (dur > 1.2) this._logEvent(t - 0.2, 'tracking_reset', d.serial, { reason: 'no_optical', ms: Math.round(dur * 1000), text: 'no optical samples' },
        'Warning', `Resetting tracking: no optical samples for ${Math.round(dur * 1000)} ms`);
      const back = BASES[Math.floor(R() * 2)].slot;
      this._logEvent(t, 'sob_add', d.serial, { base: back, seeing: [{ slot: back, base: BASES.find((b) => b.slot === back).id }] }, 'Info', `Saw ${back}, tracking from 1 base`);
      this._incident({
        ...base, severity: Math.min(1, 0.25 + dur / 3),
        metrics: { duration_s: +dur.toFixed(2), bases_before: ep.seenBefore.length, height_m: +ep.pos[1].toFixed(2) },
        causes: { occlusion: 0.62 + R() * 0.25, base_station: 0.1, radio: 0.05 },
        evidence: [
          `Driver dropped every base (${ep.seenBefore.join(', ') || 'none visible'}) just before the pose went invalid`,
          `Device was ${(ep.pos[1] * 100).toFixed(0)} cm above the floor in a cell where predicted coverage is 0–1 bases`,
          'Waist and head kept tracking at the same time, so the bases and radio were fine',
          dur > 1.2 ? `Tracking reset after ${Math.round(dur * 1000)} ms without optical data` : 'IMU bridged the gap; no tracking reset was needed',
        ],
      });
    } else if (ep.kind === 'jitter') {
      this._incident({
        ...base, severity: Math.min(1, ep.peak / 8),
        metrics: { duration_s: +dur.toFixed(2), jitter_mm_peak: +ep.peak.toFixed(2), jitter_mm_baseline: 0.4 },
        causes: { reflection: 0.5 + R() * 0.3, base_station: 0.15, imu: 0.06 },
        evidence: [
          `Jitter rose to ${ep.peak.toFixed(1)} mm RMS (baseline 0.4 mm) with all bases still visible`,
          'Driver logged rejected backfacing / non-clustered hits at the start of the episode',
          'Same spot (x < -1.2 m) produced jitter on several devices',
        ],
      });
    } else if (ep.kind === 'stale') {
      this._incident({
        ...base, severity: Math.min(1, 0.3 + dur / 4),
        metrics: { duration_s: +dur.toFixed(2), update_hz_min: +ep.minHz.toFixed(0), update_hz_nominal: d.hz },
        causes: { radio: 0.55 + R() * 0.2, battery: 0.25, usb: 0.08 },
        evidence: [
          `Pose update rate fell to ${ep.minHz.toFixed(0)} Hz (nominal ${d.hz} Hz) while optical tracking stayed valid`,
          `Battery at ${Math.round((s.battery || 0) * 100)} %`,
          'No other device was affected; each tracker has its own dongle',
        ],
      });
    } else if (ep.kind === 'disconnect') {
      this._logEvent(t, 'receiver_connected', d.serial, { dongle: d.dongle }, 'Info', `Receiver ${d.dongle} connected`);
      this._incident({
        ...base, severity: Math.min(1, 0.5 + dur / 6),
        metrics: { duration_s: +dur.toFixed(2), battery: +(s.battery || 0).toFixed(2) },
        causes: { radio: 0.6 + R() * 0.15, battery: 0.3 },
        evidence: [
          `Driver logged "wireless device disconnected" for dongle ${d.dongle}`,
          `Battery at ${Math.round((s.battery || 0) * 100)} %, the lowest of all devices`,
          'Bases were visible right up to the disconnect, so this is not an optical problem',
        ],
      });
    }
  }

  // ---------------------------------------------------------------- heatmap
  _cell(serial, p) {
    let m = this.cells.get(serial);
    if (!m) this.cells.set(serial, (m = new Map()));
    const key = `${Math.floor(p[0] / VOXEL)},${Math.floor(p[1] / VOXEL)},${Math.floor(p[2] / VOXEL)}`;
    let c = m.get(key);
    if (!c) m.set(key, (c = { n: 0, bad: 0, okN: 0, jit: 0, jumps: 0, hz: 0 }));
    return c;
  }

  _accumulate(serial, p, ok, jit, hz) {
    const c = this._cell(serial, p);
    c.n++; c.hz += hz;
    if (ok) { c.okN++; c.jit += jit; } else c.bad++;
  }

  _countJump(serial, p) { this._cell(serial, p).jumps++; }

  _heatmap(metric, serial) {
    if (metric === 'coverage') return this._coverage();
    const merged = new Map();
    for (const [ser, m] of this.cells) {
      if (serial && ser !== serial) continue;
      for (const [key, c] of m) {
        const a = merged.get(key);
        if (!a) merged.set(key, { ...c });
        else { a.n += c.n; a.bad += c.bad; a.okN += c.okN; a.jit += c.jit; a.jumps += c.jumps; a.hz += c.hz; }
      }
    }
    const cells = [];
    let min = Infinity, max = -Infinity;
    for (const [key, c] of merged) {
      let v;
      if (metric === 'samples') v = c.n;
      else if (metric === 'dropout') v = c.bad / c.n;
      else if (metric === 'jitter') { if (!c.okN) continue; v = c.jit / c.okN; }
      else if (metric === 'jumps') v = c.jumps;
      else if (metric === 'update_rate') v = c.hz / c.n;
      else v = 0;
      v = Math.round(v * 1000) / 1000;
      const [ix, iy, iz] = key.split(',').map(Number);
      cells.push([ix, iy, iz, v, c.n]);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!cells.length) { min = 0; max = 0; }
    return { voxel: VOXEL, metric, min, max, cells };
  }

  _coverage() {
    if (!this._cov) {
      const cells = [];
      let min = Infinity, max = -Infinity;
      for (let ix = -12; ix < 12; ix++) for (let iy = 0; iy < 11; iy++) for (let iz = -12; iz < 12; iz++) {
        const p = [(ix + 0.5) * VOXEL, (iy + 0.5) * VOXEL, (iz + 0.5) * VOXEL];
        const v = this.steamvr ? BASES.filter((b) => baseSees(b, p)).length : 0;
        cells.push([ix, iy, iz, v, 1]);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      this._cov = { voxel: VOXEL, metric: 'coverage', min, max, cells: this.steamvr ? cells : [] };
    }
    return this._cov;
  }

  // ---------------------------------------------------------------- findings
  _findings() {
    const by = {};
    for (const inc of this.incidents) {
      let top = 'unknown', best = -1;
      for (const [c, v] of Object.entries(inc.causes)) if (v > best) { best = v; top = c; }
      const g = (by[top] ||= { n: 0, devices: new Set(), conf: 0 });
      g.n++; g.conf += best; g.devices.add(inc.serial);
    }
    const score = (g, k) => Math.min(0.97, (g.conf / g.n) * (1 - Math.exp(-g.n / k)));
    const out = [];
    if (by.occlusion) out.push({
      cause: 'occlusion', score: score(by.occlusion, 4),
      title: 'Foot trackers lose sight of the bases in the back-right corner',
      detail: `${by.occlusion.n} dropouts, all below 0.5 m and all inside the same 1.5 m corner (x > 0.6, z > 0.4). `
        + 'Head and waist keep tracking there, and predicted coverage near the floor in that corner is 0–1 bases: something low is blocking line of sight.',
      fix: 'Move or lower the furniture in that corner, or raise/re-aim base S-9 so it looks down into the corner. A fourth base on the right wall would also cover it.',
      devices: [...by.occlusion.devices], bases: ['LHB-7C20E5D1'], incident_count: by.occlusion.n,
    });
    if (by.reflection) out.push({
      cause: 'reflection', score: score(by.reflection, 6),
      title: 'Reflective surface near the left wall is corrupting tracking',
      detail: `${by.reflection.n} jitter and jump incidents across ${by.reflection.devices.size} devices, all at x < -1.2 m, `
        + 'with all bases visible and the driver rejecting backfacing hits each time.',
      fix: 'Cover the mirror, window or glossy surface on the left wall (a sheet or matte film is enough) and re-test.',
      devices: [...by.reflection.devices], bases: [], incident_count: by.reflection.n,
    });
    if (by.radio) out.push({
      cause: 'radio', score: score(by.radio, 3),
      title: 'right_foot has a weak radio link',
      detail: `${by.radio.n} stale-pose and disconnect incidents on one tracker only, anywhere in the room, while its optical tracking was fine. Its battery is at 13 %.`,
      fix: 'Charge the tracker, then move its dongle onto a USB extension away from USB 3 ports and the PC case, in line of sight of the play space.',
      devices: [...by.radio.devices], bases: [], incident_count: by.radio.n,
    });
    if (by.calibration) out.push({
      cause: 'calibration', score: score(by.calibration, 1) * 0.5,
      title: 'Base S-9 moved slightly during the session',
      detail: 'The driver updated the tilt of base 7C20E5D1 by 0.41°. One small shift is normal after warm-up; repeated shifts mean the mount is not rigid.',
      fix: 'Check that the base is clamped firmly and not on a surface that vibrates. Re-run room setup if this repeats.',
      devices: [], bases: ['LHB-7C20E5D1'], incident_count: by.calibration.n,
    });
    return out.sort((a, b) => b.score - a.score).map((f) => ({ ...f, score: +f.score.toFixed(3) }));
  }
}
