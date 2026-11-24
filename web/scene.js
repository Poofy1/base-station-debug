// 3D room: floor grid, base stations with FOV frusta, live device markers + trails,
// instanced voxel heatmap, incident markers and the selected-incident highlight.
//
// Coordinates are the OpenVR standing universe (metres, +Y up, -Z forward), which is
// also three.js's convention. Wire quaternions are [w, x, y, z]; three wants x, y, z, w.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { STATES, GROUPS, kindInfo, trackingState, h, clamp } from './util.js';

// Sequential ramp, one hue. On the dark surface the low end recedes (dark) and the
// high end is bright.
const RAMP = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];
const RAMP_RGB = RAMP.map((hex) => new THREE.Color(hex));

export function rampColor(t, target = new THREE.Color()) {
  const x = clamp(t, 0, 1) * (RAMP_RGB.length - 1);
  const i = Math.min(RAMP_RGB.length - 2, Math.floor(x));
  return target.copy(RAMP_RGB[i]).lerp(RAMP_RGB[i + 1], x - i);
}
export const rampCSS = (reverse) => `linear-gradient(to right, ${(reverse ? [...RAMP].reverse() : RAMP).join(', ')})`;

const FOV_H = 150 * Math.PI / 180, FOV_V = 110 * Math.PI / 180, FOV_LEN = 1.3;
const TRAIL_LEN = 45;
const setQuat = (obj, q) => { if (q && q.length === 4) obj.quaternion.set(q[1], q[2], q[3], q[0]).normalize(); };

export class RoomView {
  constructor(container, { onPickIncident, onHoverCell } = {}) {
    this.container = container;
    this.onPickIncident = onPickIncident || (() => {});
    this.onHoverCell = onHoverCell || (() => {});
    this.devices = new Map();
    this.bases = new Map();
    this.labels = [];             // {el, pos: Vector3 | () => Vector3, visible: () => bool}
    this.layers = { heatmap: true, fov: true, trails: true, incidents: true };
    this.slice = [0, 10];
    this.heat = null;
    this.gridHalf = 0;
    this.topDown = false;
    this.ok = false;

    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (e) {
      container.append(h('div.room-error', 'WebGL is not available, so the 3D room cannot be drawn. The other panels still work.'));
      return;
    }
    this.ok = true;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x101113);
    container.append(this.renderer.domElement);
    this.labelLayer = h('div.room-labels');
    container.append(this.labelLayer);

    this.scene = new THREE.Scene();
    this.scene.add(new THREE.AmbientLight(0xffffff, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(2, 6, 3);
    this.scene.add(sun);

    this.persp = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
    this.persp.position.set(3.9, 3.7, 4.9);
    this.ortho = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.1, 100);
    this.ortho.position.set(0, 30, 0);
    this.ortho.up.set(0, 0, -1);
    this.ortho.lookAt(0, 0, 0);
    this.camera = this.persp;

    const el = this.renderer.domElement;
    this.orbit = new OrbitControls(this.persp, el);
    this.orbit.target.set(0, 0.9, 0);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.12;
    this.orbit.maxPolarAngle = Math.PI * 0.495;
    this.orbit.minDistance = 0.8;
    this.orbit.maxDistance = 25;
    this.orbit.update();
    this.pan = new OrbitControls(this.ortho, el);
    this.pan.enableRotate = false;
    this.pan.screenSpacePanning = true;
    this.pan.enabled = false;
    this.pan.update();

    this.gridGroup = new THREE.Group();
    this.baseGroup = new THREE.Group();
    this.deviceGroup = new THREE.Group();
    this.incidentGroup = new THREE.Group();
    this.highlight = this._makeHighlight();
    this.scene.add(this.gridGroup, this.baseGroup, this.deviceGroup, this.incidentGroup, this.highlight.group);

    // heatmap: one InstancedMesh, capacity grown on demand
    this.heatGeom = new THREE.BoxGeometry(1, 1, 1);
    this.heatMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.heatMesh = null;
    this.heatIndex = [];          // instance id -> cell

    this._setGrid(3);
    this._bindPointer(el);
    new ResizeObserver(() => this._resize()).observe(container);
    this._resize();
    this._tmp = new THREE.Vector3();
    this._clock = new THREE.Clock();
    const loop = () => { this._frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ public
  setLayer(name, on) {
    this.layers[name] = on;
    if (!this.ok) return;
    if (name === 'heatmap' && this.heatMesh) this.heatMesh.visible = on;
    if (name === 'incidents') this.incidentGroup.visible = on;
    if (name === 'fov') for (const b of this.bases.values()) b.fov.visible = on;
    if (name === 'trails') for (const d of this.devices.values()) d.trail.visible = on;
  }

  setTopDown(on) {
    this.topDown = on;
    if (!this.ok) return;
    this.camera = on ? this.ortho : this.persp;
    this.orbit.enabled = !on;
    this.pan.enabled = on;
    this._resize();
  }

  resetView() {
    if (!this.ok) return;
    const k = Math.max(1, this.gridHalf / 3);
    this.persp.position.set(3.9 * k, 3.7 * k, 4.9 * k);
    this.orbit.target.set(0, 0.9, 0);
    this.orbit.update();
    this.ortho.position.set(0, 30, 0);
    this.ortho.zoom = 1;
    this.pan.target.set(0, 0, 0);
    this.pan.update();
    this._resize();
  }

  setState(devices, bases) {
    if (!this.ok) return;
    let extent = 0;
    const seenB = new Set();
    for (const b of bases || []) {
      seenB.add(b.serial);
      let o = this.bases.get(b.serial);
      if (!o) this.bases.set(b.serial, (o = this._makeBase(b)));
      o.group.position.fromArray(b.pos);
      setQuat(o.group, b.quat);
      const live = b.connected !== false && b.valid !== false;
      if (o.live !== live) {
        o.live = live;
        o.body.material.color.set(live ? 0x9fc3f0 : 0x77756e);
        o.label.el.classList.toggle('is-off', !live);
        o.label.el.textContent = o.name + (live ? '' : ' ○ offline');
      }
      extent = Math.max(extent, Math.abs(b.pos[0]), Math.abs(b.pos[2]));
    }
    for (const [serial, o] of this.bases) if (!seenB.has(serial)) { this._dropLabel(o.label); this.baseGroup.remove(o.group); this.bases.delete(serial); }

    const seenD = new Set();
    for (const d of devices || []) {
      seenD.add(d.serial);
      let o = this.devices.get(d.serial);
      if (!o) this.devices.set(d.serial, (o = this._makeDevice(d)));
      const state = trackingState(d);
      const tracked = state === 'ok' || state === 'degraded';
      if (tracked || !o.hasPos) {
        o.target.fromArray(d.pos);
        if (!o.hasPos) { o.mesh.position.copy(o.target); o.hasPos = true; }
        setQuat(o.mesh, d.quat);
      }
      if (tracked) this._pushTrail(o, d.pos);
      if (o.state !== state) {
        o.state = state;
        const st = STATES[state];
        o.mesh.material.color.set(st.color);
        o.mesh.material.opacity = tracked ? 1 : 0.45;
        o.mesh.material.transparent = !tracked;
        o.trail.material.color.set(st.color);
        o.label.el.dataset.state = state;
        o.icon.textContent = st.icon;
      }
      const text = (d.label || d.serial) + (state === 'ok' ? '' : ' · ' + (d.incident ? kindInfo(d.incident).label : STATES[state].label));
      if (o.text !== text) { o.text = text; o.name.textContent = text; }
      extent = Math.max(extent, Math.abs(d.pos[0]), Math.abs(d.pos[2]));
    }
    for (const [serial, o] of this.devices) {
      if (seenD.has(serial)) continue;
      this._dropLabel(o.label); this.deviceGroup.remove(o.mesh, o.trail); this.devices.delete(serial);
    }
    const half = clamp(Math.ceil(extent + 0.4), 3, 10);
    if (half > this.gridHalf) this._setGrid(half);
  }

  /** snapshot: protocol heatmap object or null; reverse: low values are the bad (bright) ones. */
  setHeatmap(snapshot, { reverse = false } = {}) {
    this.heat = snapshot ? { ...snapshot, reverse } : null;
    this._rebuildHeat();
  }

  setSlice(lo, hi) { this.slice = [lo, hi]; this._rebuildHeat(); }

  /** Height range covered by the current heatmap cells, in metres. */
  heatHeightRange() {
    if (!this.heat || !this.heat.cells.length) return [0, 2.4];
    let lo = Infinity, hi = -Infinity;
    for (const c of this.heat.cells) { if (c[1] < lo) lo = c[1]; if (c[1] > hi) hi = c[1]; }
    return [lo * this.heat.voxel, (hi + 1) * this.heat.voxel];
  }

  setIncidents(list) {
    if (!this.ok) return;
    for (const m of [...this.incidentGroup.children]) { this.incidentGroup.remove(m); m.material.dispose(); }
    this._incGeom ||= new THREE.OctahedronGeometry(0.045);
    for (const inc of list) {
      if (!inc.pos) continue;
      const m = new THREE.Mesh(this._incGeom, new THREE.MeshBasicMaterial({ color: GROUPS[kindInfo(inc.kind).group].color }));
      m.position.fromArray(inc.pos);
      m.userData.incident = inc;
      this.incidentGroup.add(m);
    }
  }

  highlightIncident(inc, label) {
    if (!this.ok) return;
    const hl = this.highlight;
    hl.group.visible = !!(inc && inc.pos);
    hl.label.el.hidden = !hl.group.visible;
    if (!hl.group.visible) return;
    const [x, y, z] = inc.pos;
    hl.ring.position.set(x, 0.012, z);
    hl.ball.position.set(x, y, z);
    hl.stem.geometry.setFromPoints([new THREE.Vector3(x, 0, z), new THREE.Vector3(x, Math.max(y, 0.02), z)]);
    hl.pos.set(x, y + 0.16, z);
    hl.label.el.textContent = label || 'incident';
  }

  // ------------------------------------------------------------------ builders
  _addLabel(cls, pos, visible) {
    const el = h('div.room-label.' + cls);
    this.labelLayer.append(el);
    const l = { el, pos, visible: visible || null };
    this.labels.push(l);
    return l;
  }
  _dropLabel(l) { l.el.remove(); this.labels = this.labels.filter((x) => x !== l); }

  _setGrid(half) {
    this.gridHalf = half;
    for (const c of [...this.gridGroup.children]) { this.gridGroup.remove(c); c.geometry?.dispose(); }
    for (const l of this.labels.filter((x) => x.grid)) this._dropLabel(l);
    const grid = new THREE.GridHelper(half * 2, half * 2, 0x5d6068, 0x30333a);
    this.gridGroup.add(grid);
    const sub = new THREE.GridHelper(half * 2, half * 4, 0x1e2126, 0x1e2126);
    sub.position.y = -0.002;
    this.gridGroup.add(sub);
    // origin cross
    const cross = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.15, 0.004, 0), new THREE.Vector3(0.15, 0.004, 0),
        new THREE.Vector3(0, 0.004, -0.15), new THREE.Vector3(0, 0.004, 0.15)]),
      new THREE.LineBasicMaterial({ color: 0xc3c2b7 }));
    this.gridGroup.add(cross);
    for (let i = -half; i <= half; i++) {
      const a = this._addLabel('tick', new THREE.Vector3(i, 0, half + 0.22));
      a.el.textContent = i === half ? `${i} m  X` : String(i); a.grid = true;
      if (i === 0) continue;
      const b = this._addLabel('tick', new THREE.Vector3(-half - 0.22, 0, i));
      b.el.textContent = i === -half ? `${i} m  Z` : String(i); b.grid = true;
    }
    const f = this._addLabel('tick.front', new THREE.Vector3(0, 0, -half - 0.3));
    f.el.textContent = '▲ front (−Z)'; f.grid = true;
    // top-down frustum follows the grid
    this._resize();
  }

  _makeBase(b) {
    const group = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.07), new THREE.MeshLambertMaterial({ color: 0x9fc3f0 }));
    group.add(body);
    // FOV frustum: four corner rays of equal length plus the far rim, forward is local -Z
    const th = Math.tan(FOV_H / 2), tv = Math.tan(FOV_V / 2);
    const corner = (sx, sy) => new THREE.Vector3(sx * th, sy * tv, -1).normalize().multiplyScalar(FOV_LEN);
    const rim = [];
    const N = 12;
    const edge = (ax, ay, bx, by) => {
      for (let i = 0; i < N; i++) {
        const u = i / N;
        rim.push(new THREE.Vector3((ax + (bx - ax) * u) * th, (ay + (by - ay) * u) * tv, -1).normalize().multiplyScalar(FOV_LEN));
      }
    };
    edge(-1, -1, 1, -1); edge(1, -1, 1, 1); edge(1, 1, -1, 1); edge(-1, 1, -1, -1);
    const pts = [];
    for (const c of [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]) pts.push(new THREE.Vector3(), c);
    for (let i = 0; i < rim.length; i++) pts.push(rim[i], rim[(i + 1) % rim.length]);
    pts.push(new THREE.Vector3(), new THREE.Vector3(0, 0, -FOV_LEN * 0.55));       // boresight
    const fov = new THREE.Group();
    fov.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x86b6ef, transparent: true, opacity: 0.6 })));
    const fan = [];
    for (let i = 0; i < rim.length; i++) fan.push(new THREE.Vector3(), rim[i], rim[(i + 1) % rim.length]);
    fov.add(new THREE.Mesh(new THREE.BufferGeometry().setFromPoints(fan),
      new THREE.MeshBasicMaterial({ color: 0x86b6ef, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })));
    fov.visible = this.layers.fov;
    group.add(fov);
    this.baseGroup.add(group);
    const name = (b.channel != null ? `S-${b.channel} ` : '') + String(b.serial).replace(/^LHB-/, '');
    const label = this._addLabel('base', () => this._tmp.copy(group.position).setY(group.position.y + 0.14));
    label.el.textContent = name;
    return { group, body, fov, label, name, live: true };
  }

  _makeDevice(d) {
    let geom;
    if (d.class === 'hmd') geom = new THREE.BoxGeometry(0.19, 0.11, 0.12);
    else if (d.class === 'controller') { geom = new THREE.CylinderGeometry(0.028, 0.04, 0.16, 12); geom.rotateX(Math.PI / 2); }
    else geom = new THREE.SphereGeometry(0.055, 16, 12);
    const mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: 0xffffff }));
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
    tg.setDrawRange(0, 0);
    const trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
    trail.frustumCulled = false;
    trail.visible = this.layers.trails;
    this.deviceGroup.add(mesh, trail);
    const label = this._addLabel('device', () => this._tmp.copy(mesh.position).setY(mesh.position.y + 0.13));
    const icon = h('span.icon'), name = h('span');
    label.el.append(icon, name);
    return { mesh, trail, label, icon, name, target: new THREE.Vector3(), hasPos: false, state: null, text: null, pts: [] };
  }

  _pushTrail(o, pos) {
    const last = o.pts[o.pts.length - 1];
    if (last && Math.hypot(pos[0] - last[0], pos[1] - last[1], pos[2] - last[2]) < 0.004) return;
    o.pts.push(pos);
    if (o.pts.length > TRAIL_LEN) o.pts.shift();
    const attr = o.trail.geometry.attributes.position;
    for (let i = 0; i < o.pts.length; i++) attr.setXYZ(i, o.pts[i][0], o.pts[i][1], o.pts[i][2]);
    attr.needsUpdate = true;
    o.trail.geometry.setDrawRange(0, o.pts.length);
  }

  _makeHighlight() {
    const group = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.235, 48), mat);
    ring.rotation.x = -Math.PI / 2;
    ring.renderOrder = 10;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 10),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.9, depthTest: false }));
    ball.renderOrder = 10;
    const stem = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
    stem.renderOrder = 10;
    stem.frustumCulled = false;
    group.add(ring, ball, stem);
    group.visible = false;
    const pos = new THREE.Vector3();
    const hl = { group, ring, ball, stem, pos, label: null };
    queueMicrotask(() => { if (this.ok) { hl.label = this._addLabel('pick', pos); hl.label.el.hidden = true; } });
    return hl;
  }

  // ------------------------------------------------------------------ heatmap
  _rebuildHeat() {
    if (!this.ok) return;
    const heat = this.heat;
    const cells = heat ? heat.cells : [];
    if (!this.heatMesh || this.heatMesh.instanceMatrix.count < cells.length) {
      if (this.heatMesh) { this.scene.remove(this.heatMesh); this.heatMesh.dispose(); }
      const cap = Math.max(1024, Math.ceil(cells.length * 1.5));
      this.heatMesh = new THREE.InstancedMesh(this.heatGeom, this.heatMat, cap);
      this.heatMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.heatMesh.frustumCulled = false;
      this.heatMesh.visible = this.layers.heatmap;
      this.scene.add(this.heatMesh);
    }
    const mesh = this.heatMesh;
    this.heatIndex.length = 0;
    if (!heat) { mesh.count = 0; return; }
    const v = heat.voxel, span = heat.max - heat.min;
    const m = new THREE.Matrix4(), col = new THREE.Color();
    const [lo, hi] = this.slice;
    let n = 0;
    for (const c of cells) {
      const cy = (c[1] + 0.5) * v;
      if (cy < lo || cy > hi) continue;
      let t = span > 0 ? (c[3] - heat.min) / span : 1;
      if (heat.reverse) t = span > 0 ? 1 - t : 1;
      const s = v * (0.28 + 0.64 * t);
      m.makeScale(s, s, s).setPosition((c[0] + 0.5) * v, cy, (c[2] + 0.5) * v);
      mesh.setMatrixAt(n, m);
      mesh.setColorAt(n, rampColor(t, col));
      this.heatIndex.push(c);
      n++;
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }

  // ------------------------------------------------------------------ interaction
  _bindPointer(el) {
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down = null, pending = null;
    const setNdc = (e) => {
      const r = el.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, this.camera);
    };
    el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    el.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      if (!this.layers.incidents) return;
      setNdc(e);
      ray.params.Mesh = {};
      // generous pick radius: test distance from the ray to each marker
      let best = null, bestD = 0.14;
      for (const m of this.incidentGroup.children) {
        const dist = ray.ray.distanceToPoint(m.position);
        if (dist < bestD) { bestD = dist; best = m; }
      }
      if (best) this.onPickIncident(best.userData.incident);
    });
    el.addEventListener('pointermove', (e) => {
      if (e.buttons) { this.onHoverCell(null); return; }
      pending = e;
    });
    el.addEventListener('pointerleave', () => { pending = null; this.onHoverCell(null); });
    this._hoverTick = () => {
      if (!pending) return;
      const e = pending; pending = null;
      if (!this.heatMesh || !this.heatMesh.visible || !this.heatMesh.count) { this.onHoverCell(null); return; }
      setNdc(e);
      const hit = ray.intersectObject(this.heatMesh, false)[0];
      this.onHoverCell(hit && hit.instanceId != null ? this.heatIndex[hit.instanceId] : null, e);
    };
  }

  _resize() {
    if (!this.ok) return;
    const w = Math.max(1, this.container.clientWidth), hgt = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, hgt, false);
    this.persp.aspect = w / hgt;
    this.persp.updateProjectionMatrix();
    const half = (this.gridHalf || 3) + 0.9;
    const aspect = w / hgt;
    const hw = aspect >= 1 ? half * aspect : half, hh = aspect >= 1 ? half : half / aspect;
    this.ortho.left = -hw; this.ortho.right = hw; this.ortho.top = hh; this.ortho.bottom = -hh;
    this.ortho.updateProjectionMatrix();
    this._w = w; this._h = hgt;
  }

  _frame() {
    const dt = this._clock.getDelta();
    const k = 1 - Math.exp(-dt * 18);
    for (const o of this.devices.values()) o.mesh.position.lerp(o.target, k);
    if (this.highlight.group.visible) {
      const s = 1 + 0.25 * Math.sin(performance.now() / 260);
      this.highlight.ring.scale.setScalar(s);
      this.highlight.ball.rotation.y += dt * 1.2;
    }
    (this.topDown ? this.pan : this.orbit).update();
    this._hoverTick();
    this.renderer.render(this.scene, this.camera);
    // HTML labels follow their 3D anchors
    const v = this._tmp2 || (this._tmp2 = new THREE.Vector3());
    for (const l of this.labels) {
      if (l.el.hidden) continue;
      v.copy(typeof l.pos === 'function' ? l.pos() : l.pos).project(this.camera);
      const off = v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05;
      l.el.style.display = off ? 'none' : '';
      if (!off) l.el.style.transform = `translate(-50%, -100%) translate(${((v.x + 1) / 2 * this._w).toFixed(1)}px, ${((1 - v.y) / 2 * this._h).toFixed(1)}px)`;
    }
  }
}
