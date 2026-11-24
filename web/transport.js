// Transport: the one interface the dashboard uses to talk to the backend.
//
//   const tx = await createTransport();
//   tx.onMessage = (msg) => {}      // every WebSocket message from docs/protocol.md
//   tx.onStatus  = (s) => {}        // {state: 'connecting'|'open'|'closed', retryIn?: seconds}
//   tx.getJSON('/api/heatmap?metric=dropout')   -> Promise<object>
//   tx.start()
//
// The real transport and the mock (mock.js, `?mock=1`) both implement exactly this.

class RealTransport {
  constructor() {
    this.kind = 'real';
    this.onMessage = () => {};
    this.onStatus = () => {};
    this._delay = 500;
    this._ws = null;
    this._timer = null;
  }

  start() { this._connect(); }

  _connect() {
    clearTimeout(this._timer);
    this.onStatus({ state: 'connecting' });
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws;
    try {
      ws = new WebSocket(`${proto}//${location.host}/ws`);
    } catch (e) {
      this._retry();
      return;
    }
    this._ws = ws;
    ws.onopen = () => { this._delay = 500; this.onStatus({ state: 'open' }); };
    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg && typeof msg.type === 'string') this.onMessage(msg);
    };
    ws.onclose = () => { if (this._ws === ws) { this._ws = null; this._retry(); } };
    ws.onerror = () => { try { ws.close(); } catch { /* already closed */ } };
  }

  _retry() {
    const wait = this._delay;
    this._delay = Math.min(5000, Math.round(this._delay * 1.7));
    this.onStatus({ state: 'closed', retryIn: wait / 1000 });
    this._timer = setTimeout(() => this._connect(), wait);
  }

  async postJSON(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }

  async getJSON(path) {
    const r = await fetch(path, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }
}

export async function createTransport() {
  const q = new URLSearchParams(location.search);
  if (q.get('mock') && q.get('mock') !== '0') {
    const { MockTransport } = await import('./mock.js');
    return new MockTransport(q);
  }
  return new RealTransport();
}
