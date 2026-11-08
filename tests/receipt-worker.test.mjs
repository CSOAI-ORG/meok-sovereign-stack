// Node + in-memory R2 mock tests. These do not prove a deployed Cloudflare runtime.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import worker, { MAX_BYTES } from '../cloudflare/receipt-worker.mjs';

const token = 'test-only-token-never-use-in-production-12345';
const hash = value => createHash('sha256').update(value).digest('hex');
const receipt = { schema: 'meok.inference-receipt/0.1',
  id: 'f4e397ed-552f-41af-967c-70f146180715', observed_at: '2026-11-08T12:00:00Z',
  status: 'RUNTIME_OBSERVED', model: 'test-model', prompt_sha256: 'a'.repeat(64),
  output_sha256: 'b'.repeat(64), quality: 'UNMEASURED', signature: null,
  prompt_tokens: 12, context_truncation: 'UNKNOWN' };
const raw = JSON.stringify(receipt);
function object(bytes, size = bytes.byteLength) {
  return { size, body: new ReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }) };
}
function bucket() {
  return { records: new Map(), writes: 0, calls: [],
    async put(key, bytes, options) {
      this.calls.push(['put', key]);
      assert.equal(options.onlyIf.get('If-None-Match'), '*');
      assert.equal(options.sha256, hash(bytes));
      if (this.records.has(key)) return null;
      this.records.set(key, bytes.slice()); this.writes++;
      return { key };
    },
    async get(key) { this.calls.push(['get', key]); const bytes = this.records.get(key); return bytes ? object(bytes) : null; },
  };
}
function request(body = raw, path = '/receipts', extra = {}) {
  return new Request(`https://receipt.test${path}`, { method: 'POST', body,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...extra } });
}
const env = store => ({ RECEIPTS: store, RECEIPT_TOKEN: token });

test('POST awaits write and hash-matched readback; GET returns the exact authenticated bytes', async () => {
  const store = bucket(); const body = `\n${raw}\n`;
  const response = await worker.fetch(request(body), env(store));
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.status, 'DURABLE_READBACK'); assert.equal(result.sha256, hash(body));
  assert.deepEqual(store.calls.map(call => call[0]), ['put', 'get']);
  const get = await worker.fetch(new Request(`https://receipt.test${result.path}`,
    { headers: { Authorization: `Bearer ${token}` } }), env(store));
  assert.equal(get.status, 200); assert.equal(await get.text(), body);
  assert.equal(get.headers.get('X-Receipt-State'), 'DURABLE_READBACK');
  assert.equal(get.headers.get('Cache-Control'), 'no-store');
});

test('duplicate content uses conditional insertion and never overwrites', async () => {
  const store = bucket();
  const responses = await Promise.all([worker.fetch(request(), env(store)), worker.fetch(request(), env(store))]);
  assert.deepEqual(responses.map(response => response.status), [200, 200]);
  assert.equal(store.writes, 1); assert.equal(store.records.size, 1);
});

test('missing/wrong auth and missing secret fail before any bucket calls', async () => {
  const store = bucket();
  for (const authorization of ['', 'Bearer wrong', `Basic ${token}`, `Bearer ${token} extra`]) {
    assert.equal((await worker.fetch(request(raw, '/receipts', { Authorization: authorization }), env(store))).status, 401);
  }
  assert.equal((await worker.fetch(request(), { RECEIPTS: store })).status, 503);
  assert.equal((await worker.fetch(request(), { RECEIPTS: store, RECEIPT_TOKEN: 'short' })).status, 503);
  assert.deepEqual(store.calls, []);
});

test('malformed JSON, schema errors, plaintext fields and invalid UTF-8 are rejected', async () => {
  const store = bucket();
  const bodies = ['{', '[]', '{}', JSON.stringify({ ...receipt, schema: 'other' }),
    JSON.stringify({ ...receipt, output_sha256: null }), JSON.stringify({ ...receipt, prompt: 'private' }),
    JSON.stringify({ ...receipt, quality: 'MEASURED' }), new Uint8Array([0xff, 0xfe])];
  for (const body of bodies) assert.equal((await worker.fetch(request(body), env(store))).status, 400);
  assert.deepEqual(store.calls, []);
});

test('unknown-length stream is bounded independently of the content-length header', async () => {
  const store = bucket(); let canceled = false;
  const body = new ReadableStream({ start(controller) {
    controller.enqueue(new Uint8Array(MAX_BYTES)); controller.enqueue(new Uint8Array(1));
  }, cancel() { canceled = true; } });
  const req = new Request('https://receipt.test/receipts', { method: 'POST', body, duplex: 'half',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'Content-Length': '1' } });
  assert.equal((await worker.fetch(req, env(store))).status, 413);
  assert.equal(canceled, true); assert.deepEqual(store.calls, []);
});

test('failed, missing or corrupt readback never reports durability', async () => {
  for (const failure of ['missing', 'throw', 'corrupt', 'oversize']) {
    const store = bucket();
    store.get = async () => {
      if (failure === 'missing') return null;
      if (failure === 'throw') throw new Error('private storage detail');
      if (failure === 'oversize') return object(new Uint8Array(), MAX_BYTES + 1);
      return object(new TextEncoder().encode('different bytes'));
    };
    const response = await worker.fetch(request(), env(store));
    assert.equal(response.status, 503);
    const text = await response.text(); assert.match(text, /UNCONFIRMED/);
    assert.doesNotMatch(text, /private storage detail|different bytes/);
  }
});

test('write errors remain UNCONFIRMED even if bytes persisted before the error', async () => {
  const store = bucket(); const put = store.put.bind(store);
  store.put = async (...args) => { await put(...args); throw new Error('secret provider error'); };
  const response = await worker.fetch(request(), env(store));
  assert.equal(response.status, 503); assert.equal(store.records.size, 1);
  const text = await response.text(); assert.match(text, /UNCONFIRMED/);
  assert.doesNotMatch(text, /secret provider error|NOT_STORED/);
});

test('GET requires auth and exact hash path; missing or corrupt objects are not served', async () => {
  const store = bucket(); const path = `/receipts/${hash(raw)}`;
  assert.equal((await worker.fetch(new Request(`https://receipt.test${path}`), env(store))).status, 401);
  const get = p => worker.fetch(new Request(`https://receipt.test${p}`, { headers: { Authorization: `Bearer ${token}` } }), env(store));
  assert.equal((await get('/receipts/latest')).status, 404);
  assert.equal((await get(`${path}?token=ignored`)).status, 400);
  assert.equal((await get(path)).status, 404);
  store.records.set(`receipts/${hash(raw)}.json`, new TextEncoder().encode('corrupt'));
  assert.equal((await get(path)).status, 503);
});

test('non-JSON requests and unsupported write routes never touch storage', async () => {
  const store = bucket();
  assert.equal((await worker.fetch(request(raw, '/receipts', { 'Content-Type': 'text/plain' }), env(store))).status, 415);
  assert.equal((await worker.fetch(request(raw, '/receipts/arbitrary'), env(store))).status, 404);
  assert.equal((await worker.fetch(request(raw, '/receipts?x=1'), env(store))).status, 400);
  assert.deepEqual(store.calls, []);
});
