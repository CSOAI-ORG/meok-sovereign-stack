// Source candidate only. Storage readback proves bytes, not inference correctness.
import { createHash, timingSafeEqual } from 'node:crypto';

export const MAX_BYTES = 64 * 1024;
const HEX = /^[0-9a-f]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const ALLOWED = new Set(('schema id observed_at status specialist hive clan provider model '
  + 'expected_digest observed_digest prompt_sha256 output_sha256 config_sha256 max_tokens '
  + 'generation quality signature limitations finish_reason generated_tokens duration_ms '
  + 'error_type prompt_eval_count prompt_tokens context_truncation identity_binding').split(' '));
const headers = { 'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
const sha = value => createHash('sha256').update(value).digest('hex');
const reply = (status, body) => new Response(JSON.stringify(body), { status, headers });

function authenticated(request, secret) {
  const header = request.headers.get('Authorization') || '';
  const token = /^Bearer ([^\s]+)$/.exec(header)?.[1] || '';
  // Hash first so timingSafeEqual always receives equal-length buffers.
  const supplied = createHash('sha256').update(token).digest();
  const expected = createHash('sha256').update(secret).digest();
  return timingSafeEqual(supplied, expected) && token.length > 0;
}

async function boundedBytes(stream) {
  if (!stream) return new Uint8Array();
  const reader = stream.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BYTES) {
        await reader.cancel();
        throw new RangeError('BODY_TOO_LARGE');
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

function isReceipt(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  if (Object.keys(value).some(key => !ALLOWED.has(key))) return false;
  if (value.schema !== 'meok.inference-receipt/0.1' || !UUID.test(value.id)) return false;
  if (typeof value.observed_at !== 'string' || !Number.isFinite(Date.parse(value.observed_at))) return false;
  if (!['RUNTIME_OBSERVED', 'UNREACHABLE', 'UNCHECKABLE'].includes(value.status)) return false;
  if (typeof value.model !== 'string' || !value.model.length || value.model.length > 256) return false;
  if (!HEX.test(value.prompt_sha256) || value.quality !== 'UNMEASURED' || value.signature !== null) return false;
  if (value.status === 'RUNTIME_OBSERVED' ? !HEX.test(value.output_sha256) : value.output_sha256 !== null) return false;
  return true;
}

async function readVerified(bucket, key, hash) {
  const object = await bucket.get(key);
  if (!object) return null;
  if (!Number.isInteger(object.size) || object.size > MAX_BYTES || !object.body) throw new Error('READBACK_INVALID');
  const bytes = await boundedBytes(object.body);
  if (bytes.byteLength !== object.size || sha(bytes) !== hash) throw new Error('READBACK_MISMATCH');
  return bytes;
}

export default {
  async fetch(request, env) {
    if (typeof env.RECEIPT_TOKEN !== 'string' || env.RECEIPT_TOKEN.length < 32 || !env.RECEIPTS) {
      return reply(503, { status: 'UNCHECKABLE', reason: 'CONFIGURATION_REQUIRED' });
    }
    if (!authenticated(request, env.RECEIPT_TOKEN)) return reply(401, { status: 'UNAUTHORIZED' });
    const url = new URL(request.url);
    if (url.search) return reply(400, { status: 'REJECTED', reason: 'QUERY_NOT_SUPPORTED' });
    const match = /^\/receipts\/([0-9a-f]{64})$/.exec(url.pathname);
    if (request.method === 'GET' && match) {
      try {
        const bytes = await readVerified(env.RECEIPTS, `receipts/${match[1]}.json`, match[1]);
        if (!bytes) return reply(404, { status: 'UNCHECKABLE', reason: 'NOT_FOUND' });
        return new Response(bytes, { headers: { ...headers, ETag: `"${match[1]}"`,
          'X-Receipt-State': 'DURABLE_READBACK' } });
      } catch { return reply(503, { status: 'UNCONFIRMED', reason: 'READBACK_UNCONFIRMED' }); }
    }
    if (request.method !== 'POST' || url.pathname !== '/receipts') {
      return reply(404, { status: 'REJECTED', reason: 'ROUTE_NOT_FOUND' });
    }
    if (request.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/json') {
      return reply(415, { status: 'REJECTED', reason: 'JSON_REQUIRED' });
    }
    let bytes;
    try {
      bytes = await boundedBytes(request.body);
      const receipt = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
      if (!isReceipt(receipt)) return reply(400, { status: 'REJECTED', reason: 'INVALID_RECEIPT' });
    } catch (error) {
      return reply(error instanceof RangeError ? 413 : 400,
        { status: 'REJECTED', reason: error instanceof RangeError ? 'BODY_TOO_LARGE' : 'INVALID_RECEIPT' });
    }
    const hash = sha(bytes);
    const key = `receipts/${hash}.json`;
    try {
      // If the key already exists, never overwrite it; verify existing bytes below.
      await env.RECEIPTS.put(key, bytes, { onlyIf: new Headers({ 'If-None-Match': '*' }),
        sha256: hash, httpMetadata: { contentType: 'application/json' } });
      const readback = await readVerified(env.RECEIPTS, key, hash);
      if (!readback) throw new Error('READBACK_MISSING');
      return reply(200, { status: 'DURABLE_READBACK', sha256: hash,
        path: `/receipts/${hash}`, byte_length: readback.byteLength,
        limitation: 'Matching bytes read from R2; not correctness or permanent retention.' });
    } catch {
      // A failed/ambiguous write might have persisted. Never label it NOT_STORED.
      return reply(503, { status: 'UNCONFIRMED', sha256: hash, reason: 'WRITE_OR_READBACK_UNCONFIRMED' });
    }
  },
};
