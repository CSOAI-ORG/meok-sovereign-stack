# Receipt storage candidate

Source only: no deployment, bucket creation or paid service is activated. `wrangler.example.jsonc` is deliberately an example filename with no account or routes. The owning release workflow must supply an existing authorised R2 bucket and a secret `RECEIPT_TOKEN` of at least 32 characters. Never embed that bearer token in a public/static client.

`POST /receipts` accepts the bounded `meok.inference-receipt/0.1` metadata object. Authentication hashes both tokens before constant-time comparison. The 64 KiB limit is applied to streamed bytes, independently of `Content-Length`. The object key is SHA-256 of the exact submitted UTF-8 JSON bytes; whitespace/key-order changes produce a different key. Conditional insertion prevents this adapter from overwriting existing content.

Success requires awaited R2 insertion followed by a full bounded read and matching hash. `DURABLE_READBACK` means matching bytes were read from R2 at that time; it does not establish model correctness, signatures, permanent retention or a bucket lock. Unconfirmed writes/readbacks return `UNCONFIRMED` because a response error does not prove the bytes were not stored. Authenticated `GET /receipts/{sha256}` checks the content hash before returning bytes. No request body, token or provider error is logged.

Run `node --test tests/receipt-worker.test.mjs` from the repository root. These Node tests use an in-memory R2 mock and do not establish deployed Cloudflare behaviour or cloud durability. A future release still needs Workers-runtime validation, actual authenticated write/readback and quota checks.

APIs checked 8 November 2026 against [R2 Workers API](https://developers.cloudflare.com/r2/api/workers/workers-api-reference/), [Node crypto support](https://developers.cloudflare.com/workers/runtime-apis/nodejs/crypto/), [Workers best practices](https://developers.cloudflare.com/workers/best-practices/workers-best-practices/), latest published `@cloudflare/workers-types` and Wrangler config schema. No dependencies were installed for this source check.
