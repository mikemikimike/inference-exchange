# Status

Last update: October 2026

This page tells what works now and what does not. For the full list of rules and their status, see [Requirements](requirements/README.md). If you find a problem that is not on this page, open an issue.

## What works

### Inference

- OpenAI-compatible API: `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`.
- The OpenAI SDK works when you change `base_url`.
- A mock provider (`--mock`) serves canned replies. It needs no model.

### Routing

- Scoring on price, speed, trust level, load, and reputation.
- Preferences: `cheapest`, `fastest`, `most_secure`, `balanced`.
- Filters: minimum trust level and maximum output price.
- A queue holds requests when all providers are busy (maximum 50 requests, 30 seconds).
- Session affinity sends a session to the same provider when possible.
- You can see the traces of your own requests at `/v1/exchange/traces`.

### Accounts

- GitHub sign-in. Email and password sign-in in development only.
- Logout ends all sessions of the user.
- Create and revoke API keys.
- Delete your account.
- $10 of alpha credits for each new account. These credits are not real money.

### Billing

- Per-token billing for input, cached input, and output.
- The provider gets 90 percent. The platform gets 10 percent.
- Accounts and transactions stay after a restart.

### Encryption

- X25519 encryption for each request, from the coordinator to the provider.
- Optional encryption of the response to the consumer (needs the IE SDK).

### Operations

- `deploy/` runs Caddy (TLS), the coordinator, and Litestream (backups).
- `infra/` creates the VM with OpenTofu.
- `/metrics` gives Prometheus metrics.
- CI runs the tests, the web build, the image builds, `tofu validate`, and the requirements check.

## Known limits

### Trust levels are self-reported

A provider tells the coordinator its trust level. The coordinator does not verify it yet. No provider can pass L3 (App Attest) admission yet.

- **Effect**: "Hardened" means that the operator says so. A malicious operator can claim L2 and read prompts. The API shows `trust.basis: "self_reported"`. The UI shows a note.
- **Work**: PROV-004, PROV-005 (#1, #19, #22, #25).

### The coordinator can read prompts on the standard path

On `/v1/chat/completions`, the coordinator receives plaintext and encrypts it for the provider. Only `/v1/confidential/infer` keeps plaintext away from the coordinator.

- **Work**: #68.

### The browser trusts the JavaScript from the server

The coordinator serves the web UI. A compromised coordinator can send JavaScript that reads prompts before encryption.

- **Effect**: The SDK path can be protected against this. The browser path cannot be protected yet.
- **Work**: #68.

### TLS only with the deploy stack

`deploy/docker-compose.yml` gives TLS through Caddy. If you run the coordinator directly (`make dev`), it uses plain HTTP and WebSocket. On the public internet, use only the deploy stack.

### Billing is post-pay

The only check before a request is "balance is more than zero". A balance can go below zero by the cost of one request. If the consumer cancels a stream or it fails after delivery starts, the request is billed for the input and delivered output tokens (BILL-007).

### The price ceiling applies to output price only

`ocip_max_price` applies to the output price only. The chat UI does not send the input and cache limits (ROUTE-002, #52).

### No check of the context limit

If a request is longer than the provider context, the provider fails. The coordinator does not reject it first (ROUTE-007, #32).

### Limited fallback

If the coordinator cannot send to a provider, it tries one different provider. If a provider fails during a stream, the coordinator does not try again (ROUTE-008, #13).

### Some state is lost at restart

Connected providers, the queue, traces, reputation, and TPS data are in memory. A coordinator restart clears them. Providers connect again automatically (PROV-001).

### One coordinator process

One process handles all providers and consumers. See [decision 0004](decisions/0004-single-vm-sqlite.md).

## Not in the alpha

- App Attest admission (#19, #22)
- Attack tests for providers: debugger, memory, `/proc` (#4, #5, #6)
- Confidential session handshake (#25)
- Hardware benchmark before admission (#36)
- Real payments and payouts to providers (#44)
- Cache-aware routing (#33)
