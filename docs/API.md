# API Reference

LLM scheduling proxy that manages scheduling, rate limiting, and system-prompt caching on behalf of multiple concurrent clients.

## Authentication

Every request must include a Bearer token:

```
Authorization: Bearer <PROXY_API_KEY>
```

The server requires `PROXY_API_KEY` to be set in the environment (or `.env`) and will refuse to start without it. Requests missing a valid token receive `401 Unauthorized`.

## Endpoints

### Call Types

#### `POST /call_types`

Register or update a reusable (name, system_prompt) pair.

**Body**

| Field           | Type   | Required | Description                        |
|-----------------|--------|----------|------------------------------------|
| `name`          | string | yes      | Unique identifier for the call type |
| `system_prompt` | string | yes      | System prompt to prepend to messages |

**Response** `201` — `{"name": "...", "status": "created"|"updated"}`

#### `GET /call_types`

List registered call type names.

**Response** `200` — `{"call_types": ["orchestrator", "analyst_web_research", ...]}`

---

### Sessions

#### `POST /sessions`

Create a new session.

**Response** `200` — `{"session_id": "<uuid>"}`

#### `DELETE /sessions/{session_id}`

Tear down a session.

**Response** `204` — No content

---

### Completions

#### `POST /sessions/{session_id}/completions`

Submit a single LLM call through the scheduler.

**Body**

| Field        | Type       | Required | Description                              |
|--------------|------------|----------|------------------------------------------|
| `call_type`  | string     | yes      | Registered call type name                |
| `messages`   | list[dict] | yes      | Conversation messages (user/assistant)   |
| `max_tokens` | int        | no       | Output cap (defaults to server setting)  |
| `call_detail`| string     | no       | Sub-label for scheduling (e.g. `"plan"`) |

**Response** `200`

```json
{"content": "...", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
```

#### `POST /sessions/{session_id}/completions/batch`

Fan out multiple LLM calls concurrently within a single session.

**Body**

| Field  | Type                   | Required | Description                  |
|--------|------------------------|----------|------------------------------|
| `calls`| list[CompletionRequest]| yes      | Array of completion requests |

**Response** `200`

```json
{"completions": [{"content": "...", "usage": {...}}, ...]}
```

---

### Simulation / Benchmarking

These endpoints are for benchmark runs only — not for production use.

#### `GET /sim/config`

Returns current server configuration (scheduler, RPM, TPM, max_tokens, cost_limit, model).

#### `PATCH /sim/config`

Update mutable config. Currently supports `max_tokens` (int, >= 1).

#### `GET /sim/stats`

Snapshot of server-side metrics: rate-limiter counters, cost totals, scheduler stats, active session count.

#### `POST /sim/reset`

Reset all server state for a fresh benchmark run. Optionally switch scheduler.

**Body** (optional)

| Field      | Type   | Required | Description                          |
|------------|--------|----------|--------------------------------------|
| `scheduler`| string | no       | Switch to this scheduler on reset    |

**Response** `200` — `{"status": "reset", "scheduler": "..."}`

## Configuration

Set via environment variables or `.env`:

| Variable        | Default         | Description                 |
|-----------------|-----------------|-----------------------------|
| `PROXY_API_KEY` | —  (required)   | Bearer token for auth       |
| `SCHEDULER`     | `fifo`          | Scheduler name              |
| `RPM`           | `30`            | Requests per minute limit   |
| `TPM`           | `200000`        | Tokens per minute limit     |
| `MAX_TOKENS`    | `2048`          | Per-completion output cap   |
| `COST_LIMIT`    | `20.0`          | Hard dollar budget          |
| `MODEL`         | `gpt-4.1-nano`  | OpenAI model identifier     |
