# Empirical Analysis of OpenAI Rate Limit Behavior Under Concurrent Load

## 1. Motivation

LLM-powered applications that dispatch many concurrent requests must enforce rate
limits locally to avoid upstream 429 (rate limit exceeded) errors. The standard
approach is to mirror the provider's stated limits in a local token-bucket rate
limiter: if OpenAI advertises 2,000,000 TPM and 5,000 RPM for `gpt-4.1-nano`,
configure a local bucket with those parameters and gate every outgoing request
through it.

We implemented exactly this in our scheduling proxy. The local rate limiter
(`sim/rate_limiter.py`) uses a dual token-bucket algorithm — one bucket for RPM,
one for TPM — both configured to match OpenAI's stated limits. Before each API
call, the scheduler deducts an estimated token count from the TPM bucket. After
the call completes, the overestimate is refunded so the net deduction equals the
actual tokens consumed. The accounting is mathematically correct: each call's net
impact on the local bucket is exactly `-actual_tokens`, identical to what
OpenAI should be deducting from its own budget.

**Despite this, our 200-session load test produced 253 upstream 429 errors while
the local rate limiter never once throttled a request.** The local TPM bucket
never dropped below 1,519,714 tokens (76% of capacity) throughout the entire
run. At the same moment our bucket showed 1.5M+ tokens available, OpenAI's
`remaining-tokens` header read 0.

This 1.5M-token divergence cannot be explained by per-call estimation error. Our
estimates average 4,148 tokens versus OpenAI's likely reservation of 3,650
tokens (prompt + max\_tokens), making us *more* conservative by ~497 tokens per
call. Across 100 concurrent in-flight requests, that accounts for at most 50K
tokens of divergence — two orders of magnitude less than the observed 1.5M gap.

The divergence grows linearly with time under load, reaching +1.8M by t=100s,
which implies a constant-rate mismatch — most likely between our assumed refill
rate (`tpm / 60 = 33,333 tokens/s`) and OpenAI's actual rate-limit mechanics.
Since OpenAI does not publicly document the internal parameterization of their
rate limiter (bucket capacity, refill rate, reservation policy for in-flight
requests), the only way to understand the behavior is empirically.

We designed three experiments to characterize OpenAI's rate limiting from the
client side, covering both TPM and RPM dimensions.

## 2. Experiment Design

### 2.1 Experiment 1: Concurrent Load Test (200 Sessions)

**Goal:** Observe the divergence between a correctly-implemented local token
bucket and OpenAI's actual enforcement under realistic multi-agent workload
conditions.

**Setup:**
- 200 concurrent sessions dispatched with 0.3s stagger through a FIFO scheduler
- Each session runs a multi-step research agent workflow (plan → search → analyze → synthesize → review)
- `max_tokens = 2048`, model = `gpt-4.1-nano`
- Local rate limiter configured to OpenAI's stated limits: 5,000 RPM / 2,000,000 TPM
- Server instrumented to log every OpenAI response's `x-ratelimit-*` headers as JSON lines with millisecond timestamps

**Why this design:** This is a *realistic* workload, not a synthetic benchmark.
The multi-step agent workflow produces variable prompt sizes (87–11,593 tokens)
and variable completion lengths (30–2,048 tokens), exercising the rate limiter
across a wide range of per-call costs. The 200-session count was chosen to push
aggregate throughput toward the 2M TPM ceiling based on prior profiling (average
~2,488 actual tokens per call × ~15 calls/session × 200 sessions ÷ ~100s
duration ≈ 750K tokens/min sustained, with burst peaks significantly higher due
to staggered ramp-up).

**Instrumentation:** The server (`api.py`) captures the full set of rate-limit
headers from every OpenAI response — including 429 error responses — via the
`with_raw_response` API. Headers are logged with elapsed time from experiment
start, enabling direct comparison between the local bucket state (from
`rate_limiter.py` logs) and OpenAI's reported state (from headers).

**Data collected:**
- `results/openai_headers.log`: 2,348 JSON lines (2,095 OK + 253 429)
- `results/server.log`: local rate limiter acquire/refund events with bucket state

### 2.2 Experiment 2: TPM Burst Test

**Goal:** Determine whether OpenAI's TPM rate limiter uses a token bucket or a
sliding window algorithm, and measure its recovery dynamics.

**Setup (`experiments/burst_test.py`):**
- Phase 1 (BURST): Fire 400 concurrent requests with `max_tokens=2048` and a
  long prompt to consume a large fraction of the 2M token budget in a short
  window (~806K tokens consumed in ~31 seconds)
- Phase 2 (PROBE): After the burst, send a single tiny request (`max_tokens=1`)
  every 3 seconds for 90 seconds, recording `remaining-tokens` on each response

**Why this design:** The two candidate models make sharply different predictions
about recovery after a burst:

| Model | Prediction 30s after 806K consumed |
|---|---|
| **Token bucket** (capacity=2M, refill=33,333/s) | Linear recovery: remaining ≈ 1.19M + 1M refill = 2.19M → capped at 2M |
| **Sliding window** (2M per 60s) | Flat: remaining ≈ 1.19M until burst tokens expire at t=60s, then jump |

A probe-only Phase 2 (no concurrent load) isolates the recovery curve from
reservation noise, giving a clean signal to distinguish the two models.

**Data collected:**
- `results/burst_experiment.jsonl`: 426 data points (400 burst + 1 initial + 25 probes)

### 2.3 Experiment 3: RPM Reliability Test

**Goal:** Determine whether OpenAI's RPM (requests-per-minute) headers exhibit
the same unpredictability as TPM headers under concurrent load, and whether RPM
rate limiting has the same divergence problem.

**Setup (`experiments/rpm_test.py`):**
- Phase 1 (BURST): Fire 2,000 concurrent requests with `max_tokens=1` and a
  3-word prompt. Token cost per request ≈ 12 tokens, so total ≈ 24K tokens
  (0.001% of TPM budget). This isolates RPM from TPM.
- Phase 2 (SUSTAINED): Send at ~80 req/s for 60 seconds to maintain pressure
  near the 5,000 RPM limit
- Phase 3 (PROBE): Stop sending, probe every 1s for 90 seconds to observe
  `remaining-requests` recovery

**Why this design:** RPM has no `max_tokens` reservation ambiguity — each
request costs exactly 1 from the RPM budget regardless of token count. If RPM
headers are consistent under concurrency while TPM headers are not, it would
confirm that the TPM unpredictability stems from per-request token reservation
dynamics rather than a fundamental property of all OpenAI rate-limit headers.

**Data collected:**
- `results/rpm_experiment.jsonl`: 2,190 data points (1,997 burst + 125 sustained + 64 probes)

## 3. Findings

### 3.1 The Local Rate Limiter Never Throttled

Despite 253 upstream 429 errors, the local TPM bucket never dropped below
1,519,714 tokens. Zero TPM throttles were recorded across 1,134 successful
calls. The rate limiter was a complete no-op for the entire experiment.

The divergence between the local bucket and OpenAI's `remaining-tokens` grew
monotonically under load:

| Time | Local bucket | OpenAI `remaining-tokens` | Divergence |
|------|-------------|--------------------------|------------|
| t=0s | 1,993,509 | 1,999,753 | −6,244 (local is more conservative) |
| t=20s | 1,982,558 | 1,980,666 | +1,892 |
| t=36s | 1,916,885 | 1,711,945 | +204,940 |
| t=48s | 1,733,584 | 1,154,440 | +579,144 |
| t=60s | 1,608,753 | 533,106 | +1,075,647 |
| t=68s | 1,544,764 | 230,746 | +1,314,018 |
| t=80s | 1,676,210 | 223,430 | +1,452,780 |
| t=100s | 1,997,538 | 184,062 | +1,813,476 |

At t=0–16s (low load), the local bucket tracks ~5–10K below OpenAI's — exactly
what correct pre-deduction accounting predicts. The crossover occurs at t≈20s as
concurrent load ramps up, and the gap widens linearly at roughly 18,000
tokens/second thereafter.

The linear growth rate of 18K tokens/s implies a constant-rate refill mismatch.
Our bucket refills at 33,333 tokens/s (`2M / 60`). If OpenAI's effective refill
is ~15,000 tokens/s, the resulting drift matches the observed divergence
precisely: `(33,333 − 15,000) × 100s ≈ 1.83M`.

### 3.2 TPM Headers Are Unpredictable Under Concurrent Load

The most striking finding is that OpenAI's `remaining-tokens` header is
**not a consistent snapshot of a single shared counter** under concurrent load.

During the 429 storm (t=68–100s), successful responses and 429 rejections arrived
at the same timestamps with wildly different `remaining-tokens` values:

| Timestamp | Status | `remaining-tokens` |
|-----------|--------|--------------------|
| t=68.1s | OK | 524,766 |
| t=68.1s | OK | 52,599 |
| t=68.3s | **429** | **0** |
| t=68.3s | OK | 755,298 |
| t=68.4s | **429** | **0** |
| t=68.4s | OK | 844,959 |

72 out of 806 100ms time windows (8.9%) contained both successful and 429
responses simultaneously. During the storm, successful responses reported
`remaining-tokens` ranging from 743 to 1,091,998 — a 1,000× spread — while
simultaneous 429 responses consistently reported 0.

This pattern is consistent with OpenAI pre-reserving `prompt_tokens +
max_tokens` capacity for each in-flight request. The `remaining-tokens` header
on a successful response reflects the bucket state *after that specific
response's reservation is settled* (reservation removed, actual tokens charged),
but does not subtract reservations held by other in-flight requests. The 429
decision, however, checks available capacity *including* all in-flight
reservations — which is why it can see 0 while a concurrent completion sees
755K.

**Implication:** The `remaining-tokens` header cannot be used as a reliable
feedback signal for a closed-loop rate limiter. It systematically overstates
available capacity during high concurrency because it omits in-flight
reservation accounting.

![OpenAI remaining-tokens under concurrent load](../results/openai_remaining_unpredictable.png)

### 3.3 TPM Uses a Token Bucket, Not a Sliding Window

The burst experiment consumed 805,996 tokens across 400 concurrent requests.
The first probe after the burst (t=34.6s, 3.3s after burst end) showed
`remaining-tokens = 1,885,071`. By t=39.2s (7.9s after burst), capacity had
fully recovered to 1,999,995.

A 60-second sliding window model predicts that `remaining-tokens` should remain
at approximately `2,000,000 − 805,996 = 1,194,004` for the full 60 seconds
following the burst, then jump back to 2M as the burst tokens expire from the
trailing edge. Instead, recovery occurred within ~8 seconds, definitively ruling
out a sliding window.

The recovery rate of ~806K tokens in ~8 seconds implies a refill of ~100K
tokens/s — substantially faster than the `2M / 60 = 33,333/s` that a naive
interpretation of "2,000,000 tokens per minute" would suggest. This confirms
that OpenAI's "TPM" limit describes the token-bucket *capacity* (maximum burst
size), not the sustained refill rate. The actual refill rate appears to be
significantly higher, allowing faster recovery after bursts but making steady-
state behavior harder to predict from the stated limit alone.

### 3.4 RPM Headers Are Equally Noisy

The RPM experiment fired 2,000 concurrent `max_tokens=1` requests. Zero 429
errors occurred — the minimum `remaining-requests` observed was 4,548 (out of
5,000), meaning OpenAI counted at most 452 concurrent requests as consumed
despite receiving 2,000.

However, **RPM headers showed the same inconsistency as TPM headers.** 90 out
of 91 100ms windows with multiple readings (98.9%) showed inconsistent
`remaining-requests` values. The maximum spread within a single 200ms window was
208 — responses arriving within 200ms of each other reported
`remaining-requests` values differing by up to 208.

Recovery was nearly instantaneous: by the sustained phase (t=32s+),
`remaining-requests` had returned to 4,999 and remained there for the duration.

The effective throughput during the burst was ~4,073 requests/minute (2,000
requests over ~29.4 seconds), well below the 5,000 RPM limit. Combined with
the very fast recovery, this suggests RPM is unlikely to be the binding
constraint in practice for typical LLM workloads where API latency (2–5 seconds
per call) naturally limits concurrency.

![RPM headers under concurrent load](../results/rpm_unpredictable.png)

### 3.5 Summary of Rate Limit Header Reliability

| Property | TPM | RPM |
|----------|-----|-----|
| Stated limit | 2,000,000 | 5,000 |
| Algorithm | Token bucket | Token bucket |
| Headers consistent under concurrency? | **No** (8.9% of windows mixed OK+429) | **No** (98.9% of windows inconsistent) |
| Max header spread in 200ms window | >1,000,000 | 208 |
| 429s observed? | Yes (253 in 200-session test) | No (0 in 2,000-request burst) |
| Recovery after burst | ~8s for 806K tokens | Nearly instant |
| Suitable for closed-loop feedback? | **No** | **No** |
| Binding constraint in practice? | **Yes** | Unlikely |

## 4. Implications

### 4.1 Why Standard Token-Bucket Modeling Fails

The standard approach — configure a local token bucket with `capacity = stated_TPM`
and `refill_rate = stated_TPM / 60` — fails because it makes two assumptions
that do not hold:

1. **Refill rate assumption.** OpenAI's stated "2,000,000 tokens per minute"
   describes the bucket capacity, not the refill rate. Our burst experiment
   shows the actual refill rate is ~100K tokens/s (6M/min), roughly 3× higher
   than the 33,333/s implied by a naive `capacity / 60` calculation. Under
   sustained load, a local bucket configured with the lower refill rate
   accumulates capacity faster than OpenAI's actual depletion rate in some
   regimes but slower in others, leading to the observed drift.

2. **Observable state assumption.** The `remaining-tokens` header does not
   provide a consistent, globally-serialized view of available capacity. Under
   concurrency, it reflects a per-response post-settlement snapshot that omits
   in-flight reservations. This makes it unsuitable for closed-loop feedback
   control.

### 4.2 What Would Work

Given these findings, effective client-side rate-limit mitigation should rely on:

- **Concurrency control** rather than token-bucket modeling. Limiting the number
  of concurrent in-flight requests bounds the total in-flight `max_tokens`
  reservation at OpenAI, directly addressing the mechanism that causes 429s.
  With an average reservation of ~3,650 tokens per request, a concurrency cap of
  N limits total reserved capacity to `N × 3,650`. Setting this below a fraction
  of the 2M capacity (e.g., N ≤ 400 for ≤1.46M reserved) provides a safety
  margin.

- **Exponential backoff on 429s** as the ground-truth signal. Unlike the
  `remaining-tokens` header, a 429 response is an unambiguous indicator that
  capacity is exhausted. Backing off on 429s (which we already implement) and
  treating it as the primary rate-limit feedback is more reliable than any
  header-based approach.

- **Conservative safety margins.** If using a local token bucket, configure it
  with substantially lower capacity than the stated limit (e.g., 50–70% of
  stated TPM) to account for the unknown refill-rate mismatch and in-flight
  reservation overhead.
