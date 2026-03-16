# CS244 - LLM Agent Scheduling Strategies

A research system that studies how different scheduling strategies affect performance when multiple LLM-powered agent sessions compete for rate-limited API access.

## Two Layers

| Layer | Purpose | Entry Point |
|-------|---------|-------------|
| **Simulation** (`sim/`) | Test and compare schedulers in controlled experiments | `python -m sim.runner` |
| **Deployment** (`api.py`) | Deploy schedulers for production use by multiple clients | `hypercorn api:app` |

We run the **simulation** to benchmark schedulers; the **API** is used to deploy them. Both share the same schedulers, rate limiter, and cost tracker implementations.

See [docs/architecture.md](docs/architecture.md) for structure and [docs/API.md](docs/API.md) for the deployment API reference.

## Overview

The system simulates a realistic multi-tenant LLM workload: multiple research sessions arrive over time, each spawning a tool-calling agent that fans out parallel LLM calls competing for a shared, rate-limited API. A pluggable scheduler sits between tools and the API, controlling dispatch order. By running the same workload through different schedulers, we can empirically compare their effect on latency, throughput, and fairness.

**Model**: OpenAI `gpt-4.1-nano`

## Agent Architecture

Each research session runs a deterministic **strict pipeline** with exactly 11 LLM calls:

```
run_agent(prompt)
  Step 1:  Orchestrator plans the research                    (1 LLM call)
  Step 2:  5 analysts run in parallel (fan-out)               (5 LLM calls)
  Step 3:  Orchestrator drafts synthesis from findings         (1 LLM call)
  Step 4:  3 reviewers run in parallel (fan-out)              (3 LLM calls)
  Step 5:  Orchestrator produces final synthesis               (1 LLM call)
```

Steps 3 and 5 are **barrier points** — they cannot start until the entire preceding fan-out completes. A single slow analyst in step 2 blocks the entire session from reaching step 3. This fan-out/fan-in structure is the key bottleneck that schedulers must optimize.

### Tools

Each tool is a single LLM call with a specialized system prompt. Tools are designed with **deliberately varied output lengths** so that different tools produce meaningfully different response sizes — this creates heterogeneous workloads that exercise scheduler prioritization.

**9 Analysts** (each takes a `question` parameter), ordered by expected output length:
- `analyst_statistical` — short numbered list of data points only
- `analyst_summarizer` — brief 5-7 bullet executive summary (<200 words)
- `analyst_counterarguments` — focused 3-5 counterarguments
- `analyst_web_research` — moderate structured research briefing
- `analyst_comparative` — detailed multi-approach comparison with table
- `analyst_trend_forecast` — multi-horizon forward-looking analysis
- `analyst_deep_analysis` — exhaustive mechanism/causal examination
- `analyst_historical_context` — comprehensive historical timeline
- `analyst_expert_synthesis` — long-form technical review

**3 Reviewers** (each takes a `text` parameter), ordered by expected output length:
- `review_style` — quick pass, only flags significant issues
- `review_citations` — thorough check of every factual claim's sourcing
- `review_facts` — rigorous evaluation of every claim with verdicts and confidence

### Batch Runner (in-process)

`python -m sim.runner` orchestrates multi-scheduler comparison runs:

1. **Workload generation** — pre-generates a balanced set of prompts and arrival times from a fixed seed. The same workload is reused for every scheduler.
2. **Simulation types** — sessions arrive according to one of two patterns:
   - `constant` — uniform spacing (e.g., every 4s), simulating steady predictable load
   - `bursty` — bursts of 2-5 sessions with 15-35s gaps between bursts, simulating traffic spikes
3. **Per-scheduler run** — resets rate limiter and trace, launches all sessions with staggered arrivals, collects results.
4. **Comparison** — prints side-by-side table of metrics across all schedulers.

### API Runner (over HTTP)

`python -m sim.api_runner` drives the **strict** through the scheduling proxy (api.py) over HTTP. Each session runs as its own HTTP client, modeled as a separate machine.

**Setup**: Start the API server in one terminal, run the benchmark in another. Requires `PROXY_API_KEY` in `.env` for auth.

```bash
# Terminal 1: start the proxy
hypercorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: run the benchmark
python -m sim.api_runner --sessions 30 --schedulers fifo mapreduce mapreduce_skip_adaptive --max-tokens 2048
```

### Accessing the API

| Context | Base URL | Example |
|---------|----------|---------|
| **Local** | `http://localhost:8000` | `--base-url http://localhost:8000` |
| **Remote** | `https://llmgateway.app` | `--base-url https://llmgateway.app` |

Use the same endpoints for both. Authentication via `Authorization: Bearer <PROXY_API_KEY>` is required. Obtain the proxy key from the API admin before using. 

## Rate Limiter

`sim/rate_limiter.py` enforces dual token-bucket rate limits:

- **RPM (requests per minute)** — bucket refills at `rpm/60` per second
- **TPM (tokens per minute)** — bucket refills at `tpm/60` per second

Key behavior:
- `try_acquire(estimated_tokens)` — returns `None` if capacity available, or throttle reason
- `record_actual_usage(prompt, completion, estimated)` — reconciles actual vs estimated tokens
- `wait_time(estimated_tokens)` — seconds until capacity is available
- `available_capacity()` — snapshot of current RPM/TPM bucket levels (used by skip-based schedulers)

## Cost Tracking

`sim/cost_tracker.py` enforces a hard budget per scheduler run:

- Input: $0.10 / 1M tokens, Output: $0.40 / 1M tokens
- Pre-flight `check()` before each LLM call raises `CostLimitExceeded` if over budget
- Post-call `record()` updates totals with actual usage
- Total budget is divided equally across schedulers in a comparison run

## Schedulers

All 5 active schedulers implement the same interface:

```python
def start()                                    # start background drain task
async def stop()                               # graceful shutdown
async def submit(coro_factory, est_tokens,     # submit an LLM call
                 session_id, call_key,
                 label="") -> response
```

Parameters:
- `session_id: int` — which session this call belongs to (used for MapReduce priority)
- `call_key: str` — call type identifier, e.g. `"orchestrator:plan"`, `"analyst_web_research"` (used for token-learning EMA and logging)
- `label: str` — optional human-readable context for log lines

### 1. Backoff

**Strategy**: Decentralized — no shared queue. Each call retries independently with exponential backoff (1s -> 2s -> 4s -> ... -> 30s cap) plus random jitter on rate limit rejection. Agents don't coordinate.

### 2. FIFO

**Strategy**: Global first-in-first-out queue with a single drain task. Fair and predictable, but large calls can cause head-of-line blocking — the entire queue stalls when the front item can't fit the TPM budget.

### 3. MapReduce

**Strategy**: Group-completion-aware priority. Priority = `1 / active_calls_in_session`. As fan-out members complete, remaining members' priority increases (e.g., last analyst gets priority 1.0 vs 1/5 when all 5 were active). Session identity is provided directly via `session_id`. Accelerates fan-in barrier completion.

### 4. MapReduce Skip

**Strategy**: Combines MapReduce group-completion priority with TPM-aware skipping. When the highest-priority item can't fit the current TPM budget, dispatches the highest-priority item that *does* fit, using a bisect-sorted index for O(log n) lookup. Also uses learned output-token EMA for tighter rate-limiter reservations instead of worst-case `max_tokens`. Interruptible sleeps wake on new submissions for faster response.

### 5. MapReduce Skip Adaptive

**Strategy**: Extends MapReduce Skip with output-aware secondary priority. When multiple calls share the same MapReduce straggler priority (e.g., five analysts from one session, all at priority 1/5), the call with the lowest predicted output-token count is dispatched first. Shorter calls complete sooner, freeing rate-limiter capacity faster and reducing mean session time. Priority key: `(-mapreduce_priority, predicted_output_tokens, enqueue_order)`.

## Exploration & Learning

During development we built, tested, and ultimately deprecated 6 additional schedulers. The results below are from a 30-session run (RPM=30, TPM=200k, seed=42, max_tokens=1024, strict mode with uniform analyst prompts):

| Rank | Scheduler | Mean (s) | Median (s) | P95 (s) | Stdev (s) | RPM Throttles | Parallelism | Max Queue Wait (s) |
|------|-----------|----------|------------|---------|-----------|---------------|-------------|-------------------|
| 1 | MapReduce | 280.9 | 279.9 | 489.9 | 140.8 | 284 | 4.15 | 470.5 |
| 2 | MapReduce Improved | 285.5 | 284.8 | 484.0 | 141.2 | 285 | 4.88 | 469.6 |
| 3 | Combined MR + Token SJF | 318.0 | 316.3 | 503.7 | 134.2 | 284 | 4.10 | 460.3 |
| 4 | Combined MR + Adaptive SJF | 330.6 | 346.6 | 498.8 | 133.7 | 277 | 5.52 | 465.8 |
| 5 | Token SJF Skip | 380.6 | 408.6 | 502.0 | 129.0 | 0 | 5.35 | 452.1 |
| 6 | Backoff | 402.6 | 442.5 | 558.9 | 124.6 | 1207 | 4.76 | 0.0 |
| 7 | FIFO | 420.9 | 494.9 | 515.6 | 142.8 | 293 | 5.04 | 162.0 |
| 8 | Adaptive SJF | 429.0 | 494.6 | 512.9 | 141.5 | 287 | 4.10 | 288.0 |
| 9 | Token SJF | 479.7 | 497.3 | 505.6 | 84.9 | 290 | 3.78 | 462.4 |
| 10 | SJF | 522.1 | 518.5 | 597.1 | 45.8 | 291 | 4.06 | 567.4 |

### Deprecated Schedulers and Why They Underperformed

**SJF (Shortest Job First)** — `schedulers/deprecated/sjf_deprecated.py`
Static priority queue ordered by estimated input token count. Dispatches the smallest request first. Suffered from **starvation**: orchestrator calls grow in token size as they accumulate analyst/reviewer results, so they are perpetually deprioritized behind every small analyst call from every session. Max queue wait of 567s confirms large calls waited nearly the entire run. The low stdev (45.8s) means all sessions were equally slow — none could complete because their orchestrator calls were starved. Worst performer overall.

**Token SJF** — `schedulers/deprecated/token_sjf_deprecated.py`
SJF with learned output-token EMA for priority and tighter rate-limiter reservations. Improved on static SJF by learning actual output sizes, but still suffered from **head-of-line blocking**: when the top-priority item couldn't fit the TPM budget, the entire queue blocked waiting for capacity. Parallelism dropped to 3.78 (lowest of all schedulers) because no work could proceed while waiting. No group awareness meant fan-out stragglers got no boost.

**Token SJF Skip** — `schedulers/deprecated/token_sjf_skip_deprecated.py`
Extended Token SJF with TPM-aware skipping — when the top item can't fit the TPM budget, dispatches a smaller item that does fit. This eliminated head-of-line blocking (0 RPM throttles, 5.35 parallelism). However, it still lacked group awareness, so a session's last analyst got no priority boost over fresh analysts from other sessions. The skip mechanism alone improved throughput but couldn't optimize for session completion time.

**Adaptive SJF** — `schedulers/deprecated/` (in `combined_mapreduce_tsjf_deprecated.py` as a component)
SJF with learned wall-clock duration EMA instead of token counts. Duration-based prediction is inherently noisier than token-based (includes network latency, server queuing, variable overhead), and in this workload all calls within the same phase have similar durations. The EMA converges quickly, so all analysts get similar predicted durations, making it effectively FIFO-with-extra-steps among same-type calls. No group awareness.

**Combined MapReduce + Token SJF** — `schedulers/deprecated/combined_mapreduce_tsjf_deprecated.py`
Score = `predicted_output_tokens x active_members_in_group` (lower = better). Aimed to combine the best of both: group-completion priority and short-job-first ordering. But the token estimation component **diluted the group-completion signal** — a short-output call in a large group could be prioritized over a straggler that was the last in its group. With the initial uniform analyst prompts (all producing similar-length outputs), the token prediction added noise without value. The `token_overestimate_ratio` of 1.30 indicates the learned estimates were imperfect.

**Combined MapReduce + Adaptive SJF** — `schedulers/deprecated/combined_mapreduce_tsjf_deprecated.py`
Score = `predicted_duration x active_members_in_group`. Same idea as the token variant but using wall-clock duration EMA. Performed worse because duration predictions are noisier and less correlated with the actual resource consumed (tokens) by the rate limiter — the scheduler optimized a proxy that didn't align with the constraint.

**MapReduce Improved vs MapReduce (original)** — Both survived but were consolidated. The original required callers to manually register and deregister fan-out groups; the improved version inferred groups automatically. In the strict pipeline where phases are sequential within a session, these produce identical behavior — the ~5s difference (1.6%) was noise. We kept the improved version, renamed it to "MapReduce", and now pass `session_id: int` directly so the scheduler tracks active calls per session without any lifecycle boilerplate.

### Key Insight

The dominant factor in this workload is **fan-out/fan-in barrier completion**: a session cannot progress until its last fan-out member finishes. Schedulers that directly prioritize group stragglers (MapReduce) outperform those that optimize individual call metrics (SJF variants). The skip mechanism from Token SJF Skip is valuable for avoiding head-of-line blocking but is orthogonal to group awareness — combining both (MapReduce Skip) should capture the benefits of each.

The initial experiments used **uniform analyst prompts** that produced similar-length outputs, which limited the benefit of SJF-style prioritization. The current version uses deliberately varied prompts (short bullet lists vs. exhaustive analyses) to create heterogeneous workloads that better exercise priority-based scheduling.

## Setup

### Environment

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
PROXY_API_KEY=...   # Required for api.py and sim.api_runner
```

### Install

```bash
pip install -r requirements.txt
```

## Usage

### Single session

```bash
python main.py
python main.py --prompt "quantum error correction" --scheduler fifo --rpm 20
python main.py --random --scheduler mapreduce
```

### Batch comparison

```bash
# Run all 5 schedulers and compare
python -m sim.runner --all-schedulers --sessions 15 --seed 42

# Run specific schedulers
python -m sim.runner --schedulers fifo mapreduce mapreduce_skip --sessions 10

# Single scheduler with bursty arrivals
python -m sim.runner --scheduler fifo --sessions 15 --stagger-mode bursty

# Save results to timestamped directory
python -m sim.runner --all-schedulers --sessions 30 --output-dir results/
```

### API-based batch comparison

```bash
# Terminal 1: start the API server
hypercorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: run api_runner (same CLI as sim.runner)
python -m sim.api_runner --all-schedulers --sessions 15 --seed 42
python -m sim.api_runner --schedulers fifo mapreduce --sessions 10 --base-url http://localhost:8000
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--scheduler` | `fifo` | Single scheduler to run |
| `--schedulers` | -- | List of schedulers to compare |
| `--all-schedulers` | -- | Run all 5 schedulers |
| `--sessions` | `15` | Number of research sessions |
| `--stagger` | `4.0` | Mean seconds between arrivals |
| `--stagger-mode` | `constant` | `constant` (uniform arrivals) or `bursty` (wave clusters) |
| `--rpm` | `30` | Requests per minute limit |
| `--tpm` | `200000` | Tokens per minute limit |
| `--cost-limit` | `20.0` | Total cost cap in USD (split across schedulers) |
| `--max-tokens` | `2048` | Max tokens per completion |
| `--seed` | `42` | RNG seed for reproducibility |
| `--prompt-mode` | `strict` | `default` (LLM picks tools) or `strict` (fixed pipeline) |
| `--output-dir` | -- | Directory for output files (auto-creates timestamped subdir) |

## Architecture

See [docs/architecture.md](docs/architecture.md) for full structure. Summary:

```
main.py                      Single-session entry point
api.py                       Deployment server — scheduling proxy for multi-client use
api_example.py               Example API client — strict pipeline over HTTP
agent.py                     Research agent — strict pipeline (11 LLM calls)
llm.py                       LLM call dispatcher (routes through scheduler)
tools/
  analysts.py                9 analyst tools (varied output lengths)
  reviewers.py               3 reviewer tools (varied output lengths)
prompts/
  system.py                  Orchestrator agent system prompt
  topics.py                  Research topic list (5 topics)
schedulers/
  backoff.py                 Exponential backoff (no queue)
  fifo.py                    FIFO queue
  mapreduce.py               Group-aware dynamic priority (auto-inferred sessions)
  mapreduce_skip.py          MapReduce + TPM-aware skipping + learned tokens
  mapreduce_skip_adaptive.py MapReduce Skip + output-aware secondary priority
  deprecated/                Explored and deprecated schedulers (see above)
sim/
  runner.py                  Batch experiment runner (python -m sim.runner)
  api_runner.py              API-based benchmark (python -m sim.api_runner)
  workload.py                Workload generation (arrivals + prompt assignment)
  rate_limiter.py            Token-bucket rate limiter (RPM + TPM)
  cost_tracker.py            Budget tracking and kill switch
  trace.py                   Call logging and timeline visualization
  metrics.py                 Session statistics and comparison tables
```
