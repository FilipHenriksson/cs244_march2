# CS244 - LLM Agent Scheduling Strategies

A multi-agent research system that studies how different scheduling strategies affect performance when multiple LLM-powered agents compete for rate-limited API access.

## Overview

The system simulates a realistic multi-tenant LLM workload: multiple research sessions arrive over time, each spawning parallel agents that compete for a shared, rate-limited API. A pluggable scheduler sits between agents and the API, controlling dispatch order. By running the same workload through different schedulers, we can empirically compare their effect on latency, throughput, and fairness.

**Model**: OpenAI `gpt-4.1-nano`

## Simulation Pipeline

Each research session follows a fan-out/fan-in pattern:

```
orchestrate(topic)
  1. Breakdown     — LLM splits topic into 5 sub-questions
  2. Fan-out       — 5 agents research in parallel (asyncio.gather)
  3. Synthesis     — LLM combines all agent findings
  4. Review        — 3 reviewers run in parallel (citations, style, facts)
  5. Finalization  — LLM integrates reviewer feedback into polished output
```

### Agent Behavior

Each agent runs a tool-use loop (up to 3 rounds):

```
1. Send messages + tool schemas to LLM
2. If LLM returns tool calls → execute tools, append results, go to 1
3. If LLM returns text → return as final answer
```

Available tools (all simulated via LLM):
- **web_search(query)** — returns 3–5 search result snippets
- **summarize(text)** — distills text into key points
- **analyze(text, question)** — answers a specific question about text

### Batch Runner

`batch_runner.py` orchestrates multi-scheduler comparison runs:

1. **Workload generation** — pre-generates a balanced set of prompts (equal copies of each research topic, shuffled) and arrival times. The same workload is reused for every scheduler.
2. **Arrival modes** — sessions arrive according to one of three patterns:
   - `fixed` — uniform spacing (e.g., every 4s)
   - `poisson` — exponentially distributed inter-arrivals
   - `wave` — bursts of 2–5 sessions with 15–35s gaps between bursts
3. **Per-scheduler run** — resets rate limiter and trace, launches all sessions with staggered arrivals, collects results.
4. **Fairness controls** — scheduler order is shuffled (seeded) to avoid time-of-day bias; 30s cooldown between runs.
5. **Comparison** — prints side-by-side table of metrics across all schedulers.

## Rate Limiter

`rate_limiter.py` enforces OpenAI-style sliding-window rate limits:

- **RPM (requests per minute)** — tracks request timestamps in a 60-second rolling window
- **TPM (tokens per minute)** — tracks `(timestamp, token_count)` tuples in a 60-second rolling window

Key behavior:
- `try_acquire(estimated_tokens)` — returns True if both RPM and TPM allow the request; logs estimated tokens
- `report_actual(estimated, actual)` — called after each API response to correct the token window with real usage from `response.usage`
- `wait_time()` — estimates seconds until the oldest entry expires from the window
- Token estimates use `len(json.dumps(messages)) // 4` as a rough heuristic, then get corrected post-call

## Cost Tracking

`cost_tracker.py` enforces a hard budget per scheduler run:

- Input: $0.10 / 1M tokens, Output: $0.40 / 1M tokens
- Pre-flight `check()` before each LLM call raises `CostLimitExceeded` if over budget
- Post-call `record()` updates totals with actual usage
- Total budget is divided equally across schedulers in a comparison run
- If any scheduler hits the limit, the batch runner prints a loud warning and flags results as invalid

## Schedulers

All six schedulers implement the same interface:

```python
def start()                                    # start background drain task
async def stop()                               # graceful shutdown
async def submit(coro_factory, est_tokens,     # submit an LLM call
                 agent_id, call_type, detail,
                 group_id=None) → response
def register_group(group_id, size)             # declare a fan-out group
def deregister_member(group_id)                # one group member completed
```

### 1. Backoff

**Strategy**: Decentralized — no shared queue.

Each agent calls the API directly. On rate limit rejection, retries with exponential backoff (`1s → 2s → 4s → ... → 30s` cap) plus random jitter. Agents don't coordinate, which leads to wasted retries under contention.

### 2. FIFO

**Strategy**: Global first-in-first-out queue with a single drain task.

All calls are enqueued in arrival order. The drain task processes them sequentially, waiting for rate limiter capacity before dispatching. Fair and predictable, but large calls can cause head-of-line blocking.

### 3. SJF (Shortest Job First)

**Strategy**: Priority queue ordered by estimated token count (ascending).

Like FIFO but dispatches smaller jobs first. Uses `asyncio.PriorityQueue` with `(estimated_tokens, insertion_order)` as the key. Optimizes throughput for short requests but may starve large ones under heavy load. Priority is static — set at enqueue time and never updated.

### 4. MapReduce

**Strategy**: Dynamic group-aware priority.

Tracks fan-out groups registered by the orchestrator. Priority is computed as `1 / active_members_in_group`:

- Group of 5 agents → each starts at priority 0.2
- As members complete → remaining members' priority increases
- Last member gets priority 1.0 (same as singletons)
- Singleton calls (synthesis, finalization) always get priority 1.0

Re-picks the best job during rate limit waits, since group membership can change while sleeping. This accelerates fan-in phases — the last agent to finish gets boosted to prevent it from blocking synthesis.

### 5. Adaptive SJF

**Strategy**: SJF with learned duration prediction via EMA.

Instead of using static token estimates, learns actual wall-clock completion times per call type using an exponential moving average (α = 0.3). Prioritizes calls with the shortest predicted duration.

Key design choices:
- `agent_turn` calls are tracked per-round (`agent_turn:round-0`, `agent_turn:round-1`, etc.) since durations vary significantly by round
- **Unseen call types get priority 0.0** (highest) — exploration over exploitation
- Re-picks during rate limit waits as EMA updates from concurrent completions

### 6. Combined (MapReduce + Adaptive SJF)

**Strategy**: Merges group-aware priority with learned durations.

Priority score: `predicted_duration × active_members_in_group` (lower = better)

- Short jobs in nearly-complete groups get top priority
- Singletons (reduce-phase calls): `pred × 1`, no penalty
- Unseen call types: `0.0 × anything = 0.0`, exploration-first preserved
- As group members complete, remaining members' scores decrease (priority rises)

Combines the fan-in acceleration of MapReduce with the workload-adaptive scheduling of Adaptive SJF.

## Results

15 sessions, seed=42, stagger=4s fixed, rpm=30, tpm=200k:

```
Metric       |    mapreduce |     combined | adaptive_sjf |      backoff |         fifo |          sjf
----------------------------------------------------------------------------------------------------------
mean         |      645.64s |      661.86s |      698.61s |      715.74s |      719.52s |      725.64s
median       |      708.15s |      712.96s |      747.09s |      752.08s |      734.64s |      714.20s
p95          |      762.20s |      756.18s |      791.82s |      821.17s |      793.45s |      793.22s
stdev        |      162.54s |      125.53s |      137.26s |       94.69s |       70.98s |       38.43s
cost         |      $0.0869 |      $0.0864 |      $0.0854 |      $0.0853 |      $0.0843 |      $0.0869
```

- **mapreduce** has the lowest mean latency but highest variance
- **combined** has the best p95 and lower variance than both parent schedulers
- **sjf** is the most consistent (lowest stdev) but slowest on average
- Cost is nearly identical across all schedulers (~$0.085)

## Setup

### Environment

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

### Install

```bash
pip install -r requirements.txt
```

## Usage

### Single session

```bash
python main.py --prompt "quantum error correction" --scheduler fifo --rpm 20
```

### Batch comparison

```bash
# Run all 6 schedulers and compare
python batch_runner.py --all-schedulers --sessions 15 --seed 42

# Run specific schedulers
python batch_runner.py --schedulers fifo mapreduce combined --sessions 15

# Single scheduler
python batch_runner.py --scheduler combined --sessions 15
```

### CLI options (batch_runner.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--scheduler` | `fifo` | Single scheduler to run |
| `--schedulers` | — | List of schedulers to compare |
| `--all-schedulers` | — | Run all 6 schedulers |
| `--sessions` | `15` | Number of research sessions |
| `--stagger` | `4.0` | Mean seconds between arrivals |
| `--stagger-mode` | `fixed` | `fixed`, `poisson`, or `wave` |
| `--rpm` | `30` | Requests per minute limit |
| `--tpm` | `200000` | Tokens per minute limit |
| `--cost-limit` | `50.0` | Total cost cap in USD (split across schedulers) |
| `--seed` | `42` | RNG seed for reproducibility |

## Architecture

```
main.py                      Single-session entry point
batch_runner.py              Multi-scheduler benchmark runner
orchestrator.py              Breakdown → fan-out → synthesis → review → finalize
agent.py                     Tool-use agent loop (up to 3 rounds)
llm.py                       LLM call dispatcher (routes through scheduler)
schedulers/
  backoff.py                 Exponential backoff strategy
  fifo.py                    FIFO queue strategy
  sjf.py                     Shortest-job-first strategy
  mapreduce.py               Group-aware dynamic priority
  adaptive_sjf.py            SJF with learned duration prediction
  combined.py                MapReduce + Adaptive SJF hybrid
tools.py                     Tool schemas and execution (simulated via LLM)
rate_limiter.py              RPM/TPM sliding window enforcement
cost_tracker.py              Budget tracking and kill switch
trace.py                     Call logging and timeline visualization
metrics.py                   Session duration statistics and comparison tables
prompts.py                   Research topic list (5 topics)
```
