# Architecture

## Directory Structure

```
cs244_march2/
├── main.py                 # Single-session entry point
├── agent.py                # Research agent — strict pipeline (11 LLM calls)
├── llm.py                  # LLM call dispatch (single entry point for all API calls)
│
├── tools/                  # Agent tools (called via OpenAI function calling)
│   ├── __init__.py         # Tool registry + execute_tool() dispatcher
│   ├── analysts.py         # 9 analyst tools (varied output lengths)
│   └── reviewers.py        # 3 reviewer tools (varied output lengths)
│
├── prompts/                # All prompt definitions
│   ├── __init__.py         # Re-exports
│   ├── system.py           # Orchestrator agent system prompt
│   └── topics.py           # Research topic prompts (workload inputs)
│
├── schedulers/             # Pluggable scheduling policies
│   ├── __init__.py         # get_scheduler() factory (5 active schedulers)
│   ├── backoff.py          # Exponential backoff (no queue)
│   ├── fifo.py             # Global FIFO queue
│   ├── mapreduce.py        # Group-aware dynamic priority (auto-inferred sessions)
│   ├── mapreduce_skip.py   # MapReduce + TPM-aware skipping + learned tokens
│   ├── mapreduce_skip_adaptive.py  # MapReduce Skip + output-aware secondary priority
│   └── deprecated/         # Explored and deprecated schedulers
│       ├── sjf_deprecated.py
│       ├── token_sjf_deprecated.py
│       ├── token_sjf_skip_deprecated.py
│       ├── combined_mapreduce_tsjf_deprecated.py
│       └── mapreduce_deprecated.py
│
└── sim/                    # Simulation infrastructure
    ├── __init__.py
    ├── runner.py            # Batch experiment runner (python -m sim.runner)
    ├── workload.py          # Workload generation (arrival times, prompt assignment)
    ├── rate_limiter.py      # Token-bucket rate limiter (RPM + TPM)
    ├── cost_tracker.py      # Cost accumulation + hard budget enforcement
    ├── trace.py             # Call tracing, statistics, timeline visualization
    └── metrics.py           # Session results + scheduler comparison tables
```

## How It Works

### The Agent

`agent.py` runs a deterministic **strict pipeline** for each research session.
Every session produces exactly 11 LLM calls in a fixed fan-out/fan-in pattern:

```
User prompt
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  agent.py — strict pipeline (11 LLM calls per session)       │
│                                                              │
│  Step 1:  Orchestrator plans the research         (1 call)   │
│                                                              │
│  Step 2:  5 analysts run in parallel (fan-out)    (5 calls)  │
│           ── barrier: wait for all 5 ──                      │
│                                                              │
│  Step 3:  Orchestrator drafts synthesis            (1 call)  │
│                                                              │
│  Step 4:  3 reviewers run in parallel (fan-out)   (3 calls)  │
│           ── barrier: wait for all 3 ──                      │
│                                                              │
│  Step 5:  Orchestrator produces final synthesis    (1 call)  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Final output
```

The barriers at steps 3 and 5 are the critical bottleneck: a single slow
analyst/reviewer holds up the entire session. This is why group-aware
schedulers (MapReduce) outperform individual-call optimizers (SJF).

### Tools

Each tool is a single LLM call with a specialized system prompt. Prompts are
deliberately varied in scope and expected output length:

**9 Analysts** (`tools/analysts.py`): each takes a `question` parameter
- **Short output**: `analyst_statistical` (data list), `analyst_summarizer` (bullet summary), `analyst_counterarguments` (focused critique)
- **Medium output**: `analyst_web_research` (structured briefing), `analyst_comparative` (comparison with table), `analyst_trend_forecast` (multi-horizon analysis)
- **Long output**: `analyst_deep_analysis` (exhaustive mechanism analysis), `analyst_historical_context` (comprehensive timeline), `analyst_expert_synthesis` (long-form technical review)

**3 Reviewers** (`tools/reviewers.py`): each takes a `text` parameter
- **Short**: `review_style` — quick pass, significant issues only
- **Medium**: `review_citations` — thorough source attribution check
- **Long**: `review_facts` — rigorous evaluation of every claim with verdicts

This output-length variation creates heterogeneous workloads that exercise
scheduler prioritization of differently-sized jobs.

### LLM Dispatch

`llm.py` is the single entry point for all API calls. Every call — whether
from the orchestrator or a tool — goes through `llm_call()`, which:

1. Estimates token cost (scheduler-specific if `estimate_total_tokens` exists)
2. Pre-flight cost check (`sim/cost_tracker.py`)
3. Submits to the active scheduler (`schedulers/`)
4. Records actual cost after completion

Default max_tokens: 2048 (large enough to avoid truncation for long analysts).

### Schedulers

All 5 active schedulers implement the same interface:

```python
class Scheduler:
    def start(self)
    async def stop(self)
    async def submit(coro_factory, est_tokens, session_id, call_key, label="")
```

Parameters:
- `session_id: int` — which session this call belongs to (used for MapReduce priority)
- `call_key: str` — call type identifier, e.g. `"orchestrator:plan"`, `"analyst_web_research"` (used for token-learning EMA and logging)
- `label: str` — optional human-readable context for log lines

Swapped in at startup via `llm.set_scheduler()`. Group-aware schedulers
track active calls per session internally to boost priority for stragglers.

**Active schedulers:**

| Scheduler | Queue | Group-Aware | Token Learning | Skip | Output Priority |
|-----------|-------|-------------|----------------|------|-----------------|
| Backoff | No (inline retry) | No | No | N/A | No |
| FIFO | FIFO | No | No | No | No |
| MapReduce | Priority (MR) | Yes | No | No | No |
| MapReduce Skip | Priority (MR) | Yes | Yes (EMA) | Yes | No |
| MapReduce Skip Adaptive | Priority (MR) | Yes | Yes (EMA) | Yes | Yes |

### Simulation

The `sim/` package provides infrastructure for running experiments:

- **`runner.py`** — batch entry point (`python -m sim.runner`). Runs N sessions
  across one or more schedulers with identical workloads, then compares results.
- **`workload.py`** — deterministic prompt assignment and arrival time generation
  (constant or bursty simulation types).
- **`rate_limiter.py`** — token-bucket enforcing RPM and TPM limits. Provides
  both `try_acquire`/`wait_time` (used by blocking schedulers) and
  `available_capacity` (used by skip-based schedulers for budget queries).
- **`cost_tracker.py`** — running cost accumulator with a hard dollar cap.
- **`trace.py`** — records every LLM call with timing, queue stats, and
  generates a timeline visualization.
- **`metrics.py`** — session duration statistics and scheduler comparison tables.

## Running

Single session:
```bash
python main.py
python main.py --scheduler mapreduce --prompt "What is dark matter?"
```

Batch comparison:
```bash
python -m sim.runner --sessions 15 --scheduler fifo
python -m sim.runner --sessions 30 --all-schedulers --output-dir results/
python -m sim.runner --schedulers mapreduce mapreduce_skip --sessions 20 --stagger-mode bursty
```
