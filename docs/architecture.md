# Architecture

Codebase structure and internal flow. For the deployment API, see [API.md](API.md).

## Directory Structure

```
cs244_march2/
├── main.py                 # Single-session entry point (simulation)
├── api.py                  # Deployment server — scheduling proxy (see API.md)
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
│
└── sim/                    # Simulation infrastructure
    ├── runner.py           # Batch experiment runner (python -m sim.runner)
    ├── api_runner.py       # API-based benchmark (python -m sim.api_runner)
    ├── workload.py         # Workload generation (arrival times, prompt assignment)
    ├── rate_limiter.py     # Token-bucket rate limiter (RPM + TPM)
    ├── cost_tracker.py     # Cost accumulation + hard budget enforcement
    ├── trace.py            # Call tracing, statistics, timeline visualization
    └── metrics.py          # Session results + scheduler comparison tables
```

## LLM Dispatch

`llm.py` is the single entry point for all API calls. Every call — whether from the orchestrator or a tool — goes through `llm_call()`, which:

1. Estimates token cost (scheduler-specific if `estimate_total_tokens` exists)
2. Pre-flight cost check (`sim/cost_tracker.py`)
3. Submits to the active scheduler (`schedulers/`)
4. Records actual cost after completion

## Scheduler Feature Matrix

| Scheduler | Queue | Group-Aware | Token Learning | Skip | Output Priority |
|-----------|-------|-------------|----------------|------|-----------------|
| Backoff | No (inline retry) | No | No | N/A | No |
| FIFO | FIFO | No | No | No | No |
| MapReduce | Priority (MR) | Yes | No | No | No |
| MapReduce Skip | Priority (MR) | Yes | Yes (EMA) | Yes | No |
| MapReduce Skip Adaptive | Priority (MR) | Yes | Yes (EMA) | Yes | Yes |
