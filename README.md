# CS244 - LLM Agent Scheduling Strategies

A multi-agent research system that studies how different scheduling strategies affect performance when multiple LLM-powered agents compete for rate-limited API access.

## Overview

The system takes a research topic, breaks it into 5 sub-questions, and dispatches 5 parallel agents to investigate them. Each agent can use tools (web search, summarize, analyze) that themselves make LLM calls. Results are synthesized into a final answer. All LLM calls are routed through a shared scheduler that manages rate limits.

## Agent Loop

Each agent runs a tool-use loop for up to 3 rounds:

```
1. Send messages to LLM (with tool schemas)
2. If LLM returns tool calls → execute tools, append results, go to 1
3. If LLM returns text → return as final answer
```

The orchestrator flow:
```
orchestrate(topic)
  → LLM breaks topic into 5 sub-questions
  → 5 agents run in parallel (asyncio.gather)
  → each agent loops through tool-use rounds
  → LLM synthesizes all agent findings into final answer
```

## Schedulers

All three schedulers sit between agents and the OpenAI API, controlling how rate-limited calls are dispatched:

### Backoff
Each agent calls the API directly. On rate limit rejection, it retries with exponential backoff (1s → 2s → 4s → ... → 30s cap) plus random jitter. Simple but decentralized — agents don't coordinate, which can lead to wasted retries.

### FIFO
All calls go into a global FIFO queue. A single drain task processes them in order, waiting for rate limit capacity before dispatching each call. Fair and controlled, but large calls can block smaller ones behind them.

### SJF (Shortest Job First)
Like FIFO but uses a priority queue ordered by estimated token count (smaller jobs first). Optimizes throughput by preventing short requests from getting stuck behind long ones. May starve large requests under heavy load.

## Setup

### Environment

Create a `.env` file with your OpenAI API key:

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

### Batch run (multiple parallel sessions)

```bash
python batch_runner.py --batches 1 --sessions 5 --scheduler backoff --rpm 60 --stagger 0.5
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--scheduler` | `fifo` | `backoff`, `fifo`, or `sjf` |
| `--rpm` | 20 (main) / 60 (batch) | Requests per minute limit |
| `--tpm` | 100000 / 200000 | Tokens per minute limit |
| `--cost-limit` | 15.0 | Hard cost cap in USD |
| `--batches` | 5 | Number of sequential batches (batch runner) |
| `--sessions` | 30 | Sessions per batch (batch runner) |
| `--stagger` | 1.0 | Seconds between session starts within a batch |

## Architecture

```
main.py / batch_runner.py    Entry points
orchestrator.py              Breakdown → fan-out → synthesis
agent.py                     Tool-use agent loop
llm.py                       LLM call dispatcher (routes through scheduler)
schedulers/
  backoff.py                 Exponential backoff strategy
  fifo.py                    FIFO queue strategy
  sjf.py                     Shortest-job-first strategy
tools.py                     Tool schemas and execution
rate_limiter.py              RPM/TPM sliding window enforcement
cost_tracker.py              Budget tracking and kill switch
trace.py                     Call logging and timeline visualization
metrics.py                   Session duration statistics
prompts.py                   Research topic list for batch runs
```

The model used is `gpt-4.1-nano`.
