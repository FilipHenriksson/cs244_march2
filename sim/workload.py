"""Workload generation — deterministic prompt assignment and arrival times.

Pre-generates the full workload so every scheduler gets identical prompts
arriving at the same relative times, enabling fair side-by-side comparison.
"""

import random
from prompts import RESEARCH_PROMPTS


def compute_arrival_times(n: int, stagger: float, mode: str,
                          rng: random.Random = None) -> list[float]:
    """Return *n* cumulative arrival offsets (seconds from t=0).

    *stagger* is the mean inter-arrival gap.  *mode* selects the distribution:

    ``constant``
        Uniform spacing — sessions arrive every *stagger* seconds, producing
        a steady, predictable load.

    ``bursty``
        Wave-like clusters — bursts of 5-15% of sessions arrive nearly
        simultaneously (within *stagger* seconds of each other), separated
        by quiet gaps (3-6x the burst duration).  Scales naturally with
        session count: 200 sessions → ~8-12 bursts of 15-30 each.
        All arrivals are rescaled to fit within *n * stagger* seconds.
    """
    if mode == "constant":
        return [i * stagger for i in range(n)]
    elif mode == "bursty":
        r = rng or random
        times = []
        t = 0.0
        remaining = n
        lo = max(2, int(n * 0.05))
        hi = max(lo + 1, int(n * 0.15))
        while remaining > 0:
            burst_size = min(r.randint(lo, hi), remaining)
            for j in range(burst_size):
                times.append(t)
                t += r.uniform(stagger * 0.5, stagger * 1.5)
            remaining -= burst_size
            if remaining > 0:
                burst_dur = burst_size * stagger
                t += r.uniform(burst_dur * 3, burst_dur * 6)
        window = n * stagger
        max_t = times[-1] if times else 0.0
        if max_t > window and max_t > 0:
            scale = window / max_t
            times = [t * scale for t in times]
        return times
    else:
        raise ValueError(f"Unknown sim type: {mode}")


def generate_workload(sessions: int, stagger: float,
                      stagger_mode: str, seed: int) -> dict:
    """Pre-generate the full workload so every scheduler gets the same one."""
    rng = random.Random(seed)
    copies = sessions // len(RESEARCH_PROMPTS)
    remainder = sessions % len(RESEARCH_PROMPTS)
    prompts = RESEARCH_PROMPTS * copies + RESEARCH_PROMPTS[:remainder]
    rng.shuffle(prompts)
    arrivals = compute_arrival_times(sessions, stagger, stagger_mode, rng)
    return {"prompts": prompts, "arrivals": arrivals}
