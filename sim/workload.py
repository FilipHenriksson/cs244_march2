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
        Wave-like clusters — bursts of 2-5 sessions arrive close together
        (0.5-2s apart), separated by quiet gaps (15-35s).  Simulates
        real-world traffic spikes.
    """
    if mode == "constant":
        return [i * stagger for i in range(n)]
    elif mode == "bursty":
        r = rng or random
        times = []
        t = 0.0
        remaining = n
        while remaining > 0:
            burst_size = min(r.randint(2, 5), remaining)
            for j in range(burst_size):
                times.append(t)
                t += r.uniform(0.5, 2.0)
            remaining -= burst_size
            if remaining > 0:
                t += r.uniform(15.0, 35.0)
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
