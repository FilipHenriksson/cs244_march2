import time
from dataclasses import dataclass


@dataclass
class LLMCall:
    call_id: int
    agent_id: str
    call_type: str
    start: float
    end: float = 0.0
    detail: str = ""
    queue_wait: float = 0.0
    queue_position: int = 0
    estimated_tokens: int = 0
    retries: int = 0
    rpm_waits: int = 0
    tpm_waits: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


class Trace:
    def __init__(self):
        self.calls: list[LLMCall] = []
        self.t0 = time.time()
        self._next_id = 0

    def _elapsed(self, t: float) -> float:
        return t - self.t0

    def start_call(self, agent_id: str, call_type: str, detail: str = "",
                   queue_wait: float = 0.0, queue_position: int = 0,
                   estimated_tokens: int = 0, retries: int = 0,
                   rpm_waits: int = 0, tpm_waits: int = 0) -> LLMCall:
        call = LLMCall(
            call_id=self._next_id,
            agent_id=agent_id,
            call_type=call_type,
            start=time.time(),
            detail=detail,
            queue_wait=queue_wait,
            queue_position=queue_position,
            estimated_tokens=estimated_tokens,
            retries=retries,
            rpm_waits=rpm_waits,
            tpm_waits=tpm_waits,
        )
        self._next_id += 1
        self.calls.append(call)
        tag = f"[{agent_id}]"
        label = f"{call_type}({detail})" if detail else call_type
        extras = []
        if queue_wait > 0.01:
            extras.append(f"waited {queue_wait:.2f}s")
        if retries > 0:
            extras.append(f"{retries} retries")
        if rpm_waits > 0:
            extras.append(f"{rpm_waits} rpm throttles")
        if tpm_waits > 0:
            extras.append(f"{tpm_waits} tpm throttles")
        if queue_position > 0:
            extras.append(f"pos #{queue_position}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  {self._elapsed(call.start):6.2f}s {tag:<16} START {label}{extra_str}")
        return call

    def end_call(self, call: LLMCall):
        call.end = time.time()
        tag = f"[{call.agent_id}]"
        label = f"{call.call_type}({call.detail})" if call.detail else call.call_type
        print(f"  {self._elapsed(call.end):6.2f}s {tag:<16} END   {label} ({call.duration:.2f}s)")

    def print_summary(self, rl_stats: dict = None):
        total = time.time() - self.t0
        print(f"\n{'='*70}")
        print(f" TRACE SUMMARY")
        print(f"{'='*70}")
        print(f" Total wall time: {total:.2f}s")
        print(f" Total LLM calls: {len(self.calls)}")

        # Calls per agent
        agents = sorted(set(c.agent_id for c in self.calls))
        print(f"\n Per agent:")
        for a in agents:
            ac = [c for c in self.calls if c.agent_id == a]
            llm_t = sum(c.duration for c in ac)
            retries = sum(c.retries for c in ac)
            extra = f", {retries} retries" if retries else ""
            print(f"   {a:<28} {len(ac)} calls, {llm_t:.2f}s LLM{extra}")

        # Calls per type
        types = sorted(set(c.call_type for c in self.calls))
        print(f"\n Per call type:")
        for t in types:
            tc = [c for c in self.calls if c.call_type == t]
            print(f"   {t:<28} {len(tc)} calls, {sum(c.duration for c in tc):.2f}s total")

        # Retry stats (backoff scheduler)
        total_retries = sum(c.retries for c in self.calls)
        if total_retries > 0:
            print(f"\n Backoff stats:")
            print(f"   Total retries:      {total_retries}")
            max_retries = max(c.retries for c in self.calls)
            print(f"   Max retries (call): {max_retries}")

        # Queue stats
        queued_calls = [c for c in self.calls if c.queue_position > 0]
        waited_calls = [c for c in self.calls if c.queue_wait > 0.01]
        if queued_calls:
            print(f"\n Queue stats:")
            print(f"   Queued calls: {len(queued_calls)}/{len(self.calls)}")
            if waited_calls:
                total_wait = sum(c.queue_wait for c in waited_calls)
                print(f"   Calls that waited:  {len(waited_calls)}")
                print(f"   Total queue delay:  {total_wait:.2f}s")
                print(f"   Mean queue delay:   {total_wait / len(waited_calls):.2f}s")
                print(f"   Max queue delay:    {max(c.queue_wait for c in waited_calls):.2f}s")
                types_waited = sorted(set(c.call_type for c in waited_calls))
                if len(types_waited) > 1:
                    print(f"   By call type:")
                    for t in types_waited:
                        tw = [c for c in waited_calls if c.call_type == t]
                        total_t = sum(c.queue_wait for c in tw)
                        print(f"     {t:<28} {len(tw)} waited, "
                              f"{total_t:.2f}s total, max {max(c.queue_wait for c in tw):.2f}s")
                total_rpm = sum(c.rpm_waits for c in self.calls)
                total_tpm = sum(c.tpm_waits for c in self.calls)
                if total_rpm > 0 or total_tpm > 0:
                    print(f"   Throttle events:    {total_rpm} RPM, {total_tpm} TPM")
            else:
                print(f"   No calls waited for capacity")

        # Rate limiter stats
        if rl_stats:
            print(f"\n Rate limiter stats:")
            print(f"   Successful acquires: {rl_stats['total_acquires']}")
            print(f"   RPM throttles:       {rl_stats['rpm_throttles']}")
            print(f"   TPM throttles:       {rl_stats['tpm_throttles']}")
            if rl_stats['total_actual'] > 0:
                print(f"   Token estimation:")
                print(f"     Total estimated:   {rl_stats['total_estimated']:,}")
                print(f"     Total actual:      {rl_stats['total_actual']:,}")
                overest = rl_stats['token_overestimate']
                ratio = rl_stats['total_estimated'] / rl_stats['total_actual']
                print(f"     Overestimate:      {overest:,} ({ratio:.1f}x)")

        # Totals
        total_llm = sum(c.duration for c in self.calls)
        print(f"\n Total LLM time (sum): {total_llm:.2f}s")
        print(f" Wall time:            {total:.2f}s")
        print(f" Parallelism ratio:    {total_llm / total:.1f}x")

        # Timeline
        print(f"\n{'='*70}")
        print(f" TIMELINE")
        print(f"{'='*70}")
        width = 60
        for c in sorted(self.calls, key=lambda c: c.start):
            s = self._elapsed(c.start)
            e = self._elapsed(c.end)
            tag = f"{c.agent_id}:{c.call_type}"
            if c.detail:
                tag += f"({c.detail[:20]})"
            bar_start = int(s / total * width)
            bar_end = max(bar_start + 1, int(e / total * width))
            bar = " " * bar_start + "#" * (bar_end - bar_start) + " " * (width - bar_end)
            print(f" {s:5.1f}s |{bar}| {e:5.1f}s  {tag}")

        print(f"        {'0':}<{width}>{total:.1f}s")
        print(f"{'='*70}\n")

    def get_stats(self) -> dict:
        """Return computed metrics as a dict for comparison tables."""
        total_wall = time.time() - self.t0
        total_llm = sum(c.duration for c in self.calls)
        waited = [c for c in self.calls if c.queue_wait > 0.01]
        return {
            "wall_time": total_wall,
            "total_llm_calls": len(self.calls),
            "total_llm_time": total_llm,
            "parallelism": total_llm / total_wall if total_wall > 0 else 0,
            "mean_queue_wait": (sum(c.queue_wait for c in waited) / len(waited)) if waited else 0,
            "max_queue_wait": max((c.queue_wait for c in waited), default=0),
        }

    def reset(self):
        """Clear all call history for a fresh run."""
        self.calls.clear()
        self._next_id = 0
        self.t0 = time.time()


# Global trace instance
trace = Trace()
