"""Causal DSpark prefix scheduling, following arXiv:2607.05147 §3.2.2.

Costs are frozen before a measured run. Confidence k is conditional on the
already sampled prefix ending at k-1; it must not depend on candidate k.
The first non-improving prefix ends the search. A global argmax is invalid.
"""

from dataclasses import dataclass
import math
from typing import Iterable


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Decision:
    depth: int
    expected_tokens: float
    predicted_tokens_per_ms: float


@dataclass(frozen=True)
class PrefixPolicy:
    # Index zero includes a draft pass and one target token, with no drafts
    # accepted. Every other entry includes that same fixed-width draft pass.
    cycle_ms: tuple[float, ...]
    temperatures: tuple[float, ...] = ()

    def __post_init__(self):
        if len(self.cycle_ms) < 2 or any(
            not math.isfinite(x) or x <= 0 for x in self.cycle_ms
        ):
            raise ValueError("Need finite, positive cycle costs for depths 0..K")
        if self.temperatures and (
            len(self.temperatures) != len(self.cycle_ms) - 1
            or any(not math.isfinite(t) or t <= 0 for t in self.temperatures)
        ):
            raise ValueError("Need one finite, positive temperature per position")

    def choose(self, confidence_logits: Iterable[float]) -> Decision:
        survival, expected = 1.0, 1.0
        chosen = Decision(0, expected, expected / self.cycle_ms[0])
        for k, logit in enumerate(confidence_logits, 1):
            if k >= len(self.cycle_ms):
                break
            if not math.isfinite(logit):
                break  # Fail closed to the prefix already admitted.
            temperature = self.temperatures[k - 1] if self.temperatures else 1.0
            survival *= sigmoid(logit / temperature)
            expected += survival
            rate = expected / self.cycle_ms[k]
            if rate <= chosen.predicted_tokens_per_ms:
                break
            chosen = Decision(k, expected, rate)
        return chosen


class Experiment:
    """Explicitly opt-in, single-request instrumentation and prefix selection.

    With no forced width/policy, observe the existing acceptance controller.
    Profiling with a forced width and depth measures costs for PrefixPolicy.
    Event timings include the pending draft synchronization and target verify;
    they exclude queue emission between cycles. End-to-end timings remain the
    primary performance metric. No online timing feeds a decoding decision.
    """

    def __init__(self, *, draft_width=None, forced_depth=None, policy=None):
        if forced_depth is not None and policy is not None:
            raise ValueError("Choose forced depth OR confidence scheduling")
        if forced_depth is not None and not 0 <= forced_depth <= draft_width:
            raise ValueError("Forced depth must fit the fixed draft width")
        if policy is not None and len(policy.cycle_ms) != draft_width + 1:
            raise ValueError("Cost curve must cover the fixed draft width")
        self.draft_width = draft_width
        self.forced_depth = forced_depth
        self.policy = policy
        self.events = []
        self.record_events = True

    def select(self, logits, current_depth):
        if self.policy is not None:
            return min(current_depth, self.policy.choose(iter(logits)).depth)
        if self.forced_depth is not None:
            return min(current_depth, self.forced_depth)
        return current_depth

    def record(self, **event):
        if self.record_events:
            self.events.append(event)
