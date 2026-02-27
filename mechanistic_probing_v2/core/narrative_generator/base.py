"""Abstract base class for narrative trial generators."""

from abc import ABC, abstractmethod
import random


class NarrativeTrialGenerator(ABC):
    """Abstract base class for all domain-specific narrative generators."""

    @abstractmethod
    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str, seed: int) -> dict:
        """Generate a single trial. Returns dict matching output schema."""
        ...

    @abstractmethod
    def validate_trial(self, trial: dict) -> bool:
        """Validate all ground truth constraints. Raises AssertionError on failure."""
        ...

    @staticmethod
    def weighted_random_choice(weights: dict, rng: random.Random) -> str:
        """Pick a key from weights dict proportional to its value."""
        items = list(weights.keys())
        cumulative = []
        total = 0
        for item in items:
            total += weights[item]
            cumulative.append(total)
        if total == 0:
            return items[0]
        r = rng.random() * total
        for i, c in enumerate(cumulative):
            if r <= c:
                return items[i]
        return items[-1]

    @staticmethod
    def render_template(template: str, data: dict) -> str:
        """Fill {placeholder} in template with values from data dict.
        Missing keys are left as-is."""
        result = template
        for key, value in data.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    @staticmethod
    def interpolate_range(trajectory: dict, time: float) -> tuple:
        """Linearly interpolate between two nearest time anchors in a trajectory dict.

        trajectory: {time_int: (lo, hi), ...}
        Returns (lo, hi) tuple.
        """
        times = sorted(trajectory.keys())
        if time <= times[0]:
            return trajectory[times[0]]
        if time >= times[-1]:
            return trajectory[times[-1]]
        for i in range(len(times) - 1):
            if times[i] <= time <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                lo0, hi0 = trajectory[t0]
                lo1, hi1 = trajectory[t1]
                frac = (time - t0) / (t1 - t0)
                lo = int(lo0 + (lo1 - lo0) * frac)
                hi = int(hi0 + (hi1 - hi0) * frac)
                return (lo, hi)
        return trajectory[times[-1]]
