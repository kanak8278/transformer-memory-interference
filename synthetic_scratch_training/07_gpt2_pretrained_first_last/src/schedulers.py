"""Learning-rate schedules, all as pure step->lr functions/objects so the training
loop stays identical across arms (we just swap the schedule).

Three modes:
  cosine   -- nanoGPT baseline: linear warmup -> cosine decay to min_lr. The
              validated choice for LLM-style training; our control arm.
  sgdr     -- CosineAnnealingWarmRestarts (Loshchilov & Hutter 2017): warmup once,
              then cosine down to min_lr over period T_0, jump back to peak, repeat
              with the period multiplied by t_mult each restart. The "oscillating"
              arm -- LR periodically goes back up.
  plateau  -- reactive restart: hold peak (after warmup); each time val accuracy
              fails to improve for `patience` evals, cosine-anneal down over a
              window; whenever it's improving, ride back up toward peak. This is
              the closest match to "lr keeps going up and down IF the model is not
              improving" -- the oscillation is triggered by the plateau, not a
              fixed clock.

Research note (see setup.md): warm restarts are NOT standard for large-LLM
pretraining and can hurt there. We run them here anyway because this is a small
algorithmic / grokking-regime task where a periodic kick may help escape the
plateau, and because we compare directly against the cosine control.
"""

import math


def cosine_lr(step, *, peak_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def sgdr_lr(step, *, peak_lr, min_lr, warmup_steps, t_0, t_mult=2):
    """Cosine annealing with warm restarts. Warmup happens once, before the first
    cosine cycle; restarts thereafter go straight back to peak (standard SGDR)."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    s = step - warmup_steps
    # locate current cycle
    period = t_0
    cycle_start = 0
    while s >= cycle_start + period:
        cycle_start += period
        period *= t_mult
    t_cur = s - cycle_start
    coeff = 0.5 * (1.0 + math.cos(math.pi * t_cur / period))
    return min_lr + coeff * (peak_lr - min_lr)


class PlateauRestartLR:
    """Reactive up/down schedule driven by validation accuracy.

    State machine over eval events (call .on_eval(val_acc) at each eval) plus a
    per-step lr(step) query:
      - warmup: linear warmup to peak.
      - After warmup, maintain a scalar `level` in [min_frac, 1.0] scaling peak_lr.
      - Each eval: if val improved by > threshold, nudge level UP (ride back toward
        peak). If it failed to improve for `patience` consecutive evals, drop into a
        DECAY state that cosine-lowers `level` toward min over `decay_evals` evals,
        then the next improvement pulls it back up -> oscillation.
    Between evals lr is held at the current level (piecewise-constant per eval
    window); fine for our eval cadence.
    """

    def __init__(self, *, peak_lr, min_lr, warmup_steps, patience=2, threshold=1e-3,
                 decay_evals=3, up_factor=1.0):
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.min_frac = min_lr / peak_lr
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.threshold = threshold
        self.decay_evals = decay_evals
        self.up_factor = up_factor
        self.level = 1.0
        self.best = -1.0
        self.stall = 0
        self.decaying = 0  # >0 => within a decay ramp, counts down
        self.history = []

    def lr(self, step):
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)
        return max(self.min_lr, self.peak_lr * self.level)

    def on_eval(self, val_acc):
        improved = val_acc > self.best + self.threshold
        if improved:
            self.best = val_acc
            self.stall = 0
            self.decaying = 0
            # ride back up toward peak
            self.level = min(1.0, self.level * (1.0 + self.up_factor) if self.level < 1.0 else 1.0)
            if self.level == 0:
                self.level = self.min_frac
        else:
            self.stall += 1
            if self.decaying > 0:
                # continue current decay ramp
                self.decaying -= 1
                frac = self.decaying / max(1, self.decay_evals)
                coeff = 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
                self.level = self.min_frac + (1.0 - self.min_frac) * (1.0 - coeff)
            elif self.stall >= self.patience:
                # start a fresh decay ramp
                self.decaying = self.decay_evals
                self.stall = 0
        self.level = max(self.min_frac, min(1.0, self.level))
        self.history.append({"val_acc": val_acc, "level": self.level, "stall": self.stall,
                             "decaying": self.decaying})
        return self.lr(self.warmup_steps + 1)  # representative post-warmup lr
