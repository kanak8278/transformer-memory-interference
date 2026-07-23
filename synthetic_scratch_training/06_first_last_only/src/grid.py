"""(K,N) grid + held-out step-query split.

See ../setup.md "Grid" and "Held-out step-query split".
"""

import random

K_VALUES = [2, 4, 6, 8, 10, 12]
N_VALUES = [2, 4, 6, 8, 10, 12]

HELD_OUT_FRACTION = 0.25


def interior_steps(n: int):
    """Steps 2..n-1 (excludes step=1 first-value-query and step=n last-value-query)."""
    return list(range(2, n))


def all_interior_cells():
    """All (n, step) pairs across the N sweep, independent of K. 30 cells for N_VALUES above."""
    cells = []
    for n in N_VALUES:
        for s in interior_steps(n):
            cells.append((n, s))
    return cells


def select_held_out_cells(seed: int, fraction: float = HELD_OUT_FRACTION):
    """Deterministically pick ~fraction of interior (n,step) cells to hold out of training.

    Constraint: a step value s is only held out in (n,s) if s keeps at least one other
    training-time exposure — either as the last-value-query at n'=s (always-trained,
    since s is itself one of the N_VALUES for s in {2,4,6,8,10,12}), or as a
    non-held-out interior cell at some other n' != n.
    """
    cells = all_interior_cells()
    rng = random.Random(seed)
    order = cells[:]
    rng.shuffle(order)

    # exposure[s] = number of training-time sources for step token <S_s> right now.
    # baseline: 1 if s is itself a valid N (last-value-query gives free exposure), else 0.
    exposure = {}
    for n, s in cells:
        exposure.setdefault(s, 1 if s in N_VALUES else 0)
    for n, s in cells:
        exposure[s] += 1  # each interior occurrence, before any hold-out

    target = round(len(cells) * fraction)
    held_out = []
    for n, s in order:
        if len(held_out) >= target:
            break
        if exposure[s] - 1 >= 1:  # safe to remove one exposure of s and still have >=1 left
            held_out.append((n, s))
            exposure[s] -= 1

    return set(held_out)


def allowed_steps_for_training(n: int, held_out_cells):
    """Steps 1..n usable as training queries for this n (excludes held-out interior steps)."""
    return [s for s in range(1, n + 1) if (n, s) not in held_out_cells]


def all_triples(held_out_cells=None, only_held_out=False):
    """(K, N, step) triples. If held_out_cells given, filter to allowed-only or held-out-only."""
    triples = []
    for k in K_VALUES:
        for n in N_VALUES:
            for s in range(1, n + 1):
                is_held_out = held_out_cells is not None and (n, s) in held_out_cells
                if only_held_out and not is_held_out:
                    continue
                if not only_held_out and held_out_cells is not None and is_held_out:
                    continue
                triples.append((k, n, s))
    return triples
