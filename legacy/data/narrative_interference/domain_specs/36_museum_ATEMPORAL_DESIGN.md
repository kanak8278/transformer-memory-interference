# Museum Floor Observations — Atemporal Narrative Generator: System Design

> **Domain 36.** Unlike domains 1–35, this domain exists as an **experimental
> control**, not as another naturalistic setting. Its defining property is the
> *absence* of something every other narrative domain has.

---

## 1. What We're Building

A narrative domain that reads as a real document but carries **no order
information in its tokens**. Order exists only in sequence position, exactly as
in the synthetic plain key-value format.

### 1.1 The gap this fills

Every existing narrative domain carries an order signal (dates, clock times,
game minutes, Zulu times). The only format without one is synthetic plain KV.
So "naturalistic" and "carries an order cue" are perfectly confounded in the
current results, and narrative RI/PI numbers cannot be attributed to either.

|                  | no order cue      | order cue present                  |
|------------------|-------------------|------------------------------------|
| **synthetic**    | plain KV ✓        | block KV ✓                         |
| **naturalistic** | **museum M0 ← this domain** | wildlife / ICU / ATC / Dota ✓ |

### 1.2 The design criterion

Take any single item out of the stream and read it in isolation. **Does it tell
you its own rank?**

- **Block KV** — `[Update 2] / programming language: color` says "I am rank 2",
  and the query (`"...in Update 2?"`) contains the matching token. Retrieval is
  a lexical match. Shuffle the blocks: the answer is unchanged.
- **Plain KV** — `programming language: color` says nothing about rank. The
  query (`"What was the first value of...?"`) has nothing to match against.
  Shuffle the lines: the answer changes completely.

Operationalised as the **shuffle test**:

> Permute the units of the stream. Does the correct answer survive?
> Block: survives fully. Plain: destroyed fully.

| format          | order lives in                    | survives shuffling |
|-----------------|-----------------------------------|--------------------|
| block KV        | a rank token inside every item    | yes — fully lexical |
| plain KV        | sequence position only            | no — fully positional |
| **museum M0**   | sequence position only            | **no — by construction** |

Museum M0 is targeted at the bottom row. Every design decision below follows
from that one requirement.

### 1.3 Isomorphism to plain

The tracked sentences are a surface re-encoding of plain KV — entity = key,
value = value, sentence = line:

```
plain                              museum M0
─────────────────────────────      ──────────────────────────────────
programming language: smell        Rohan was recorded at Sunflowers
dinosaur genus: trail              The docent noted Priya at Nighthawks
programming language: color        Rohan stood before Guernica
```

Naturalism is added along a dimension (sentence frame, filler) that carries no
order information. This is the sense in which M0 "is plain": not similar to it,
isomorphic to it.

---

## 2. Core Design Decisions

Each entry records the decision and the reasoning, so the trade-offs are not
re-litigated later.

### 2.1 Genre: observational log, not flowing prose

**Decision.** The target register is a **floor-observation log / field report**,
not narrative prose.

**Why.** Anaphora is an order cue built into English: a pronoun cannot precede
its referent, and "the guard" cannot precede "a guard". Givenness marking
therefore encodes first-mention as reliably as `[Update 2]` does. Banning
anaphora is fatal to *prose* (it degenerates into a list) but costs nothing in
an *observational log*, a genre where repeated full identifiers are idiomatic —
real field reports genuinely write "EM544 relocated to X. EM544 captured at Y."

**Consequence.** "Naturalistic" here means *realistic document genre*, not
*flowing prose*. This must be stated plainly in the paper. The defence against a
reviewer calling M0 degenerate: it is the same register as wildlife's existing
`field` voice, with the order cues removed.

### 2.2 Tense: observational past

**Decision.** Simple past with an observational frame — "Rohan **was recorded
at** Sunflowers". No perfect aspect, no present tense.

**Why.** Two reasons, one discarded and one live.

- *Discarded:* present tense creates a literal contradiction (a visitor cannot
  be at two works at once). This turned out **not** to be a problem — a reader
  resolves it as movement, and resolves it *by reading text order as time
  order*, which is precisely the positional inference we want. Not a blocker.
- *Live:* present tense invites the model to narrativise a walk-through, and a
  walk-through has a route (see §2.6). Observational past preserves the
  positional reading without inviting route inference. Free, so take it.

Perfect aspect ("had moved to") is banned outright — it encodes anteriority
grammatically.

### 2.3 Attribute values: categorical and unordered

**Decision.** Every trackable attribute draws from an **unordered categorical
pool** in which any value may *legally* follow any other.

**Why.** Values self-order under a monotone domain constraint even with all
markers stripped. `battery: 64%` … `battery: 41%` encodes rank because battery
cannot rise; Dota gold does the same. Museum must have **no quantity with a
direction** — no counters, no elapsed anything, no percentages that drift.

Artworks are ideal: *Sunflowers* → *Guernica* → *Bathers* and *Bathers* →
*Sunflowers* → *Guernica* are equally legal, so the value sequence carries no
order information.

**Legality is not the same as emission.** The requirement above is that no
*constraint* on the pool induces an order. It does **not** require the generator
to actually emit repeats — see §2.3.1.

### 2.3.1 Within-visitor uniqueness is an invariant

**Decision.** A visitor's tracked values are drawn **without replacement**. The
same visitor is never recorded at the same artwork twice, in *either* pool mode
(§2.5). This is a generator invariant, not a config.

**Why.** Plain draws `values_per_cat[cat] = selected[idx:idx+num_updates]`, so
its values are unique within a key as well as across keys
(`dataset_configs.py:436-440`). Matching plain is this domain's entire purpose,
and the invariant buys three things:

1. **Universal check 4 (RI ≠ PI) is satisfied by construction.** No rejection
   sampling, so no acceptance rate to monitor and no risk of differential
   filtering across the sweep grid (see §10.1).
2. **Exact parity with plain** on value-repetition structure, removing a
   confound that has nothing to do with the order cue.
3. It cannot leak order. Knowing that values are distinct tells you nothing
   about which came first.

**Cost.** A visitor cannot revisit a work, which is mildly less realistic. Worth
it: plain has no repeats either, and under `shared_pool` repetition still occurs
*across* visitors, which is where the interference actually lives.

### 2.4 One record per sentence

**Decision.** Each tracked sentence carries exactly one (visitor, attribute,
value) triple. No co-occurring second attribute.

**Why.** Plain KV lines carry exactly one value; a gallery clause alongside the
artwork would add a distractor plain does not have. It also cuts the length
overhead (§2.7).

### 2.5 Value pool: disjoint primary, shared available

**Decision.** Two configs, controlling **cross-visitor sharing only**.
`disjoint_pool` is the **primary** comparison; `shared_pool` is a supported
secondary. Within-visitor uniqueness (§2.3.1) holds in both and is not a config.

| mode            | across visitors            | within a visitor        |
|-----------------|----------------------------|-------------------------|
| `disjoint_pool` | no value shared            | unique (§2.3.1)         |
| `shared_pool`   | values may recur           | unique (§2.3.1)         |

**Why.** Plain draws all values with `rng.sample(pool, n_keys * n_updates)` and
slices per category, so values are **globally unique within a trial** — unique
across keys *and* within each key. `disjoint_pool` reproduces both properties
exactly.

`shared_pool` relaxes only the cross-visitor half, which is where interference
actually comes from: the same artwork surfacing for several visitors is what
makes the queried visitor's values hard to isolate. It produces genuinely higher
interference than plain, so it is a secondary variant rather than the headline
comparison — otherwise museum would look harder for reasons unrelated to the
order cue.

### 2.5.1 `shared_pool` requires a working set

**Problem.** Sampling each visitor's deck independently from the full 1,861-title
pool produces almost no overlap — the pool is far too large relative to a trial.
Measured over 400 simulations:

| grid  | disjoint distinct values | shared distinct values |
|-------|--------------------------|------------------------|
| 3 × 20| 60                       | 59.3                   |
| 5 × 20| 100                      | **97.8**               |

A 2% difference. `shared_pool` would be indistinguishable from `disjoint_pool`
and **E4 would measure nothing**.

**Fix.** `shared_pool` first draws a per-trial **working set** of `W` titles,
then samples each visitor's deck from `W` instead of from the full pool:

```
W = clamp(ceil(working_set_multiplier * n_updates),
          lo = n_updates,                 # every visitor shares one value set
          hi = n_visitors * n_updates)    # equivalent to disjoint
```

| `working_set_multiplier` | effect |
|--------------------------|--------|
| **1.0** (default) | `W = n_updates`. Each deck needs `n_updates` distinct values from a set of exactly that size, so **every visitor is associated with the identical set of artworks** and only the ordering differs. Maximum interference. |
| intermediate (e.g. 2.0) | partial overlap; a milder condition if `m = 1` floors accuracy |
| `n_visitors` | approaches `disjoint_pool` but does **not** reach it — see below |

Distinct values in context, measured over 400 simulations:

| grid   | disjoint | `m=1.0` | `m=2.0` | `m=n_visitors` |
|--------|----------|---------|---------|----------------|
| 3 × 5  | 15       | 5.0     | 8.7     | 10.6           |
| 3 × 20 | 60       | 20.0    | 34.9    | 42.2           |
| 5 × 10 | 50       | 10.0    | 19.3    | 33.8           |
| 5 × 20 | 100      | 20.0    | 38.7    | 67.3           |

**`m = n_visitors` is not equivalent to `disjoint_pool`.** It sets
`W = n_visitors × n_updates`, but each deck is then drawn *independently* from
that set, so decks still collide — 67.3 distinct rather than 100 at 5 × 20.
Disjoint avoids collisions by slicing a **single** draw (`selected[i*nu:(i+1)*nu]`),
which is a structurally different operation. The two modes therefore remain
genuinely distinct code paths; `m` cannot be used to interpolate all the way to
disjoint.

At `m = 1` the queried visitor's values cannot be isolated by *which* artworks
appear at all — only by position — which is the purest form of the interference
the domain measures. Within-visitor uniqueness (§2.3.1) is unaffected: each
visitor still receives every working-set value exactly once, so RI ≠ PI holds by
construction because the orderings differ.

### 2.6 No route priors

**Decision, enforced as generator assertions, not conventions:**

1. Gallery names are **proper names only** — no `Atrium`, `Lobby`, `Entrance
   Hall`, `Rotunda`, `Exit Court`, and no cardinal-direction names that imply
   traversal.
2. **No entry/exit events.** Visitors are never introduced or dismissed.
3. **No adjacency** in the state machine. Any gallery may follow any gallery.

**Why.** A naturalistic domain supplies order priors the text does not — a model
knows visits have a start, a route, and an end, and may answer from an invented
plausible route rather than from position. Plain KV has no world model and so no
prior.

**Status.** The empirical prior tests (§10.3) are **deferred**. These three
constraints are therefore the *only* protection against §2.6, which is why they
are assertions.

### 2.7 Length: two configs, treated as a slope

**Decision.** Ship `filler` and `no_filler` configs. **`no_filler` is the
default** and carries the headline result; `filler` is the paired comparison. Do
not attempt exact token parity with plain.

**Why.** Per-item length (≈4 chars/token estimate, n_keys=3, n_updates=5):

| format                             | ~tokens/item | vs plain |
|------------------------------------|--------------|----------|
| plain `dinosaur genus: solve`      | 5.2          | 1.0×     |
| museum, no filler, one record/sent | 8.2          | 1.6×     |
| museum, no filler, + gallery clause| 13.2         | 2.5×     |

No-filler does not reach parity. That is acceptable and arguably preferable: the
two configs give **two points on a length axis inside the domain**. If both
yield the same PI−RI gap, length is not driving the effect and the comparison to
plain is safe; if they differ, the slope is the finding.

An optional padded-plain control (plain + untracked filler lines to museum token
count) gives a single clean headline number if wanted. Cheap — a render change
to the existing plain builder.

**Interaction with §13.** `no_filler` is the tightest config against plain, and
also the one where the listiness risk is worst — filler was the element carrying
the naturalism. Defaulting to `no_filler` therefore puts the default and the
main open risk in the same cell. This does not change the default, but it does
set the pilot's focus: **read the `no_filler` output specifically**, since that
is the config that has to survive a reviewer.

### 2.8 Rank-blind template selection

**Decision.** Templates are sampled uniformly from a single pool, independent of
mention rank. There is no first-mention template family.

**Why.** The natural writing instinct — a fuller frame on first mention, terser
after — reintroduces exactly the signal being removed.

### 2.9 Filler never names a tracked visitor

**Decision.** Filler sentences must not mention any visitor in the roster.
Enforced by the linter (§9).

**Why.** A filler mention of the queried visitor that carries no tracked value
gives the model a false anchor for "first mention".

### 2.10 Fame priors: accepted as variance

**Decision.** Real or famous-sounding titles are permitted. Not controlled.

**Why.** Title→position assignment is random, so fame bias lands on RI and PI
equally across trials. It contributes **variance, not bias** to the PI−RI gap.
The only cost is a little statistical power at 50 trials/cell. Recorded here so
the assumption is explicit rather than silent.

### 2.11 Universal Principle 2 is met, not deviated from

An earlier draft of this spec claimed museum could support only ~7 trackable
attributes and recorded a deviation from Principle 2. **That was wrong**, and it
came from conflating attributes with values.

- **Attributes** (what Principle 2 counts) are schema slots. Museum supports
  **18** (§4) — comfortably inside the 15–25 band.
- **Values** are pool entries. A museum holds thousands of objects, so the
  value pools here are among the largest of any domain (§5.1).

The atemporal requirement (§2.3) restricts attribute **shape** — no numerics, no
direction, no counters — not attribute **count**. What is genuinely true is that
museum has *no numeric attributes at all*, where other domains draw many of
their 15–25 from numeric readings. That is a difference in composition, not a
shortfall in variety.

---

## 3. Entities

**Visitors.** **2–50 per trial**, sampled from a 139-name multi-origin pool
(`museum_data.json:VISITOR_NAMES`, built by `build_pools.py`) so no visitor is
systematically easier to track.

Constraints:
- The header roster lists visitors in an order **shuffled independently** of
  narrative order, so roster position cannot correlate with first-mention rank.

**Removed constraint.** An earlier draft required that no two visitors in a
trial share a first initial. That is unsatisfiable at 50 visitors (26 letters)
and was never needed: §2.1 bans anaphora, so every reference is a full name and
there is nothing to disambiguate.

**Note on name reuse.** At `n_visitors = 50` a trial draws 50 of 139 names, so
rosters overlap substantially across seeds. This is harmless — names are never
answer values under §4.0, only artwork titles are — but expand the pool if
cross-trial name variety ever becomes relevant.

---

## 4. TRACKABLE_ATTRIBUTES

All categorical. All unordered. None numeric. Values are dealt without
replacement per visitor (§2.3.1).

| attribute             | value pool         | notes                                     |
|-----------------------|--------------------|-------------------------------------------|
| `artwork`             | `ARTWORKS` (§5.1)  | **primary** — default tracked attribute    |
| `gallery`             | `GALLERIES` (§5.2) | proper names only                          |
| `wing`                | `WINGS`            | proper names only, no cardinal directions  |
| `exhibition`          | `EXHIBITIONS`      | named shows, permanent and temporary       |
| `docent`              | `DOCENTS`          | staff first names, disjoint from visitors  |
| `companion`           | visitor roster     | who the visitor was logged alongside       |
| `audio_topic`         | `AUDIO_TOPICS`     | named topics, **never numbered tracks**    |
| `audio_language`      | `LANGUAGES`        | selected guide language                    |
| `seating`             | `SEATING`          | named benches and alcoves                  |
| `sketching_subject`   | `ARTWORKS`         | shares the artwork pool                    |
| `photograph_subject`  | `ARTWORKS`         | shares the artwork pool                    |
| `wall_text`           | `INTERPRETIVE_PANELS` | named interpretive panels               |
| `gallery_talk`        | `TALKS`            | named talks and tours                      |
| `medium_focus`        | `MEDIA`            | bronzes, textiles, works on paper, …       |
| `guidebook_section`   | `GUIDEBOOK_SECTIONS` | named sections                           |
| `conservation_display`| `CONSERVATION_STATIONS` | named open-conservation stations      |
| `study_room_request`  | `ARTWORKS`         | shares the artwork pool                    |
| `shop_item`           | `SHOP_ITEMS`       | named items examined in the shop           |

Eighteen attributes, satisfying Universal Principle 2 (15–25). Note that
`artwork`, `sketching_subject`, `photograph_subject` and `study_room_request`
deliberately **share the `ARTWORKS` pool** — that overlap is a free source of
interference under the `mixed` attribute mode.

**Banned attribute shapes** (assertion in the generator): anything numeric,
anything with a natural direction, anything cumulative, anything named with an
index. This is the *only* real restriction the atemporal requirement imposes on
the attribute space, and it is a restriction on attribute **shape**, not on
attribute **count** — museum simply has no numeric attributes where other
domains have many. `audio_topic` is explicitly *named* rather than numbered:
"track 3" would be a rank token. `wing` must not carry cardinal directions
(§2.6).

### 4.0 Only `artwork` is wired into the generator

**Decision.** `artwork` is the sole tracked attribute. The other seventeen are
documented above as the domain's attribute space (satisfying Principle 2) but
are **not implemented** in v1.

**Why.** Three simplifications fall out at once:

1. **`attribute_mode` ceases to exist.** Principle 2's `same` vs `mixed` split
   (track one attribute type across all entities for maximum interference, vs
   mixing types for variety) is meaningless with a single attribute — `same` and
   `mixed` are the same configuration. The knob is removed, not defaulted.
2. **Pool-cardinality gating (§4.1) becomes trivial** — `ARTWORKS` at 1,861 is
   the only pool that ever mattered.
3. **Cross-pool substring safety (§4.2) stops being a live risk** — it only
   materialised under `mixed`, where two attributes could both be answer values
   in one trial.

`same`-equivalent behaviour is also the *harder* condition, so nothing is lost:
every value in a trial is an artwork title, maximally confusable, which is what
the interference measurement wants.

### 4.1 Pool cardinality gates which attributes are trackable

*(Retained for the case where the other attributes are implemented later. Under
§4.0 only `ARTWORKS` is in play and this section is inert.)*

Attribute pools have wildly different natural sizes — a museum holds thousands
of works but has perhaps fifteen rooms — and the pool mode (§2.5) sets how many
unique values a trial consumes:

| mode            | unique values needed per trial |
|-----------------|--------------------------------|
| `disjoint_pool` | `n_visitors × n_updates`       |
| `shared_pool`   | `n_updates` (per visitor, drawn independently) |

Against the built pools:

| pool                    | size  | usable disjoint | usable shared |
|-------------------------|-------|-----------------|---------------|
| `ARTWORKS`              | 1861  | full grid (needs 750 at 15×50) | full grid |
| `VISITOR_NAMES`         | 139   | small grids only | up to n_updates=139 |
| `GALLERIES`             | 15    | tiny grids only | up to n_updates=15 |
| all others              | 6–12  | not usable      | up to n_updates=6–12 |

**Consequence.** `artwork` is the only attribute that supports the headline
experiment. The other seventeen are usable at small grid points, or under
`shared_pool`, or as filler-adjacent colour — they exist for variety and to
satisfy Principle 2, not to carry the primary result. This is not a defect: the
same is true of plain, whose key names are arbitrary and whose *values* carry
the experiment.

**Implementation.** Mirror the mechanism plain already has —
`get_eligible_categories(dataset_type, min_values=...)`
(`dataset_configs.py:321`) filters categories by pool size and returns them
sorted largest-first. Museum gets `get_eligible_attributes(min_values=...)`
with the same contract, and the generator raises rather than silently
degenerating when no attribute qualifies for the requested grid point.

### 4.1.1 At 50 visitors the binding constraint is tokens, not the pool

With `n_visitors` raised to 50 (§3), `ARTWORKS` at 1,861 still supports
`n_updates` up to **37** under `disjoint_pool` (1861 ÷ 50). Context length binds
first:

| grid    | records | ~tokens (no filler, 8.2/record) | status |
|---------|---------|--------------------------------|--------|
| 50 × 5  | 250     | 2,090                          | fine   |
| 50 × 10 | 500     | 4,140                          | fine   |
| 50 × 20 | 1,000   | 8,240                          | **skipped** |
| 50 × 37 | 1,850   | 15,210                         | skipped |

`narrative_experiment.py:105` skips any prompt over **8,000 tokens**, which caps
50 visitors at `n_updates ≈ 19` — well below the pool's 37.

**Action.** Raise that threshold rather than letting cells silently disappear
from the sweep. It predates the current models; Qwen2.5 supports 32k context.
A cell that is skipped rather than run produces a missing entry that looks
identical to a failed one in the results table.

**Do not merge attribute pools to gain capacity.** Merging `GALLERIES`,
`DOCENTS`, `EXHIBITIONS` etc. into `ARTWORKS` would add roughly 120 values to
1,861 (~6%) while making the values **type-heterogeneous** — a trial would mix
`Rohan was recorded at Ellery Room` with `Rohan was recorded at Ripple`. That is
Principle 2's `mixed` condition, which makes values *easier* to segregate and so
*lowers* interference, contradicting §4.0's choice of the homogeneous, harder
condition. The capacity gain is negligible and unnecessary; the cost is real.

### 4.2 Cross-pool substring safety

Because `ARTWORKS` is 1,861 **common English words** (§5.1), those words hide
inside proper nouns constantly — `Whitfield`←`field`, `Radcliffe`←`cliff`,
`Sandoval`←`oval`, `Halloran`←`hall`. Any attribute whose values are
English-like proper nouns (`VISITOR_NAMES`, `DOCENTS`) must therefore be
filtered against `ARTWORKS`, not just against itself. `build_pools.py` does this
via `cross_safe()`.

**Inert in v1.** The risk requires two attributes to be answer values within one
trial, which cannot happen under §4.0's single tracked attribute. `build_pools.py`
applies `cross_safe()` regardless, so the pools are already safe if the other
attributes are implemented later — but nothing in v1 depends on it.

Note that visitor names still appear in every sentence and in the question, they
are simply never *answers*. Substring collisions between a name and an artwork
title are therefore harmless in v1: the scorer only matches the expected answer.

---

## 5. Value Pools

### 5.1 Artworks — reuse plain's value pool

**Decision.** `ARTWORKS` is `mechanistic_probing_v2/core/data/arbitrary_single.json`
— the exact pool plain KV draws from — capitalized and filtered for substring
safety. **1,861 titles** survive the filter, against a worst-case requirement of
750 (§4.1).

**Why not a curated art-title pool.** A fresh pool would have its own token-length
and word-frequency distribution, and any museum-vs-plain difference could then be
attributed to the values rather than to the sentence frame. Reusing plain's pool
removes that confound completely: the two formats become **value-identical**, and
the only thing that differs is the frame. Given the domain exists purely to
isolate the frame, that is worth more than bespoke titles.

**Values are neutral labels, not curated titles — decided at the §13 gate.**
The pilot (§13) showed the reused pool is arbitrary English words, ~half of
which are verbs/adjectives, not nouns. In the artwork frames ("was recorded at
X", "stood before X") a non-noun value is not just unusual but *ungrammatical* —
"stood before **Devote**", "logged at **Swore**", "noted at **Mild**" parse as
corruption, not as titling.

This was raised as a decision and resolved: **keep the pool and the frames
exactly as they are.** Rationale —

1. **It does not affect the experiment.** Retrieval is identical whether the
   value is a noun or a verb; the interference measurement is untouched.
2. **Value-identity with plain is worth more than prose polish.** It makes E3
   (museum vs plain) a clean test of *sentence structure alone*, with zero value
   distribution confound — the strongest version of that comparison.
3. **It fixes two older issues for free.** Single arbitrary words are single- or
   near-single-token and globally unique, so answer scoring is exact (resolving
   the multi-token-answer hazard, old issue #2) and there is no fame prior (old
   issue #7).

**Consequence for the paper.** The values are described as **neutral one-word
labels on works in an observation log**, not as naturalistic art titles. The
naturalism claim rests on the *sentence structure and log genre* (which the gate
confirmed reads as prose, §13), not on the label strings. Do not oversell the
titles.

**Filter rule.** One rule, not two: **no title may be a case-insensitive
substring of another**. 439 of the 2,300 source words (19.1%) violate it and are
dropped. The earlier "no shared 4-character stem" rule has been **removed** — the
containment scorer only cross-matches on true substring containment, so
`Nighthawks`/`Nightfall` was never actually a hazard, and the rule cost pool size
for no benefit.

Built by `narrative_generator/museum/build_pools.py`; the invariant is asserted
at load time so a future pool edit cannot silently break the scorer.

### 5.2 Galleries

Proper names only — no route position, no cardinal directions.

```python
GALLERIES = [
    "Ellery Room", "Kessler Gallery", "Marchetti Room", "Vance Room",
    "Osgood Gallery", "Brightwater Room", "Calloway Gallery",
    "Thorne Room", "Ambrose Gallery", "Lindqvist Room",
]
```

### 5.3 Header

Atemporal. No duration, no session framing, no arc.

```
Floor observations, {museum_name}, {building}. Statements are listed in logging
order. A visitor may be logged at several works.
Visitors on record: {shuffled_roster}.
```

The sentence *"Statements are listed in logging order"* is the **single
permitted order statement** — the counterpart of plain's *"Each key gets updated
multiple times."* It licenses the RI/PI question without contributing a locating
token, and is held constant across every condition.

Note the wording avoids the pronoun *"they"* deliberately: the M0 linter (§9)
bans third-person pronouns, and the more natural phrasing "in the order they
were logged" trips it. The linter caught this during implementation — evidence
the guard works.

---

## 6. State Machine

Deliberately physics-free. This is why the generator is small: no trajectories,
no correlation engine, no seasonal curves, no impossible-transition filtering.

Values are dealt up front, exactly as plain does, then consumed in emission
order:

```
# setup — mirrors dataset_configs.py:428-440
if pool_mode == "disjoint_pool":
    # one draw, sliced per visitor: unique across AND within visitors
    selected = rng.sample(ARTWORKS, n_visitors * n_updates)
    for i, visitor in enumerate(roster):
        deck[visitor] = selected[i * n_updates : (i + 1) * n_updates]

else:  # shared_pool — §2.5.1
    W = clamp(ceil(working_set_multiplier * n_updates),
              n_updates, n_visitors * n_updates)
    working_set = rng.sample(ARTWORKS, W)
    for visitor in roster:
        deck[visitor] = rng.sample(working_set, n_updates)   # overlap across
                                                             # visitors, never
                                                             # within (§2.3.1)

# emission
for each emission slot:
    visitor = weighted_random_choice(visitor_weights, rng)
    value   = deck[visitor].pop(0)
    emit_record(visitor, "artwork", value)
```

**There is no transition constraint at all.** `rng.sample` draws without
replacement, so within-visitor uniqueness (§2.3.1) is structural rather than
enforced by a check. Universal checks 4 and 6 are therefore both satisfied by
construction — no resampling, no rejection, no retry loop. No adjacency, no
reachability, no state coherence.

The only thing the emission loop chooses is **which visitor speaks next**, which
is what creates the interleaving that RI/PI depends on.

### 6.1 Event and filler types

| type              | carries tracked value | notes                                |
|-------------------|-----------------------|--------------------------------------|
| `observation`     | yes                   | the tracked record                   |
| `room_filler`     | no                    | lighting, seating, placards, layout  |
| `crowd_filler`    | no                    | school groups, queue, general traffic |
| `facility_filler` | no                    | guard stations, doorways, cloakroom  |

Filler must be **stative and atemporal**, must not name a tracked visitor
(§2.9), and must not introduce a recurring entity whose article would alternate
a/the across mentions (§9, category 2).

Filler budgets follow Universal Principle 4 (`minimal` / `light` / `medium` /
`heavy`); the `no_filler` config pins this to `minimal`.

---

## 7. Render Modes

One state machine, three renderings. **Same seed → identical event sequence**,
so trials are **paired** and differ only by the injected marker.

| mode | rendering | analogue | status |
|------|-----------|----------|--------|
| `M0` `atemporal` | `Rohan was recorded at Ripple.` | plain KV | **built** |
| `M1` `timestamped` | `At 2:14, Rohan was recorded at Ripple.` | wildlife / ATC | built |
| `M2` `indexed` | `Rohan's third recorded stop: Ripple.` | block KV | built |
| `M3` `anaphoric` | `Rohan was recorded at Ripple. He was recorded at Blot.` | — | **optional, not v1** |

M0 is the headline control. M1/M2 come nearly free once the renderer is flagged,
and give a within-domain ablation of the order cue with entities, values,
filler, length and answer position held byte-identical apart from the marker.

**Caveat.** M1/M2 are not perfectly token-paired with M0 — markers add tokens.
Keep them short and fixed-width to minimise the gap.

### 7.0 M3 — the anaphoric mode (registered, not built)

M0's no-pronoun rule (§2.1) is a hard requirement, because anaphora is an order
cue: a pronoun cannot precede its referent. Relaxing it deliberately turns that
into a fourth condition, and it probes a **different class of cue** from M1/M2.

M1 and M2 supply an explicit *rank* (`2:14`, `third`). Anaphora supplies only a
binary **first / not-first** distinction — a pronoun says "a referent was
already introduced" and nothing about position beyond that.

That maps directly onto the RI/PI split. Given *"Rohan was recorded at Ripple.
He was recorded at Blot."*, a model looking for the first mention can simply
find the sentence carrying the full name; a model looking for the last mention
gains nothing. **Prediction: M3 selectively raises RI and leaves PI flat**,
narrowing the gap — a directional result neither M1 nor M2 can produce, since
their markers serve both conditions equally.

Cost is one additional template bank; M1/M2 already establish the render-flag
pattern. Deferred from v1 to keep scope down, not because it is unattractive.

### 7.1 M0 template bank

Rank-blind (§2.8), no pronouns, no connectives, no back-references, varied verb
frames to avoid degenerating into a table:

```
{visitor} was recorded at {value}.
{visitor} was logged at {value}.
{visitor} was observed at {value}.
{visitor} stood before {value}.
{visitor} spent the observation at {value}.
The docent noted {visitor} at {value}.
{visitor}'s attention rested on {value}.
Floor staff logged {visitor} at {value}.
{visitor} was noted at {value}.
The record for {visitor} gives {value}.
```

---

## 8. Ground Truth Schema

Unchanged from Universal Principle 5 — `config`, `narrative`, `questions`,
`entity_tracking`, `full_state_log`, `mention_counts`. `full_state_log` entries
carry `slot_index` in place of the other domains' `time` field.

RI/PI answers are the first and last entries of `entity_tracking[key]`, appended
at emission — i.e. keyed to **reading order**, matching plain.

### 8.1 Questions

Document-positional, never world-temporal:

```python
RI_QUESTION_TEMPLATES = [
    f"In this report, what was the first {attr_label} recorded for {visitor}?",
    f"What was the first {attr_label} logged for {visitor} in this report?",
    f"At the first record for {visitor} in this report, what was the {attr_label}?",
]
PI_QUESTION_TEMPLATES = [
    f"In this report, what was the last {attr_label} recorded for {visitor}?",
    f"What was the final {attr_label} logged for {visitor} in this report?",
    f"At the last record for {visitor} in this report, what was the {attr_label}?",
]
```

"in this report" is load-bearing: it anchors *first*/*last* to document position,
which is the only order signal present.

---

## 9. Order-Cue Linter

Runs on **every** generated trial. Any match is a hard failure.

| # | category | banned |
|---|----------|--------|
| 1 | explicit temporal | dates, clock times, durations, `over the`, `during the afternoon` |
| 2 | relative deictics | `now`, `then`, `next`, `later`, `after`, `before`, `subsequently`, `finally`, `again`, `still`, `already`, `returns`, `back to`, `another`, `continues`, `keeps`, `moves on` |
| 3 | aspect | perfect (`had` + participle), change-of-state progressive |
| 4 | back-reference | `previous`, `prior`, `earlier`, `last seen`, `from X to Y`, any delta or `prev_*` field |
| 5 | monotone state | any numeric attribute; any `%`; any counter |
| 6 | causality | `because`, `so that`, `prompting`, `leading to`, `which drew` |
| 7 | discourse ordinals | `first`/`finally` as discourse markers, numbered lists, section headers |
| 8 | anaphora | third-person pronouns in tracked sentences |
| 9 | article alternation | a filler entity recurring with both `a` and `the` |
| 10 | roster leak | filler naming a tracked visitor (§2.9) |

Worth pointing the same linter at the four existing domains to get a hard leak
count for the paper.

---

## 10. Validation

### 10.1 Universal checks (Principle 7)

All six apply unchanged, and checks 4 and 6 are satisfied **by construction**
rather than by validation — §2.3.1 deals each visitor's values without
replacement, so first ≠ last and no consecutive repeat are both structural. They
are still asserted, but an assertion failure indicates a generator bug, not a
trial to discard.

**No rejection sampling anywhere in this domain.** An earlier draft proposed
rejecting trials that failed check 4 and monitoring the acceptance rate against
`num_updates`. That machinery is unnecessary under §2.3.1 and is recorded here
only so it is not reintroduced: the reason it would have mattered is that
rejection rate rises with `num_updates` (a 2-mention visitor cannot collide at
all, a 30-mention visitor collides at roughly 1/pool), and rejection
preferentially removes trials where the visitor circled back to their opening
value — i.e. trials with fewer distinct competing values, which are the *easier*
ones. That would have deleted easy trials disproportionately from the
high-`num_updates` cells and shown up as an interference effect that was really
a filtering artifact. Drawing without replacement removes the failure mode at
source.

### 10.2 Domain-specific checks

7. **Order-cue linter** (§9) — zero matches.
8. **Pool assertions** — no artwork title is a substring of another; no gallery
   name implies route position; no attribute is numeric.
9. **Rank-blindness assertion** — template selection is independent of mention
   rank.
10. **Roster-order assertion** — header roster order is uncorrelated with
    first-mention order across a batch.

### 10.3 Deferred: prior and leak probes

Not run in the first pass (§2.6). Retained here as the protocol if the
constraints prove insufficient:

- **(a) No-narrative prior probe.** Roster + artwork set only, no observations,
  same question. Above chance = pure world-prior. Chance = 1/(distinct values
  logged for that visitor).
- **(b) Shuffled-narrative control.** Permute sentences, then score **twice**:
  against the *original* answer (tests leakage/priors — should be at chance) and
  against the *new positional* answer (tests position tracking — should match
  unshuffled accuracy). The single most informative check available.
- **(c) Mirrored counterbalancing.** Re-render with first and last values
  swapped in position. An accuracy difference *is* the prior's contribution,
  quantified; also catches fame bias (§2.10).
- **(d) Error clustering.** On wrong answers, test uniformity over the visitor's
  other values; concentration identifies the prior.

Run against plain KV, all four should sit at chance by construction — which
validates the protocol itself.

---

## 11. Experiment Plan

| id | comparison | isolates |
|----|------------|----------|
| **E1** | museum M0 vs museum M1 vs M2, paired by seed | the order cue, everything else identical |
| **E2** | museum M0 (`no_filler`) vs museum M0 (`filler`) | length / distance |
| **E3** | museum M0 vs plain KV | naturalism, order cue held at zero |
| **E4** | museum `disjoint_pool` vs `shared_pool` | interference level |

Models and grid follow §5.2 of `NARRATIVE_TRANSFER_RESULTS.md` — Qwen2.5-1.5B-Instruct
and Claude Haiku, 50 trials/cell over the existing (n_keys × n_updates) grid.

**Refusals are counted separately from wrong answers** (`n_garbage` vs
`n_failures`, as in `v3/arbitrary_single_table.csv`). A model may reasonably
answer "cannot be determined" when every order cue is absent; scoring that as
wrong on RI but not PI would manufacture a gap that is not interference.

### 11.1 Predictions, committed in advance

- **RI drops M2 → M1 → M0, PI flat, gap widens at M0** → the order cue is
  load-bearing for RI specifically, and quantifies how much of narrative RI
  resistance is cue-driven rather than memory-driven.
- **M0 ≈ existing narrative domains** → the cue was never used; PI > RI is
  robust to it and §5 needs no caveat.
- **M0 ≈ plain KV throughout** → strongest version: PI > RI is format-independent,
  and the ICU/ATC nulls are domain effects rather than cue effects.

---

## 12. File Layout

```
data/narrative_interference/domain_specs/36_museum_ATEMPORAL_DESIGN.md   # this file
narrative_generator/museum/
    __init__.py
    generator.py               # MuseumTrialGenerator(render_mode=...)
    data/museum_data.json      # galleries, artworks, names, docents, topics
    data/museum_templates.json # M0 / M1 / M2 banks + filler banks
```

Register `MuseumTrialGenerator` in `narrative_generator/__init__.py`; add a case
to `v3/scripts/test_narrative_generators.py`.

**Size estimate: 500–700 lines.** Wildlife is 1,964 and ATC 1,997, but the bulk
of those is domain physics — seasonal trajectories, ecological correlations,
separation constraints. Museum has none of it (§6).

---

## 13. Gate — PASSED

The §13 gate has been run. `narrative_generator/museum/pilot_render.py` renders
M0 at `no_filler` (the default and most-exposed config, §2.7); batches were read
at K=3/N=5, K=5/N=10, K=10/N=5.

**Listiness: passed.** The observation-log genre holds under generation. The
varied verb frames keep the output reading as prose, not a table, even in the
50-record wall at K=5/N=10:

> *Ines stood before Carrot. Tenzin's attention rested on Mild. Kwame was
> observed at Risen. Ines's attention rested on Model. Tenzin stood before
> Attic…*

This was the central risk in the original plan, and it did not materialise. M0
at `no_filler` is a viable structured narrative, not merely padded plain.

**Value plausibility: a real finding, resolved by decision.** The reused plain
pool (§5.1) puts non-noun words into noun slots, producing ungrammatical
readings ("stood before Devote"). Resolved by keeping the pool and frames as-is
and describing the values as **neutral labels**, not art titles (§5.1). The
naturalism claim therefore rests on structure and genre, which passed, not on
the label strings.

**Net.** The domain supports the intended claim — *PI > RI persists in a
structured narrative with no order cues* — with the values scoped as neutral
labels. The weaker "length-matched plain variant" fallback is **not** needed.

### 13.1 Residual risks (carried forward)

- **Prior tests deferred (§2.6, §10.3).** With no empirical prior probe, the
  no-route-name / no-adjacency / no-entry-exit constraints are the only
  protection. They must be generator assertions.
- **Template repetition at large N.** ~10 templates over 50 slots repeats each
  ~5×. Passed the read, but expand the M0 bank before the full sweep if the
  higher-N cells look mechanical.
- **Refusals** on M0 ordinal/interior queries (§11) must be counted separately
  from wrong answers.
