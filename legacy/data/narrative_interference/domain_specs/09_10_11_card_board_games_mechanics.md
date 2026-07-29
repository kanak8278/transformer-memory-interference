# Domain Specs 09-11: Card & Board Game Mechanics

Three game domains for state simulation: Poker Tournament, Chess Tournament, Hearthstone.

---

## Domain 09: Poker Tournament (No-Limit Texas Hold'em)

### Hand Rankings (Highest to Lowest)

| Rank | Hand | Description | Example |
|------|------|-------------|---------|
| 1 | Royal Flush | A-K-Q-J-10 of same suit | As Ks Qs Js 10s |
| 2 | Straight Flush | Five sequential cards of same suit | 7h 8h 9h 10h Jh |
| 3 | Four of a Kind (Quads) | Four cards of same rank | 9c 9d 9h 9s Kd |
| 4 | Full House (Boat) | Three of a kind + pair | Qs Qh Qd 8s 8c |
| 5 | Flush | Five cards of same suit, not sequential | Ah 9h 7h 4h 2h |
| 6 | Straight | Five sequential cards, mixed suits | 5c 6d 7h 8s 9c |
| 7 | Three of a Kind (Trips/Set) | Three cards of same rank | Jc Jd Jh 4s 7c |
| 8 | Two Pair | Two different pairs | As Ah 8d 8c Kh |
| 9 | One Pair | One pair | 10h 10d Ac 7s 3d |
| 10 | High Card | No made hand; highest card plays | Ah Ks 9d 7c 3h |

**Key probabilities (5-card):**
- Royal Flush: 1 in 649,740 (0.000154%)
- Straight Flush: 1 in 72,193 (0.00139%)
- Four of a Kind: 1 in 4,165 (0.024%)
- Full House: 1 in 694 (0.144%)
- Flush: 1 in 509 (0.197%)
- Straight: 1 in 255 (0.392%)
- Three of a Kind: 1 in 47 (2.13%)
- Two Pair: 1 in 21 (4.75%)
- One Pair: 1 in 2.4 (42.26%)
- High Card: 1 in 2 (50.12%)

### Position Names (10-Handed Table)

| Seat | Abbreviation | Full Name | Relative Position |
|------|-------------|-----------|-------------------|
| 1 | SB | Small Blind | Forced bet, acts first post-flop |
| 2 | BB | Big Blind | Forced bet, acts second post-flop |
| 3 | UTG | Under the Gun | First to act preflop |
| 4 | UTG+1 | Under the Gun +1 | Early position |
| 5 | UTG+2 | Under the Gun +2 | Early position |
| 6 | MP | Middle Position | Also called Lojack (LJ) |
| 7 | MP+1 | Middle Position +1 | Also called Hijack (HJ) |
| 8 | CO | Cutoff | One before the button |
| 9 | BTN | Button (Dealer) | Last to act post-flop (best position) |
| 10 | SB | (cycles) | |

**6-handed table:** SB, BB, UTG, MP (HJ), CO, BTN

### Typical Action Frequencies by Position (Preflop, Full Ring)

| Position | VPIP Range | PFR Range | Open-Raise % | Fold % |
|----------|-----------|-----------|--------------|--------|
| UTG | 10-15% | 8-12% | ~12% | ~88% |
| UTG+1 | 12-16% | 10-14% | ~14% | ~86% |
| UTG+2 | 13-18% | 11-15% | ~15% | ~85% |
| MP | 15-20% | 12-17% | ~17% | ~83% |
| HJ | 17-22% | 14-19% | ~19% | ~81% |
| CO | 22-28% | 18-24% | ~25% | ~75% |
| BTN | 28-38% | 22-30% | ~35% | ~65% |
| SB | 25-35% | 18-25% | ~30% (vs unopened) | ~70% |
| BB | 30-45% | 8-14% | N/A (defend) | ~55-65% |

**Postflop action distribution (approximate):**
- Flop: Check ~45%, Bet ~30%, Call ~15%, Raise ~8%, Fold ~2% (varies)
- Turn: Check ~50%, Bet ~25%, Call ~12%, Raise ~5%, Fold ~8%
- River: Check ~55%, Bet ~20%, Call ~10%, Raise ~3%, Fold ~12%

### Available Actions

| Action | Description |
|--------|-------------|
| Fold | Surrender hand, lose any chips already invested |
| Check | Pass action (only if no bet to call) |
| Call | Match the current bet |
| Raise | Increase the current bet (minimum raise = previous raise size) |
| All-In | Bet all remaining chips |

### Common Bet Sizing

**Preflop:**
- Standard open-raise: 2.2x-3x BB (live: 2.5x-3x; online: 2.2x-2.5x)
- 3-bet (re-raise): 3x the open raise (in position), 3.5-4x (out of position)
- 4-bet: 2.2x-2.5x the 3-bet
- Add 1 BB per limper when raising over limps

**Postflop:**
- Small bet: 25-33% of pot (common on dry boards)
- Medium bet: 50% of pot (standard continuation bet)
- Large bet: 66-75% of pot (polarized range, value/bluff)
- Pot-sized bet: 100% of pot (maximum pressure)
- Overbet: 125-200% of pot (nuts or air, polarized)

### Blind Structure (Standard 20-Level Tournament)

**Starting stacks by tournament type:**
| Tournament Type | Starting Chips | Starting BB |
|-----------------|---------------|-------------|
| Turbo/Low-buy-in | 1,500 | 30 BB |
| Standard | 5,000 | 50 BB |
| Deep Stack | 10,000 | 100 BB |
| Super Deep Stack | 25,000 | 250 BB |
| WSOP Main Event | 60,000 | 600 BB |

**Standard 20-Level Blind Schedule (5,000 starting chips, 20-min levels):**

| Level | Small Blind | Big Blind | Ante | BB in Starting Stack |
|-------|-------------|-----------|------|---------------------|
| 1 | 25 | 50 | 0 | 100 BB |
| 2 | 50 | 100 | 0 | 50 BB |
| 3 | 75 | 150 | 0 | 33 BB |
| 4 | 100 | 200 | 0 | 25 BB |
| 5 | 150 | 300 | 25 | 17 BB |
| 6 | 200 | 400 | 50 | 13 BB |
| 7 | 300 | 600 | 75 | 8 BB |
| 8 | 400 | 800 | 100 | 6 BB |
| 9 | 500 | 1,000 | 100 | 5 BB |
| 10 | 600 | 1,200 | 200 | 4 BB |
| 11 | 800 | 1,600 | 200 | 3 BB |
| 12 | 1,000 | 2,000 | 300 | 2.5 BB |
| 13 | 1,500 | 3,000 | 400 | 1.7 BB |
| 14 | 2,000 | 4,000 | 500 | 1.3 BB |
| 15 | 2,500 | 5,000 | 500 | 1 BB |
| 16 | 3,000 | 6,000 | 1,000 | < 1 BB |
| 17 | 4,000 | 8,000 | 1,000 | < 1 BB |
| 18 | 5,000 | 10,000 | 1,000 | < 1 BB |
| 19 | 7,500 | 15,000 | 2,000 | < 1 BB |
| 20 | 10,000 | 20,000 | 2,000 | < 1 BB |

**Level duration norms:**
- Turbo: 5-8 minutes per level
- Standard online: 10-15 minutes per level
- Standard live: 20-30 minutes per level
- Deep stack live: 40-60 minutes per level
- WSOP Main Event: 120 minutes per level (Day 1-2), 90 minutes (Day 3+)

**Blind escalation rule of thumb:** Blinds roughly double every 3-4 levels.

### Tournament Stages

| Stage | Typical Timing | Avg Stack (BB) | Key Strategic Adjustments |
|-------|---------------|----------------|---------------------------|
| Early | Levels 1-5 | 50-100 BB | Play tight, build reads, speculate with implied odds |
| Middle | Levels 6-10 | 20-50 BB | Widen stealing range, attack short stacks |
| Late/Bubble | Levels 10-14 | 10-25 BB | ICM pressure peaks; big stacks bully, short stacks shove/fold |
| In the Money (ITM) | Post-bubble | 10-30 BB | Relief from bubble; some players loosen, pay jumps start |
| Final Table | Last 9 players | 15-60 BB | Pay jumps significant, ICM dominant, deal-making possible |
| Heads-Up | Last 2 players | Varies | ICM irrelevant (winner-take-remaining); aggressive play |

### Pot Odds Basics

**Formula:** Pot Odds = Cost to Call / (Pot + Cost to Call)

**Common scenarios:**
| Bet Size (as % of pot) | Pot Odds | Equity Needed to Call |
|------------------------|----------|-----------------------|
| 25% pot | 5:1 | 16.7% |
| 33% pot | 4:1 | 20% |
| 50% pot | 3:1 | 25% |
| 66% pot | 2.5:1 | 28.5% |
| 75% pot | 2.3:1 | 30% |
| 100% pot (pot-sized) | 2:1 | 33% |
| 150% pot | 1.67:1 | 37.5% |
| 200% pot | 1.5:1 | 40% |

**Implied odds:** Adjust when future streets can pay off draws. Strong implied odds when:
- Deep stacks (100+ BB)
- Opponent has strong but not nutted range
- Draw is well-disguised (e.g., gutshot vs flush draw)

### ICM (Independent Chip Model)

**Core concept:** Tournament chips have diminishing marginal value. Doubling your stack does NOT double your equity because you can only lose what you have but the payout structure is top-heavy.

**ICM Formula (Malmuth-Harville):**
- P(player i finishes 1st) = chips_i / total_chips
- P(player i finishes 2nd) = Sum over all j != i of [P(j finishes 1st) * (chips_i / (total_chips - chips_j))]
- Continue recursively for 3rd, 4th, etc.
- Tournament equity = Sum of [P(finish k) * Prize(k)] for all k

**ICM consequences:**
- Chip EV != Dollar EV
- Losing chips costs more equity than winning the same amount gains
- Short stacks gain disproportionate equity from other players busting
- Big stacks should pressure medium stacks (not other big stacks)
- Bubble is the point of maximum ICM pressure

### Payout Structures

**Typical % of field paid:** 10-15% (standard), up to 20% (large fields)

**Standard payout distribution (9-player final table, $10,000 prize pool example):**

| Place | % of Prize Pool | Payout |
|-------|----------------|--------|
| 1st | 30-35% | $3,000-$3,500 |
| 2nd | 20-22% | $2,000-$2,200 |
| 3rd | 13-15% | $1,300-$1,500 |
| 4th | 9-10% | $900-$1,000 |
| 5th | 6-7% | $600-$700 |
| 6th | 5% | $500 |
| 7th | 4% | $400 |
| 8th | 3% | $300 |
| 9th | 2.5% | $250 |

**Payout structure types:**
- **Top-heavy:** 1st gets 40%+, big gaps between places (high-roller events)
- **Flat:** More evenly distributed (satellite-style, 50/50 SNGs)
- **Standard:** Compromise; roughly 30% to 1st, min-cash = 2x buy-in

### Dealing / Game Flow

1. Dealer button rotates clockwise each hand
2. SB posts half the big blind; BB posts full big blind
3. Two hole cards dealt to each player
4. Preflop betting round (UTG acts first)
5. Flop: 3 community cards dealt (1 burn card first)
6. Flop betting round (SB/first active player left of dealer acts first)
7. Turn: 1 community card dealt (1 burn card first)
8. Turn betting round
9. River: 1 community card dealt (1 burn card first)
10. River betting round
11. Showdown (if 2+ players remain): best 5-card hand from 7 cards wins

---

## Domain 10: Chess Tournament (FIDE Swiss System)

### Swiss System Pairing Rules

**Core principles (FIDE Dutch System):**
1. **No repeat opponents:** Two players may meet at most once during a tournament
2. **Score groups:** Players are grouped by current score; pairings are made within score groups
3. **Top-half vs bottom-half:** Within a score group, the top-rated half is paired against the bottom-rated half (player 1 vs player N/2+1, player 2 vs player N/2+2, etc.)
4. **Color alternation:** Players should alternate between White and Black each round
5. **Color balance:** The absolute difference between games with White and games with Black must not exceed 2 at any point
6. **Due color:** A player's "due color" is the color they have played fewer times; if equal, it alternates from the previous round
7. **Floaters:** Unpaired players from a higher score group are "floated down" to the next group and paired preferentially
8. **Byes:** In odd-numbered fields, the lowest-rated player in the lowest score group receives a half-point bye (0.5 points)
9. **No double floats:** A player should not be floated down in two consecutive rounds (soft rule, can be violated if necessary)

**Number of rounds:** Typically ceil(log2(N)) rounds for N players. E.g., 64 players = 6-7 rounds; 100 players = 7 rounds; 200+ players = 9-11 rounds.

### Elo Rating System

**Rating formula:**
- Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
- New rating: R'_A = R_A + K * (S_A - E_A)
  - Where S_A = actual score (1 for win, 0.5 for draw, 0 for loss)
  - R_A = current rating, R_B = opponent rating

**K-factor (FIDE, as of 2024):**

| Condition | K-factor |
|-----------|----------|
| New player (fewer than 30 rated games) | 40 |
| Rating below 2400 (never reached 2400) | 20 |
| Rating has reached 2400 at any point | 10 |

**Rating examples:**
- If 2000-rated player beats 2200-rated player: E = 1/(1+10^(200/400)) = 0.24; gain = 20*(1-0.24) = +15.2 points
- If 2000-rated player draws 2200-rated player: gain = 20*(0.5-0.24) = +5.2 points
- If 2000-rated player loses to 2200-rated player: gain = 20*(0-0.24) = -4.8 points

**Rating categories:**

| Rating Range | Approximate Strength |
|-------------|---------------------|
| 2700+ | Super Grandmaster / World Elite |
| 2500-2699 | Grandmaster level |
| 2400-2499 | International Master level |
| 2300-2399 | FIDE Master level |
| 2200-2299 | Candidate Master level |
| 2000-2199 | Expert / National Master |
| 1800-1999 | Class A / Strong Club |
| 1600-1799 | Class B / Club Player |
| 1400-1599 | Class C / Intermediate |
| 1200-1399 | Class D / Casual |
| 1000-1199 | Beginner |
| Below 1000 | Novice |

### Time Controls

| Format | FIDE Standard | Notation | Typical Tournament Use |
|--------|--------------|----------|----------------------|
| Classical | 90 min + 30 sec/move | 90+30 | World Championship cycle, national championships |
| Classical (alt) | 120 min/40 moves + 30 min + 30 sec/move | 40/120+30+30 | Strong round-robins |
| Rapid | 15 min + 10 sec/move | 15+10 | FIDE Grand Prix Rapid, World Rapid Championship |
| Rapid (alt) | 25 min + 10 sec/move | 25+10 | Major rapid events |
| Blitz | 3 min + 2 sec/move | 3+2 | World Blitz Championship |
| Blitz (alt) | 5 min + 0 sec | 5+0 | Online blitz, casual |
| Bullet | 1 min + 0 sec | 1+0 | Online only |
| Armageddon | White: 5 min, Black: 4 min (draw = Black wins) | Special | Tiebreaks |

**Time control classification (FIDE):**
- Classical: all moves in >= 60 minutes per player
- Rapid: all moves in > 10 min but < 60 min per player
- Blitz: all moves in <= 10 min per player

### Typical Game Lengths

| Format | Average Moves | Average Duration (both players) |
|--------|--------------|-------------------------------|
| Classical | 40-45 moves | 3-5 hours |
| Rapid (15+10) | 35-40 moves | 40-60 minutes |
| Blitz (3+2) | 30-38 moves | 6-10 minutes |
| Bullet (1+0) | 25-35 moves | 2-3 minutes |

**For reference:**
- Shortest possible game: 2 moves (Fool's Mate)
- Longest tournament game on record: 269 moves (Nikolic vs Arsovic, 1989)
- 50-move rule: draw can be claimed if 50 moves pass with no capture or pawn move
- 75-move rule: automatic draw after 75 such moves

### Tournament Scoring

| Result | White Score | Black Score | Notation |
|--------|------------|-------------|----------|
| White wins | 1 | 0 | 1-0 |
| Draw | 0.5 | 0.5 | 1/2-1/2 or 0.5-0.5 |
| Black wins | 0 | 1 | 0-1 |
| Forfeit (White wins) | 1 | 0 | +/- |
| Forfeit (Black wins) | 0 | 1 | -/+ |
| Double forfeit | 0 | 0 | -/- |

### Common Openings with ECO Codes (25+ Openings)

**A: Flank Openings & Non-1.d4 d5 / Non-1.e4**

| ECO | Opening Name | Key Moves |
|-----|-------------|-----------|
| A00 | Irregular Openings (e.g., Grob, Bird) | Various |
| A04-A09 | Reti Opening | 1. Nf3 |
| A10-A39 | English Opening | 1. c4 |
| A40-A44 | Old Benoni / Benoni Defense | 1. d4 c5 or Nf6 ... c5 |
| A45-A50 | Indian Game / Trompowsky | 1. d4 Nf6 (non-standard) |
| A57-A59 | Benko Gambit | 1. d4 Nf6 2. c4 c5 3. d5 b5 |
| A60-A79 | Benoni Defense (Modern) | 1. d4 Nf6 2. c4 c5 3. d5 e6 |
| A80-A99 | Dutch Defense | 1. d4 f5 |

**B: 1.e4 (Semi-Open Games, Non-1...e5)**

| ECO | Opening Name | Key Moves |
|-----|-------------|-----------|
| B00-B09 | Pirc Defense / Scandinavian | 1. e4 d6 / 1. e4 d5 |
| B10-B19 | Caro-Kann Defense | 1. e4 c6 |
| B20-B99 | Sicilian Defense (all variations) | 1. e4 c5 |
| B20-B29 | Sicilian (misc lines, Closed) | 1. e4 c5 (various) |
| B30-B39 | Sicilian Rossolimo / Sveshnikov | 1. e4 c5 2. Nf3 Nc6 3. Bb5 / ... e5 |
| B40-B49 | Sicilian (Paulsen, Kan, Taimanov) | 1. e4 c5 2. Nf3 e6 |
| B50-B59 | Sicilian (misc, Moscow, Canal) | 1. e4 c5 2. Nf3 d6 (various) |
| B60-B69 | Sicilian Richter-Rauzer | 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 Nc6 6. Bg5 |
| B70-B79 | Sicilian Dragon | 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 |
| B80-B89 | Sicilian Scheveningen | 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e6 |
| B90-B99 | Sicilian Najdorf | 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 |

**C: 1.e4 e5 (Open Games) and French**

| ECO | Opening Name | Key Moves |
|-----|-------------|-----------|
| C00-C19 | French Defense | 1. e4 e6 |
| C20-C29 | King's Gambit / misc 1.e4 e5 | 1. e4 e5 2. f4 (King's Gambit) |
| C30-C39 | King's Gambit | 1. e4 e5 2. f4 |
| C40-C49 | Scotch / Philidor / Four Knights | 1. e4 e5 2. Nf3 (various) |
| C42-C43 | Petroff (Russian) Defense | 1. e4 e5 2. Nf3 Nf6 |
| C44-C49 | Scotch / Four Knights / Three Knights | 1. e4 e5 2. Nf3 Nc6 3. d4 |
| C50-C59 | Italian Game / Evans Gambit / Giuoco Piano | 1. e4 e5 2. Nf3 Nc6 3. Bc4 |
| C60-C99 | Ruy Lopez (Spanish) | 1. e4 e5 2. Nf3 Nc6 3. Bb5 |
| C60-C69 | Ruy Lopez (various, Exchange, Berlin) | 3... a6 4. Ba4 (various) |
| C70-C79 | Ruy Lopez (Morphy Defense, Deferred) | 3... a6 4. Ba4 Nf6 5. O-O |
| C80-C89 | Ruy Lopez (Open, Closed) | 5... Nxe4 (Open) / 5... Be7 (Closed) |
| C90-C99 | Ruy Lopez (Closed, Breyer, Chigorin) | 5... Be7 6. Re1 b5 7. Bb3 d6 8. c3 |

**D: 1.d4 d5 (Closed Games)**

| ECO | Opening Name | Key Moves |
|-----|-------------|-----------|
| D00-D05 | Queen's Pawn Game (misc) | 1. d4 d5 (various) |
| D06-D09 | Queen's Gambit (misc) | 1. d4 d5 2. c4 |
| D10-D19 | Slav Defense | 1. d4 d5 2. c4 c6 |
| D20-D29 | Queen's Gambit Accepted (QGA) | 1. d4 d5 2. c4 dxc4 |
| D30-D69 | Queen's Gambit Declined (QGD) | 1. d4 d5 2. c4 e6 |
| D35-D36 | QGD Exchange Variation | 2... e6 3. Nc3 Nf6 4. cxd5 |
| D43-D49 | QGD Semi-Slav | 1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 |
| D50-D59 | QGD (various, Cambridge Springs, Tartakower) | Classical QGD lines |
| D60-D69 | QGD Orthodox | 2... e6 3. Nc3 Nf6 4. Bg5 Be7 |
| D70-D79 | Grunfeld Defense | 1. d4 Nf6 2. c4 g6 3. Nc3 d5 |
| D80-D99 | Grunfeld (Exchange, Russian) | 4. cxd5 Nxd5 (Exchange) |

**E: 1.d4 Nf6 (Indian Defenses)**

| ECO | Opening Name | Key Moves |
|-----|-------------|-----------|
| E00-E09 | Catalan Opening | 1. d4 Nf6 2. c4 e6 3. g3 |
| E10-E19 | Queen's Indian / Bogo-Indian | 1. d4 Nf6 2. c4 e6 3. Nf3 b6 / Bb4+ |
| E20-E59 | Nimzo-Indian Defense | 1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 |
| E60-E99 | King's Indian Defense (KID) | 1. d4 Nf6 2. c4 g6 (without ...d5) |
| E70-E79 | KID Four Pawns / Classical | Various |
| E80-E89 | KID Samisch | 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. f3 |
| E90-E99 | KID Classical / Mar del Plata | 5. Nf3 O-O 6. Be2 e5 |

### Tiebreak Systems

| System | Description | Usage |
|--------|-------------|-------|
| **Buchholz** | Sum of all opponents' scores | Most common in Swiss |
| **Median Buchholz** (Cut-1) | Buchholz minus highest and lowest opponent scores | Reduces outlier impact |
| **Sonneborn-Berger (SB)** | Sum of (score against each opponent * that opponent's final score); wins count opponent's full score, draws count half | Round-robin standard |
| **Direct Encounter** | Head-to-head result between tied players | Simple, intuitive |
| **Cumulative (Progressive)** | Sum of running score after each round (e.g., 1+1.5+2+3 = 7.5) | Rewards early wins |
| **Number of Wins** | Player with more wins breaks ahead | Rewards decisive play |
| **Average Rating of Opponents (ARO)** | Mean rating of all opponents faced | Strength-of-schedule |
| **Koya System** | Score against opponents with 50%+ | Round-robin tiebreak |

**Typical tiebreak priority in FIDE Swiss:**
1. Direct encounter
2. Buchholz Cut-1
3. Buchholz
4. Sonneborn-Berger
5. Number of wins
6. Number of games with Black

### Title Norms and Requirements

| Title | Abbreviation | Rating Requirement | Norm Requirements |
|-------|-------------|-------------------|-------------------|
| Grandmaster | GM | 2500+ | 3 norms in events with avg rating >= 2380, performance >= 2600, at least 3 GMs in field, 9+ rounds |
| International Master | IM | 2400+ | 3 norms with performance >= 2450, avg opponent >= 2230, 9+ rounds |
| FIDE Master | FM | 2300+ | No norms required; rating alone sufficient |
| Candidate Master | CM | 2200+ | No norms required; rating alone sufficient |
| Woman Grandmaster | WGM | 2300+ | 3 norms (similar structure to GM but lower thresholds) |
| Woman International Master | WIM | 2200+ | 3 norms |
| Woman FIDE Master | WFM | 2100+ | No norms required |
| Woman Candidate Master | WCM | 2000+ | No norms required |

**Norm details:**
- Minimum 9 games per norm (can be combined: e.g., 27 games across 3 events)
- At least 3 titled players in the field (for GM norms: at least 3 GMs, from at least 2 federations)
- At least 2 different federations among opponents
- Performance rating must meet threshold over the norm event
- Norms do not expire once earned

### FIDE Rating Floor

- Minimum published rating: 1000
- Players cannot drop below 1000 in published lists
- Historical minimum was 1200 (changed in 2012), then 1000

---

## Domain 11: Hearthstone

### Mana System

| Mechanic | Details |
|----------|---------|
| Starting mana | 0 (gain 1 empty crystal at start of each turn) |
| Turn 1 | 1 mana crystal |
| Turn 2 | 2 mana crystals |
| Turn N (N <= 10) | N mana crystals |
| Maximum mana | 10 crystals (no more gained after turn 10) |
| Mana refill | All crystals refill to full at start of each turn |
| Excess mana | If at 10 crystals and would gain another, draw "Excess Mana" card (draw 1 card) |
| The Coin | Going-second player receives The Coin (0-cost spell: gain 1 mana crystal this turn only) |
| Overload (X) | Shaman mechanic: locks X mana crystals next turn; e.g., Overload (2) means 2 fewer usable crystals next turn |
| Wild Growth (spell) | Gain 1 empty mana crystal permanently; if at 10, draw Excess Mana instead |
| Innervate (spell) | Gain 1 mana this turn only (Druid) |

### Card Types

| Type | Description |
|------|-------------|
| **Minion** | Creature placed on board; has Attack/Health; can attack each turn (except summon turn unless Rush/Charge) |
| **Spell** | One-time effect; discarded after use; subtypes: Arcane, Fel, Fire, Frost, Holy, Nature, Shadow |
| **Weapon** | Equips to hero; has Attack/Durability; hero can attack with it; loses 1 durability per attack |
| **Hero Card** | Replaces hero portrait, hero power, and grants armor; counts as playing a hero |
| **Location** | Placed on board; has durability; activatable effect; takes 1 turn to "cooldown" between uses; cannot be attacked or targeted |

### Card Rarity

| Rarity | Color | Deck Limit | Dust to Craft | Dust to Disenchant |
|--------|-------|-----------|---------------|-------------------|
| Common | White | 2 per deck | 40 | 5 |
| Rare | Blue | 2 per deck | 100 | 20 |
| Epic | Purple | 2 per deck | 400 | 100 |
| Legendary | Orange | 1 per deck | 1,600 | 400 |

### Deck Construction Rules

| Rule | Value |
|------|-------|
| Deck size | Exactly 30 cards |
| Maximum copies of non-Legendary | 2 |
| Maximum copies of Legendary | 1 |
| Class restriction | Deck belongs to 1 class; can include class cards + neutral cards |
| Death Knight rune restriction | DK cards may require specific rune types (Blood, Frost, Unholy); deck limited to 3 rune slots |

### Board and Hand Limits

| Limit | Value |
|-------|-------|
| Board limit | 7 minions per side (14 total) |
| Hand limit | 10 cards; any drawn beyond 10 are burned (destroyed, revealed to opponent) |
| Deck limit | 30 cards at start; 60 with certain cards (e.g., Prince Renathal sets Health to 40, deck to 40) |
| Fatigue | When drawing from empty deck: take 1 damage first time, 2 second, 3 third, etc. (cumulative +1 each draw) |

### Fatigue Damage Sequence

| Draw Attempt from Empty Deck | Damage Taken | Total Fatigue Damage So Far |
|-----------------------------|-------------|---------------------------|
| 1st | 1 | 1 |
| 2nd | 2 | 3 |
| 3rd | 3 | 6 |
| 4th | 4 | 10 |
| 5th | 5 | 15 |
| 6th | 6 | 21 |
| 7th | 7 | 28 |
| 8th | 8 | 36 |
| Nth | N | N*(N+1)/2 |

**Note:** Fatigue damage bypasses armor and goes straight to Health. Each hero starts with 30 Health.

### All 11 Classes with Hero Powers

| # | Class | Hero | Hero Power Name | Cost | Effect |
|---|-------|------|----------------|------|--------|
| 1 | Druid | Malfurion Stormrage | Shapeshift | 2 mana | +1 Attack this turn and +1 Armor |
| 2 | Hunter | Rexxar | Steady Shot | 2 mana | Deal 2 damage to the enemy hero |
| 3 | Mage | Jaina Proudmoore | Fireblast | 2 mana | Deal 1 damage to any target |
| 4 | Paladin | Uther the Lightbringer | Reinforce | 2 mana | Summon a 1/1 Silver Hand Recruit |
| 5 | Priest | Anduin Wrynn | Lesser Heal | 2 mana | Restore 2 Health to any target |
| 6 | Rogue | Valeera Sanguinar | Dagger Mastery | 2 mana | Equip a 1/2 Wicked Knife weapon |
| 7 | Shaman | Thrall | Totemic Call | 2 mana | Summon a random basic Totem (see below) |
| 8 | Warlock | Gul'dan | Life Tap | 2 mana | Draw a card; take 2 damage to your hero |
| 9 | Warrior | Garrosh Hellscream | Armor Up! | 2 mana | Gain 2 Armor |
| 10 | Demon Hunter | Illidan Stormrage | Demon Claws | 1 mana | +1 Attack this turn |
| 11 | Death Knight | The Lich King | Ghoul Charge | 2 mana | Summon a 1/1 Ghoul with Charge; it dies at end of turn |

**Shaman Basic Totems (from Totemic Call):**
- Healing Totem: 0/2, at end of your turn restore 1 Health to all friendly minions
- Searing Totem: 1/1
- Stoneclaw Totem: 0/2, Taunt
- Wrath of Air Totem: 0/2, Spell Damage +1

**Note:** Cannot summon a duplicate totem already on your board.

**Upgraded Hero Powers (Justicar Trueheart / Baku the Mooneater):**

| Class | Upgraded Hero Power | Cost | Effect |
|-------|-------------------|------|--------|
| Druid | Dire Shapeshift | 2 | +2 Attack this turn and +2 Armor |
| Hunter | Ballista Shot | 2 | Deal 3 damage to enemy hero |
| Mage | Fireblast Rank 2 | 2 | Deal 2 damage to any target |
| Paladin | The Silver Hand | 2 | Summon two 1/1 Silver Hand Recruits |
| Priest | Heal | 2 | Restore 4 Health to any target |
| Rogue | Poisoned Daggers | 2 | Equip a 2/2 Wicked Knife |
| Shaman | Totemic Slam | 2 | Summon a Totem of your choice |
| Warlock | Soul Tap | 2 | Draw a card (no damage) |
| Warrior | Tank Up! | 2 | Gain 4 Armor |
| Demon Hunter | Demon Claws (upgraded) | 1 | +2 Attack this turn |
| Death Knight | Ghoul Charge (upgraded) | 2 | Summon a 2/1 Ghoul with Charge; dies at end of turn |

### Minion Keywords (Comprehensive List)

**Evergreen / Core Keywords:**

| Keyword | Effect |
|---------|--------|
| **Taunt** | Enemies must attack this minion before other targets; does not block spells or hero powers |
| **Charge** | Can attack immediately the turn it is played (including face) |
| **Rush** | Can attack enemy minions immediately the turn it is played; cannot attack hero until next turn |
| **Divine Shield** | First instance of damage dealt to this minion is negated (shield pops) |
| **Windfury** | Can attack twice per turn |
| **Mega-Windfury** | Can attack four times per turn |
| **Stealth** | Cannot be targeted by enemy spells, hero powers, or attacks; lost when the minion attacks |
| **Poisonous** | Any damage this deals to a minion destroys it |
| **Lifesteal** | Damage dealt by this also heals your hero for the same amount |
| **Reborn** | First time this dies, return it to life with 1 Health |
| **Deathrattle** | Triggers an effect when the minion dies |
| **Battlecry** | Triggers an effect when played from hand |
| **Combo** | Triggers bonus effect if another card was played this turn first (Rogue keyword) |
| **Choose One** | Select one of two effects when played (Druid keyword) |
| **Overload (X)** | Playing this locks X mana crystals next turn (Shaman keyword) |
| **Spell Damage +X** | Your spells deal X extra damage |
| **Freeze** | Frozen character loses next attack opportunity |
| **Silence** | Remove all card text, enchantments, and keywords from a minion |
| **Immune** | Cannot be damaged or targeted (hero or minion) |
| **Elusive** | Cannot be targeted by spells or hero powers (either player) |

**Expansion-Specific Keywords:**

| Keyword | Expansion | Effect |
|---------|-----------|--------|
| **Discover** | League of Explorers | Choose one of 3 randomly offered cards; add it to your hand |
| **Inspire** | The Grand Tournament | Effect triggers each time you use your Hero Power |
| **Adapt** | Journey to Un'Goro | Choose 1 of 3 random adaptations (from pool of 10 buffs) |
| **Recruit** | Kobolds & Catacombs | Summon a minion from your deck |
| **Echo** | The Witchwood | Can be played multiple times per turn (each copy costs same mana) |
| **Magnetic** | The Boomsday Project | Play to left of a friendly Mech to merge (add stats + text); otherwise plays as normal minion |
| **Overkill** | Rastakhan's Rumble | Effect triggers if this deals excess lethal damage on your turn |
| **Twinspell** | Rise of Shadows | After casting, adds a copy of the spell to your hand (without Twinspell) |
| **Lackey** | Rise of Shadows | 1-cost 1/1 minions with powerful Battlecries (6 types) |
| **Invoke** | Descent of Dragons | Triggers an effect and upgrades Galakrond hero card |
| **Outcast** | Ashes of Outland | Bonus effect if played from leftmost or rightmost position in hand (DH keyword) |
| **Spellburst** | Scholomance Academy | Effect triggers once, the next time you cast a spell |
| **Corrupt** | Madness at the Darkmoon Faire | Upgrades in hand when you play a card that costs more |
| **Frenzy** | Forged in the Barrens | Effect triggers the first time this survives damage |
| **Tradeable** | United in Stormwind | Drag to deck: pay 1 mana, shuffle this in, draw a new card |
| **Questline** | United in Stormwind | Multi-step quest with intermediate rewards |
| **Honorable Kill** | Fractured in Alterac Valley | Effect triggers if this deals exactly lethal damage |
| **Colossal +X** | Voyage to the Sunken City | When summoned, also summon X appendage minions alongside |
| **Dredge** | Voyage to the Sunken City | Look at bottom 3 cards of deck; pick one to put on top |
| **Infuse (X)** | Murder at Castle Nathria | While in hand, upgrades after X friendly minions die |
| **Manathirst (X)** | March of the Lich King | Bonus effect if you have X total mana crystals |
| **Corpse** | March of the Lich King | Resource for Death Knight; friendly minions that die generate Corpses |
| **Forge** | TITANS | Drag to deck: pay 2 mana to upgrade the card; draw it back improved |
| **Titan** | TITANS | Has 3 unique abilities; uses one per turn instead of attacking (for 3 turns) |
| **Miniaturize** | Showdown in the Badlands | When played, adds a 1-mana 1/1 copy to your hand (keeps card text) |
| **Excavate** | Showdown in the Badlands | Progress through tiers of buried treasure (4 tiers) |
| **Quickdraw** | Showdown in the Badlands | Bonus if played on the same turn it was drawn or created |
| **Starship** | Perils in Paradise | Modular ship assembled from Starship Piece cards |
| **Tourist** | Perils in Paradise | Allows including cards from another class in your deck |
| **Rewind** | Across the Timeways | After this dies, return it to your hand |
| **Fabled** | Across the Timeways | Can only have 1 Fabled card per deck (similar to Legendary limit but cross-rarity) |

### Game Flow

| Phase | Details |
|-------|---------|
| Mulligan | Each player draws 3 cards (going first) or 4 cards (going second); may replace any/all |
| Going-second bonus | The Coin (0-cost: gain 1 mana this turn) + 1 extra card in opening hand |
| Turn structure | (1) Draw card, (2) Gain mana crystal (up to 10), (3) Refill mana, (4) Play cards / attack / use Hero Power |
| Turn timer | 75 seconds (standard); rope appears with ~15 seconds remaining |
| Win condition | Reduce enemy hero Health to 0 or below |
| Starting Health | 30 for all heroes (40 with Prince Renathal) |
| Starting Armor | 0 (some Hero cards grant armor on play) |

### Typical Game Length

| Archetype Matchup | Turns | Real Time |
|-------------------|-------|-----------|
| Aggro vs Aggro | 5-7 turns | 4-6 minutes |
| Aggro vs Control | 6-9 turns | 6-10 minutes |
| Midrange vs Midrange | 8-11 turns | 8-12 minutes |
| Control vs Control | 10-15 turns | 12-20 minutes |
| Fatigue/Value | 15-25+ turns | 20-30+ minutes |
| Average (ladder) | 8-12 turns | 7-12 minutes |

### Attack Rules

| Rule | Detail |
|------|--------|
| Newly played minions | Cannot attack (summoning sickness) unless Rush or Charge |
| Rush minions | Can attack enemy minions on play turn, not hero |
| Charge minions | Can attack anything on play turn |
| Taunt targeting | Must attack Taunt minions before non-Taunt targets |
| Stealth | Cannot be attacked or targeted; breaks on attacking |
| Immune | Cannot be damaged; can still attack |
| Frozen | Loses next attack opportunity; thaws after skipping one attack phase |
| Windfury | 2 attacks per turn |
| Hero attack | Possible with weapons, hero powers (Druid, DH), or spell buffs; hero takes damage equal to target's Attack when attacking minions |

### Class Identity Summary

| Class | Primary Archetypes | Unique Mechanics |
|-------|-------------------|-----------------|
| Druid | Ramp, Token, Combo | Choose One, mana ramp, large minions, armor gain |
| Hunter | Aggro, Beast, Secrets | Beasts, Secrets, face damage, traps |
| Mage | Spell, Tempo, Freeze | Secrets, Freeze, spell generation, burn damage |
| Paladin | Aggro, Midrange, Control | Silver Hand Recruits, buffs, Divine Shield, healing |
| Priest | Control, Combo, Resurrect | Healing, mind control, copy/steal, resurrect |
| Rogue | Tempo, Combo, Miracle | Combo keyword, weapon buffs, card generation, stealth |
| Shaman | Aggro, Control, Evolve | Overload, Totems, Evolve, Elementals |
| Warlock | Zoo, Handlock, Control | Self-damage draw (Life Tap), Demons, discard synergy |
| Warrior | Control, Aggro, Enrage | Armor, weapons, Enrage, removal, rush minions |
| Demon Hunter | Aggro, Tempo, Combo | Outcast, cheap hero power (1 mana), attack buffs, card draw |
| Death Knight | Control, Aggro, Combo | Corpses, Runes (Blood/Frost/Unholy), Ghoul tokens, Undead tribe |

---

## Cross-Domain Comparison (State Simulation Implications)

| Property | Poker Tournament | Chess Tournament | Hearthstone |
|----------|-----------------|-----------------|-------------|
| Hidden information | Hole cards (high) | None (perfect info) | Hand + deck (high) |
| Randomness source | Card shuffle, deal | None | Card draw, random effects |
| State complexity | Stack sizes, blinds, position, community cards | Board position, clock, rating | Board, hand, deck, mana, health, weapons |
| Number of players per game | 2-10 | 2 | 2 |
| Tournament structure | Single elimination (bust = out) | Swiss (all play all rounds) | Ladder (continuous) or bracket |
| Key state variables | Chip counts, blind level, position, cards, pot | Board position (64 squares, 32 pieces), clock, score | 2 heroes (health+armor), 2 boards (up to 7 each), 2 hands (up to 10 each), 2 decks, mana, weapons |
| Branching factor | ~5 actions per decision point | ~35 legal moves per position | ~10-20 actions per turn |
| Typical decisions per game | 10-30 (per hand); 200+ hands per tournament | 40-80 moves per game | 30-80 actions per game |
| Time pressure | Blind escalation (meta-clock) | Physical clock per player | Turn timer (75 sec) |
