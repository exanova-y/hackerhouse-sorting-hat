#!/usr/bin/env python3
"""
SF Group House Sorting Hat

PCA and house primitive matrix live in sorting_hat_pca (computed once on import).

Methodology
───────────
1. Seven house vocabulary clusters are each scored on 15 primitive semantic axes.
2. PCA via SVD on the 7×15 matrix → k principal axes (data-driven, 85% threshold).
3. ~128-word local bank scored in same 15-D space; Datamuse expands it online.
4. Each quiz run: 10 questions of 5 words sampled via greedy farthest-first
   diversity in PC space — maximally discriminative, different every retake.
"""

import math, sys, time, random, json, urllib.request, urllib.parse

from sorting_hat_pca import DEFAULT_HOUSES, N_PRIM, _DIM_LABELS, _K, _PCA, project

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
PINK   = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def clr(text, *codes):
    return "".join(codes) + str(text) + RESET


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Local word bank  (~128 nouns, 15-D primitive scores)
#
#  nv() keyword args match PRIM_NAMES order; omitted dims default to 0.10.
# ══════════════════════════════════════════════════════════════════════════════
def nv(aes=.10, sci=.10, com=.10, pre=.10, vis=.10,
       ske=.10, bio=.10, ima=.10, cmp=.10, ncf=.10,
       phi=.10, lng=.10, apl=.10, lfs=.10, eps=.10) -> list:
    return [aes,sci,com,pre,vis,ske,bio,ima,cmp,ncf,phi,lng,apl,lfs,eps]

LOCAL_WORD_BANK: dict[str, list] = {

    # ── aesthetic / imaginative  (Vivarium cluster) ───────────────────────────
    "aether":       nv(aes=.80, ima=.75, phi=.65, ncf=.50, vis=.55),
    "aurora":       nv(aes=.85, ima=.70, phi=.50, vis=.45),
    "bloom":        nv(aes=.75, bio=.55, phi=.55, com=.50),
    "chimera":      nv(aes=.70, ima=.80, phi=.60, ncf=.55, vis=.60),
    "cosmos":       nv(aes=.70, phi=.70, vis=.75, lng=.65, ima=.60),
    "drift":        nv(aes=.60, ima=.70, ncf=.65, phi=.55),
    "ephemeral":    nv(aes=.75, phi=.65, ima=.65),
    "filament":     nv(aes=.60, sci=.40, ima=.55, phi=.50),
    "fractal":      nv(aes=.65, sci=.45, ima=.70, phi=.55, ncf=.45),
    "gossamer":     nv(aes=.80, ima=.65, phi=.55),
    "haze":         nv(aes=.65, phi=.55, ima=.60, ncf=.40),
    "iridescent":   nv(aes=.85, ima=.65, phi=.45),
    "liminal":      nv(aes=.65, phi=.70, ncf=.50, vis=.50),
    "luminous":     nv(aes=.85, ima=.65, phi=.50),
    "mirage":       nv(aes=.70, phi=.60, ima=.65, vis=.50, ncf=.45),
    "muse":         nv(aes=.70, ima=.80, phi=.60, ncf=.40),
    "numinous":     nv(aes=.75, phi=.80, ima=.70, vis=.60),
    "prism":        nv(aes=.75, ima=.65, phi=.55, vis=.45),
    "resonance":    nv(aes=.65, phi=.65, ima=.60, com=.45),
    "reverie":      nv(aes=.80, ima=.80, phi=.65, ncf=.45),
    "shimmer":      nv(aes=.80, ima=.65, phi=.45),
    "sublime":      nv(aes=.90, phi=.75, ima=.75, vis=.55),
    "threshold":    nv(aes=.55, phi=.60, com=.50, ncf=.45, vis=.50),
    "veil":         nv(aes=.65, phi=.55, ncf=.45, ima=.45),
    "wanderer":     nv(aes=.55, ncf=.65, phi=.50, ima=.55),

    # ── communal / lifestyle  (Residency + Convent cluster) ───────────────────
    "bivouac":      nv(com=.65, lfs=.55, apl=.40, bio=.40),
    "canopy":       nv(aes=.65, com=.55, bio=.50, phi=.45),
    "commune":      nv(com=.90, lfs=.50, phi=.35),
    "confluence":   nv(aes=.55, com=.65, phi=.45, vis=.40),
    "cultivate":    nv(bio=.65, com=.55, apl=.55, lfs=.50),
    "dwelling":     nv(com=.65, lfs=.60, aes=.40),
    "feast":        nv(com=.70, bio=.55, lfs=.60),
    "gather":       nv(com=.80, lfs=.55, phi=.35),
    "grove":        nv(bio=.65, com=.55, aes=.50, phi=.45),
    "harbor":       nv(com=.70, lfs=.55, aes=.40),
    "hearth":       nv(com=.80, lfs=.70, bio=.35),
    "kinship":      nv(com=.80, lfs=.55, phi=.40),
    "mosaic":       nv(aes=.65, com=.55, ima=.55),
    "nest":         nv(com=.75, lfs=.65, bio=.45),
    "ritual":       nv(lfs=.75, com=.65, bio=.45, phi=.40),
    "shelter":      nv(com=.70, lfs=.55, bio=.40),
    "somatic":      nv(bio=.75, lfs=.65, phi=.40),
    "sustain":      nv(bio=.55, com=.55, lng=.60, apl=.50),
    "weave":        nv(com=.70, aes=.55, apl=.40, lfs=.45),

    # ── scientific / applied  (Embassy + Stanford cluster) ────────────────────
    "algorithm":    nv(sci=.80, apl=.75, eps=.60, pre=.40),
    "axiom":        nv(sci=.70, eps=.70, phi=.50, pre=.40),
    "benchmark":    nv(cmp=.65, pre=.60, apl=.55, eps=.50, sci=.55),
    "blueprint":    nv(sci=.60, apl=.70, lng=.50, pre=.40),
    "calibrate":    nv(sci=.65, apl=.65, bio=.40, eps=.55),
    "cascade":      nv(sci=.55, apl=.50, vis=.45, lng=.40),
    "circuit":      nv(sci=.75, apl=.70, eps=.45),
    "compile":      nv(sci=.70, apl=.65, eps=.50, cmp=.40),
    "convergence":  nv(sci=.60, apl=.55, lng=.50, vis=.50, eps=.55),
    "debug":        nv(sci=.65, apl=.60, ske=.50, eps=.60),
    "entropy":      nv(sci=.65, phi=.50, ncf=.45, vis=.40),
    "fabricate":    nv(sci=.55, apl=.75, cmp=.50),
    "framework":    nv(sci=.60, apl=.65, pre=.45, eps=.50),
    "gradient":     nv(sci=.65, apl=.55, eps=.45, lng=.40),
    "hypothesis":   nv(sci=.70, eps=.70, ske=.55, phi=.45),
    "inference":    nv(sci=.65, eps=.75, phi=.50, ske=.55),
    "iterate":      nv(apl=.75, sci=.55, eps=.55, cmp=.45),
    "lattice":      nv(sci=.60, apl=.55, aes=.45, pre=.40),
    "optimize":     nv(cmp=.75, apl=.65, pre=.50, sci=.55),
    "pipeline":     nv(sci=.55, apl=.70, cmp=.50, pre=.40),
    "probe":        nv(sci=.70, eps=.65, ske=.55, apl=.40),
    "protocol":     nv(sci=.60, apl=.70, eps=.50, lfs=.40, bio=.40),
    "scaffold":     nv(apl=.75, sci=.55, pre=.35, eps=.45),
    "signal":       nv(sci=.55, eps=.65, apl=.50, lng=.45),
    "substrate":    nv(sci=.65, bio=.55, apl=.50, lng=.40),
    "synthesis":    nv(sci=.65, apl=.55, ima=.40, vis=.45, eps=.50),
    "theorem":      nv(sci=.70, eps=.70, phi=.55, pre=.45),
    "transmit":     nv(sci=.55, apl=.55, com=.40, eps=.45),
    "vector":       nv(sci=.70, apl=.65, eps=.55, cmp=.40),

    # ── prestige / competitive  (Stanford cluster) ────────────────────────────
    "archetype":    nv(pre=.70, phi=.55, ima=.45, lng=.40),
    "canon":        nv(pre=.65, phi=.50, lng=.45, eps=.45),
    "credential":   nv(pre=.80, cmp=.65, sci=.40),
    "hierarchy":    nv(pre=.75, cmp=.60, sci=.45),
    "legacy":       nv(pre=.65, lng=.65, phi=.50, cmp=.45),
    "mandate":      nv(pre=.70, cmp=.60, lng=.45),
    "merit":        nv(pre=.65, cmp=.70, eps=.50),
    "summit":       nv(pre=.70, cmp=.75, lng=.45, vis=.40),
    "tenure":       nv(pre=.80, cmp=.55, sci=.45),

    # ── nonconformist  (Vivarium edges + Guzey's) ─────────────────────────────
    "anomaly":      nv(ske=.65, ncf=.65, eps=.55, sci=.45),
    "contraband":   nv(ncf=.80, ske=.55, phi=.40),
    "deviant":      nv(ncf=.80, ske=.55, phi=.45),
    "glitch":       nv(ncf=.80, ima=.55, ske=.55, sci=.30),
    "hack":         nv(ncf=.70, apl=.60, sci=.45, cmp=.40),
    "heresy":       nv(ncf=.75, ske=.65, phi=.55),
    "outlier":      nv(ncf=.75, ske=.60, eps=.50),
    "quake":        nv(ncf=.75, cmp=.50, vis=.45, ske=.45),
    "rupture":      nv(ncf=.75, ske=.55, vis=.40),
    "splice":       nv(sci=.55, ncf=.60, bio=.55, apl=.50),
    "subvert":      nv(ncf=.80, ske=.60, phi=.45),

    # ── epistemic / skeptical  (Guzey's cluster) ──────────────────────────────
    "audit":        nv(eps=.70, ske=.65, sci=.50, apl=.55),
    "clarify":      nv(eps=.75, apl=.55, phi=.40),
    "contradict":   nv(ske=.70, ncf=.65, eps=.60),
    "critique":     nv(ske=.70, eps=.65, phi=.50, ncf=.55),
    "debunk":       nv(ske=.75, eps=.70, ncf=.60),
    "diagnose":     nv(sci=.60, eps=.65, ske=.55, apl=.50),
    "dissect":      nv(sci=.55, eps=.70, ske=.60, apl=.45),
    "falsify":      nv(ske=.80, eps=.70, sci=.50, phi=.40),
    "fringe":       nv(ncf=.75, ske=.55, phi=.45),
    "isolate":      nv(sci=.55, eps=.60, ske=.50, apl=.45),
    "parse":        nv(sci=.60, eps=.65, apl=.55),
    "refute":       nv(ske=.75, eps=.70, sci=.45, phi=.40),
    "scrutinize":   nv(eps=.70, ske=.65, sci=.50, apl=.50),

    # ── visionary / longtermist  (Embassy cluster) ────────────────────────────
    "covenant":     nv(lng=.75, com=.55, phi=.60, vis=.65),
    "epoch":        nv(lng=.75, phi=.60, vis=.65),
    "foundation":   nv(lng=.65, apl=.60, pre=.50, vis=.50),
    "genesis":      nv(lng=.65, vis=.70, phi=.55, ima=.50),
    "horizon":      nv(vis=.75, lng=.65, phi=.55, aes=.40),
    "launch":       nv(cmp=.75, apl=.65, lng=.55, vis=.55),
    "millennium":   nv(lng=.80, phi=.60, vis=.65),
    "monument":     nv(pre=.60, lng=.65, apl=.50, phi=.50),
    "trajectory":   nv(lng=.70, vis=.65, sci=.50, cmp=.50),

    # ── biological  (Aevitas cluster) ─────────────────────────────────────────
    "biorhythm":    nv(bio=.80, lfs=.70, sci=.45, apl=.40),
    "circadian":    nv(bio=.80, lfs=.75, sci=.50),
    "metabolize":   nv(bio=.75, sci=.55, apl=.55, lfs=.45),
    "regenerate":   nv(bio=.70, lfs=.55, lng=.55, vis=.45),
    "sequencer":    nv(sci=.65, bio=.60, apl=.55, lng=.45),
    "telomere":     nv(bio=.80, sci=.65, lng=.70, apl=.45),

    # ── present / residual ────────────────────────────────────────────────────
    "artifact":     nv(phi=.55, ncf=.45, ske=.50, aes=.45),
    "echo":         nv(aes=.55, phi=.60, ima=.50, ncf=.40),
    "imprint":      nv(lfs=.55, phi=.50, bio=.40),
    "residue":      nv(phi=.50, ncf=.40, ske=.50, aes=.35),
    "sediment":     nv(phi=.50, aes=.40, ncf=.35),
    "trace":        nv(phi=.50, eps=.45, aes=.40, sci=.35),
    "vestige":      nv(phi=.55, aes=.45, ncf=.40),
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Datamuse online expansion
#
#  For each primitive axis, two seed terms are queried via rel_trg.
#  Returned nouns inherit a sparse vector centred on that primitive (0.65)
#  with 0.10 baseline on all others — approximate but directionally correct.
# ══════════════════════════════════════════════════════════════════════════════
DATAMUSE_SEEDS: dict[int, list[str]] = {
    0:  ["beauty",       "sublime"],      # aesthetic
    1:  ["molecule",     "theorem"],      # scientific
    2:  ["community",    "gathering"],    # communal
    3:  ["prestige",     "elite"],        # prestige
    4:  ["future",       "horizon"],      # visionary
    5:  ["doubt",        "skepticism"],   # skeptical
    6:  ["organism",     "longevity"],    # biological
    7:  ["imagination",  "dream"],        # imaginative
    8:  ["ambition",     "winning"],      # competitive
    9:  ["rebel",        "outsider"],     # nonconform
    10: ["meaning",      "wonder"],       # philosophic
    11: ["legacy",       "civilization"], # longtermist
    12: ["engineer",     "craft"],        # applied
    13: ["ritual",       "habit"],        # lifestyle
    14: ["evidence",     "clarity"],      # epistemic
}

def fetch_extended_bank(local: dict, seeds: dict,
                        timeout: float = 3.0) -> tuple[dict, int]:
    """
    Query Datamuse words?rel_trg=SEED for each primitive axis.
    Accepts only single-token nouns not already in the local bank.
    Returns (extended_bank, n_new_words).
    """
    bank = dict(local)
    n_new = 0
    for prim_idx, seed_terms in seeds.items():
        for seed in seed_terms:
            try:
                qs  = urllib.parse.urlencode({"rel_trg": seed, "max": 30, "md": "p"})
                url = f"https://api.datamuse.com/words?{qs}"
                req = urllib.request.Request(url, headers={"User-Agent":"SF-SortingHat/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read().decode())
                for entry in data:
                    w    = entry.get("word","").lower().strip()
                    tags = entry.get("tags", [])
                    if (w and " " not in w and len(w) > 3
                            and "n" in tags and w not in bank):
                        vec          = [0.10] * N_PRIM
                        vec[prim_idx] = 0.65
                        bank[w]      = vec
                        n_new       += 1
            except Exception:
                pass
    return bank, n_new


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Adaptive question generation
#
#  Q1: pure farthest-first diversity (no signal yet).
#  Q2+: seed the question from the word most aligned to the *discriminating
#       direction* — the unit vector separating the current top-2 houses in
#       PC space.  Farthest-first then fans out for within-question diversity.
#
#  This means later questions home in on exactly the axis that distinguishes
#  the user's leading candidates, rather than re-exploring already-ruled-out
#  regions of personality space.
# ══════════════════════════════════════════════════════════════════════════════
def _euclid(a: list, b: list) -> float:
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def _farthest_first(words: list, vecs: list, seed_idx: int, k: int) -> list:
    """Greedy farthest-first starting from seed_idx. Returns list of indices."""
    sel = [seed_idx]
    while len(sel) < k and len(sel) < len(words):
        best_i, best_d = -1, -1.0
        for i in range(len(words)):
            if i in sel:
                continue
            d = min(_euclid(vecs[i], vecs[s]) for s in sel)
            if d > best_d:
                best_d, best_i = d, i
        sel.append(best_i)
    return sel

def discriminating_direction(score_vec: list, houses: list):
    """
    Rank houses by cosine similarity to the current score vector.
    Return the normalised difference vector  (profile_#1 − profile_#2),
    which points from the runner-up toward the leader in PC space —
    i.e. the axis that currently matters most for the final result.
    Also returns the two competing house dicts for display.
    """
    if all(x == 0.0 for x in score_vec):
        return None, None, None                     # no signal yet

    ranked = sorted(houses,
                    key=lambda h: cosine_similarity(score_vec, h["profile"]),
                    reverse=True)
    h1, h2  = ranked[0], ranked[1]
    diff    = [a - b for a, b in zip(h1["profile"], h2["profile"])]
    mag     = math.sqrt(sum(x**2 for x in diff))
    if mag == 0:
        return None, None, None

    return [x / mag for x in diff], h1, h2

def adaptive_sample(score_vec: list, pool: dict,
                    houses: list, k: int = 5) -> tuple[list, str]:
    """
    Build one question of k words.

    • If score_vec is zero (Q1): random seed → farthest-first.
    • Otherwise: find the word whose PC vector most strongly aligns with the
      discriminating direction between the top-2 houses, use it as the seed,
      then fan out via farthest-first for within-question diversity.

    Returns ([(word, pc_vec), …], hint_string).
    """
    words = list(pool)
    vecs  = [pool[w] for w in words]

    disc_dir, h1, h2 = discriminating_direction(score_vec, houses)

    if disc_dir is None:
        seed_idx = random.randrange(len(words))
        hint     = ""
    else:
        # word with highest |dot product| with the discriminating direction
        dots     = [abs(sum(v[i]*disc_dir[i] for i in range(len(disc_dir))))
                    for v in vecs]
        seed_idx = max(range(len(words)), key=lambda i: dots[i])
        hint     = f"{h1['emoji']} {h1['name']}  vs  {h2['emoji']} {h2['name']}"

    sel = _farthest_first(words, vecs, seed_idx, k)
    return [(words[i], vecs[i]) for i in sel], hint


# ══════════════════════════════════════════════════════════════════════════════
#  Maths helpers
# ══════════════════════════════════════════════════════════════════════════════
def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return 0.0 if (mag_a == 0 or mag_b == 0) else dot/(mag_a*mag_b)

def vec_add(t: list, v: list) -> list:
    return [a+b for a,b in zip(t,v)]


# ══════════════════════════════════════════════════════════════════════════════
#  Flowchart steps
# ══════════════════════════════════════════════════════════════════════════════
def fetch_resources() -> tuple:
    # print(clr("\n  Fetching house data from hackermap.org…", YELLOW))
    # try:
    #     req = urllib.request.Request("https://hackermap.org",
    #                                  headers={"User-Agent":"SF-SortingHat/1.0"})
    #     with urllib.request.urlopen(req, timeout=4) as r:
    #         r.read()
    #     print(clr("  Connected — using live community profiles.", GREEN))
    # except Exception:
    #     print(clr("  (Offline — using built-in house profiles.)", PINK))

    print(clr("  Expanding word bank via Datamuse…", YELLOW))
    bank, n = fetch_extended_bank(LOCAL_WORD_BANK, DATAMUSE_SEEDS)
    if n > 0:
        print(clr(f"  +{n} words fetched  ({len(bank)} total in bank).", GREEN))
    else:
        print(clr(f"  (Offline — {len(LOCAL_WORD_BANK)} local words in bank.)", PINK))

    time.sleep(0.3)
    return DEFAULT_HOUSES, bank


def print_pca_report():
    k, var = _PCA["k"], _PCA["var"]
    print(clr(f"\n  ── {k} personality axes derived by PCA "
              f"({_PCA['cumvar']:.0%} variance explained) ──", DIM))
    for i, (lbl, v) in enumerate(zip(_DIM_LABELS, var)):
        print(clr(f"  PC{i+1}  {lbl:<32}  {'█'*round(v*36)}  {v:.1%}", DIM))
    print()


def welcome_screen():
    print()
    print(clr("  ╔═══════════════════════════════════════╗", BOLD+YELLOW))
    print(clr("  ║   🎩  SF  S O R T I N G   H A T  🎩   ║", BOLD+YELLOW))
    print(clr("  ╚═══════════════════════════════════════╝", BOLD+YELLOW))
    print_pca_report()
    print("  Answer 10 questions to find your ideal SF group house.")
    print("  Pick your favourite word each round.")
    print()
    input(clr("  ▶  Press any key to start ", CYAN+BOLD))


def display_question(n: int, nouns: list, hint: str = "") -> tuple:
    print()
    print(clr(f"  ── Question {n} / 10 ─────────────────────────", BOLD))
    print(clr('  "Pick the word that feels most ', BOLD) +
          clr("you", BOLD+PINK) + clr('"', BOLD))
    print()
    for i, (word, _) in enumerate(nouns, 1):
        print(f"    {clr(i, YELLOW+BOLD)}.  {word.capitalize()}")
    print()
    while True:
        raw = input(clr("  Your choice (1-5): ", CYAN)).strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            idx = int(raw)-1
            print(clr(f"  ✔  {nouns[idx][0].capitalize()}", GREEN))
            return nouns[idx]
        print(clr("  Please enter a number between 1 and 5.", RED))


def compute_vibe_match(score_vec: list, houses: list) -> list:
    sims = [(h, cosine_similarity(score_vec, h["profile"])) for h in houses]
    lo   = min(s for _,s in sims)
    hi   = max(s for _,s in sims)
    span = hi - lo if hi > lo else 1.0
    results = [{**h, "pct": round(55 + ((s-lo)/span)*40), "sim": s}
               for h,s in sims]
    return sorted(results, key=lambda x: x["sim"], reverse=True)


def progress_bar(pct: int, width: int = 26, color: str = GREEN) -> str:
    f = round(pct/100*width)
    return color + "█"*f + RESET + "░"*(width-f)


def results_screen(ranked: list):
    print()
    print(clr("  ╔═══════════════════════════════════════╗", BOLD+CYAN))
    print(clr("  ║      🏆  Results — Houses by Match %  ║", BOLD+CYAN))
    print(clr("  ╚═══════════════════════════════════════╝", BOLD+CYAN))
    print()
    for i, h in enumerate(ranked):
        medal = ["🥇","🥈","🥉"][i] if i < 3 else "  "
        print(f"  {medal}  {clr(h['emoji']+' '+h['name'], BOLD):<22}  "
              f"{progress_bar(h['pct'], color=h['color'])}  {clr(str(h['pct'])+'%', BOLD)}")
        print(f"       {h['desc']}")
        print()
    top = ranked[0]
    print(clr(f"  You belong in {top['emoji']} {top['name']}!", BOLD+top["color"]))
    print()



# ══════════════════════════════════════════════════════════════════════════════
#  Main loop  (mirrors flowchart)
# ══════════════════════════════════════════════════════════════════════════════
def run():
    houses, bank = fetch_resources()
    welcome_screen()

    # Project bank into PC space once per session
    proj = {w: project(v, _PCA) for w, v in bank.items()}

    while True:
        # Fresh state for each quiz run
        used         = set()
        n            = 1
        score_vector = [0.0] * _K

        while True:
            # Build available pool (no repeats within a run)
            pool = {w: v for w, v in proj.items() if w not in used}
            if len(pool) < 5:
                pool = dict(proj)
                used.clear()

            # Generate this question adaptively from the current score signal
            nouns, hint = adaptive_sample(score_vector, pool, houses)
            used.update(w for w, _ in nouns)
            random.shuffle(nouns)

            word, vec = display_question(n, nouns, hint)
            score_vector = vec_add(score_vector, vec)
            if n < 10:
                n += 1
            else:
                break

        print(clr("\n  Computing your vibe match…", PINK))
        time.sleep(0.8)
        ranked = compute_vibe_match(score_vector, houses)
        results_screen(ranked)

        print(clr("  What would you like to do?", BOLD))
        print("    1.  Retake quiz\n    2.  Exit")
        print()
        action = input(clr("  Your choice (1-2): ", CYAN)).strip()

        if action == "2":
            print(clr("\n  Resetting — back to Question 1!\n", GREEN))
            time.sleep(0.4)
        else:
            print(clr("\n  🎉  END · Thanks for sorting!\n", BOLD+YELLOW))
            break


if __name__ == "__main__":
    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print(clr("\n\n  Goodbye! 🎩\n", YELLOW))
        sys.exit(0)
