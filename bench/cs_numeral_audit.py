"""Audit for `narrator.cs_numerals` — rerun before changing that table.

Three checks, because the first two are blind to what the third catches:

- ACCEPTANCE: every generated sequence must be accepted, with exactly its
  value, by `cs_numeral_acceptor` — a parser built to fail differently, which
  is why it lives here and not in the library.
- ORACLE: `num2words` as a second opinion on the values (skipped if absent).
- AMBIGUITY: no sequence may spell two different numbers.

The acceptance check is the one that earned its keep: it caught junk the other
two could not see, because a phrase nobody says is neither ambiguous nor
value-wrong. Adding a form by hand instead of regenerating is what this exists
to make loud.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs_numeral_acceptor import accept

from narrator import cs_numerals as gen

fail = 0
def check(cond, msg):
    global fail
    if not cond:
        fail += 1
        print("FAIL:", msg)

# 1. junk must be absent from the generator AND rejected by the acceptor
JUNK = [
    # gender leaks (the v1 defect)
    (("jedna","tisíc"), 1000), (("jedno","tisíc"), 1000), (("jednu","tisíc"), 1000),
    (("jedné","tisíce"), 1000), (("jednou","tisícem"), 1000),
    (("jedna","milion"), 10**6), (("jedno","milión"), 10**6),
    (("jeden","miliarda"), 10**9), (("jednoho","miliardy"), 10**9),
    (("dvě","tisíce"), 2000), (("dvě","miliony"), 2*10**6), (("dva","miliardy"), 2*10**9),
    (("dvacet","jeden","milionů"), 21*10**6), (("dvacet","jedno","tisíc"), 21000),
    (("dvacet","jednu","tisíc"), 21000), (("sto","jedno","tisíc"), 101000),
    # the five historical defects + structural exclusions
    (("nula","tisíc"), 0), (("nula","tisíc"), 1000), (("dvacet","sto"), 2000),
    (("dvacet","stech"), 2000), (("dva","jedna"), 21), (("tři","nula"), 30),
    (("deset","třicet"), 40), (("pět","dvanáct"), 512), (("dvacet","deset"), 30),
    (("stem","padesáti"), 150),
]
for seq, v in JUNK:
    check(seq not in gen.gen(v), f"generator emits junk {seq} for {v}")
    check(not accept(seq), f"acceptor accepts junk {seq} -> {accept(seq)}")

# 2. required good forms present, and accepted at the right value
GOOD = [
    (("jeden","tisíc"), 1000), (("tisíc",), 1000), (("dva","tisíce"), 2000),
    (("dvě","miliardy"), 2*10**9), (("jedna","miliarda"), 10**9),
    (("jednu","miliardu"), 10**9), (("jedné","miliardy"), 10**9),
    (("jednou","miliardou"), 10**9), (("jeden","milion"), 10**6),
    (("jednoho","milionu"), 10**6), (("jedním","milionem"), 10**6),
    (("dvacet","jeden","milion"), 21*10**6), (("dvacet","jedna","milionů"), 21*10**6),
    (("dvacet","jeden","tisíc"), 21000), (("dvacet","jedna","tisíc"), 21000),
    (("dvacet","dva","tisíce"), 22000), (("dvacet","dva","tisíc"), 22000),
    (("dvaadvacet","tisíc"), 22000), (("pětadvacet","tisíc"), 25000),
    (("dvacet","tisíc"), 20000), (("třicet","tisíc"), 30000),
    (("deseti","tisícům"), 10000), (("dvě","stě","tisíc"), 200000),
    (("dvou","tisíc","čtyřiceti","osmi"), 2048),
    (("dva","tisíce","dvacet","šest"), 2026),
    (("devatenáct","set","osmdesát","devět"), 1989),
    (("dvě","stě","padesáti"), 250),      # frozen head + declined tail
    (("se",), None),                       # placeholder guard below removes
    (("30","tisíc"), 30000), (("2","miliony"), 2*10**6), (("1","milion"), 10**6),
    (("dvacet","tisíc","pět","set"), 20500),
    (("milion","dvě","stě","tisíc"), 1200000),
    (("před","dvaceti","jedna"), None),   # placeholder
    (("dvaceti","jedna"), 21),
    (("sto","jedna"), 101), (("sto","jeden","tisíc"), 101000),
    (("ve","stě"), None),                 # placeholder
    (("stě","padesáti"), 150),            # "ve stě padesáti"
]
for seq, v in GOOD:
    if v is None:
        continue
    check(seq in gen.gen(v), f"generator misses {seq} for {v}")
    check(accept(seq) == {v} or v in accept(seq),
          f"acceptor wrong on {seq}: {accept(seq)} != {v}")

# 3. num2words oracle (space-insensitive), values to 1e9
try:
    from num2words import num2words
except ImportError:
    num2words = None
if num2words is None:
    # Counted as a failure, not a note. "ALL PASS" with a silently absent check
    # is the shape this whole issue is about: a green result that did not verify
    # what it claims to.
    check(False, "oracle SKIPPED — pip install num2words; ALL PASS requires it")
    print("oracle: SKIPPED (pip install num2words)")
else:
    random.seed(7)
    samples = list(range(1, 1000)) + [random.randrange(1000, 10**6) for _ in range(3000)] \
            + [random.randrange(10**6, 10**9) for _ in range(500)]
    bad = sum(1 for v in samples
              if num2words(v, lang="cs").replace(" ", "")
              not in {"".join(s) for s in gen.gen(v)})
    check(bad == 0, f"{bad} oracle mismatches")
    print(f"oracle: {len(samples)} samples, {bad} mismatches")

# 4. WELL-FORMEDNESS INVARIANT: every generated sequence accepted with
# exactly its value — over segments, thousands, round sets, random large
t0 = time.time()
sweep = list(range(1, 1000)) + list(range(1000, 5000)) \
      + [k * 1000 for k in range(1, 1000)] + [k * 10**6 for k in range(1, 1000)] \
      + [k * 10**9 for k in range(1, 100)] \
      + [random.randrange(1000, 10**6) for _ in range(2000)] \
      + [random.randrange(10**6, 10**12) for _ in range(500)]
n_seq = 0
bad_acc = []
for v in sweep:
    for s in gen.gen(v):
        n_seq += 1
        got = accept(s)
        if got != {v}:
            bad_acc.append((v, s, got))
check(not bad_acc, f"{len(bad_acc)} generated sequences not exactly accepted")
for v, s, got in bad_acc[:15]:
    print("  ", v, " ".join(s), "->", got)
print(f"acceptance: {n_seq} generated sequences over {len(set(sweep))} values, "
      f"{len(bad_acc)} failures, {time.time()-t0:.1f}s")

# 5. eager tier: size, build, ambiguity
vals = set(list(range(1, 2101)) + [k*1000 for k in range(3, 1000)]
           + [k*10**6 for k in range(1, 1000)]
           + [k*1000 + h*100 for k in range(1, 100) for h in range(1, 10)])
t0 = time.time()
table = {}
amb = 0
for v in vals:
    for s in gen.gen(v):
        if len(s) >= 2:
            key = " ".join(s)
            if key in table and table[key] != v:
                amb += 1
            table[key] = v
dt = time.time() - t0
mb = sum(map(sys.getsizeof, table)) / 1e6
check(amb == 0, f"{amb} ambiguous sequences in eager tier")
print(f"eager tier: {len(table):,} entries, build {dt:.2f}s, ~{mb:.0f}MB keys, ambiguous {amb}")

# 6. token inventory + diff vs repo blind set
from narrator.verify import _NUMBER_WORDS_CS  # noqa: E402

tokens = set()
for v in list(range(1, 1000)) + [k*1000 for k in (1,2,5,21,22,101)] \
       + [k*10**6 for k in (1,2,5,21)] + [k*10**9 for k in (1,2,5,21)]:
    for s in gen.gen(v):
        tokens |= {t for t in s if not t.isdigit()}
missing = sorted(t for t in tokens if t not in _NUMBER_WORDS_CS)
print(f"inventory: {len(tokens)} word tokens; missing from _NUMBER_WORDS_CS: {missing}")

print("STEM filtered everywhere:", all("stem" not in s for v in (100,150,1150) for s in gen.gen(v)))
print("ALL PASS" if fail == 0 else f"{fail} FAILURES")
sys.exit(1 if fail else 0)
