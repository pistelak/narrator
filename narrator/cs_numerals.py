"""Czech cardinals, GENERATED: a value to every way of spelling it.

The verifier compares a script against a transcript, and Czech spells numbers
out while every recogniser writes digits back. Comparing them needs the two to
meet somewhere, and five previous attempts met them by PARSING a run of numeral
tokens and computing a value. Every one shipped a wrong integer, each passing
its own suite (issue #23 records all five): an explicit zero read as an absent
multiplier, an English compound composed under `cs`, "dvacet sto" fabricated as
2000, the oblique hundreds inheriting the thousands' coefficient range, and the
acronym STEM certified as the number 100.

This inverts it. Nothing parses: a value is expanded into the complete set of
sequences that spell it, and a run of tokens is looked up. A sequence Czech
cannot produce is not in the map, so a fabricated value is not expressible —
the failure mode becomes a MISS, which lands on the suppression that was there
before, rather than a confident wrong number. Under-generation costs a false
suppression, which is the status quo and visible
over-generation is only
dangerous where a string means something else, which is a finite set that was
enumerated against a Czech wordlist before this table existed.

Only sequences of TWO OR MORE tokens live here. Single tokens keep their
existing treatment in `verify._NUMERAL_VALUES`, and that one restriction does a
surprising amount of work: it dissolves the bare-plural ambiguity that forced
"sta", "tisíce" and "miliony" to stay unvalued, because "dva tisíce" is exactly
2000 even though "tisíce" alone means "thousands of". It also keeps `stem` out
without a special case — bare-100 instrumental is the only place it occurs, and
150-style compounds freeze to "sto" ("se sto padesáti").

Coefficient agreement is selected by the GENDER and CLASS of the scale unit.
An earlier version crossed every gender form of 1 and 2 with every unit —
"jedna tisíc", "dvě tisíce", "dva miliardy", "dvacet jeden milionů" — junk no
collision or ambiguity check could see, because junk is neither ambiguous nor
value-wrong, merely never said. It was caught by an independent acceptor that
tags each token morphologically and unifies features, deliberately built to
fail differently from this generator
it lives in `bench/`, not here, because
shipping a parser is how this becomes attempt six.
"""
from itertools import product

CASES = ("nom", "gen", "dat", "loc", "ins")

UNITS = {
    1: {"nom": ["jeden", "jedna", "jedno", "jednu"],  # m/f/n + fem acc
        "gen": ["jednoho", "jedné"], "dat": ["jednomu", "jedné"],
        "loc": ["jednom", "jedné"], "ins": ["jedním", "jednou"]},
    2: {"nom": ["dva", "dvě"], "gen": ["dvou"], "dat": ["dvěma"],
        "loc": ["dvou"], "ins": ["dvěma"]},
    3: {"nom": ["tři"], "gen": ["tří", "třech"], "dat": ["třem"],
        "loc": ["třech"], "ins": ["třemi"]},
    4: {"nom": ["čtyři"], "gen": ["čtyř", "čtyřech"], "dat": ["čtyřem"],
        "loc": ["čtyřech"], "ins": ["čtyřmi"]},
}
for v, w, o in [(5,"pět","pěti"),(6,"šest","šesti"),(7,"sedm","sedmi"),
                (8,"osm","osmi"),(9,"devět","devíti")]:
    UNITS[v] = {"nom":[w],"gen":[o],"dat":[o],"loc":[o],"ins":[o]}

TEENS = {}
for v, w, o in [(10,"deset","deseti"),(11,"jedenáct","jedenácti"),
                (12,"dvanáct","dvanácti"),(13,"třináct","třinácti"),
                (14,"čtrnáct","čtrnácti"),(15,"patnáct","patnácti"),
                (16,"šestnáct","šestnácti"),(17,"sedmnáct","sedmnácti"),
                (18,"osmnáct","osmnácti"),(19,"devatenáct","devatenácti")]:
    obl = [o] + (["desíti"] if v == 10 else [])
    TEENS[v] = {"nom":[w],"gen":obl,"dat":obl,"loc":obl,"ins":obl}

TENS = {}
for v, w, o in [(20,"dvacet","dvaceti"),(30,"třicet","třiceti"),
                (40,"čtyřicet","čtyřiceti"),(50,"padesát","padesáti"),
                (60,"šedesát","šedesáti"),(70,"sedmdesát","sedmdesáti"),
                (80,"osmdesát","osmdesáti"),(90,"devadesát","devadesáti")]:
    TENS[v] = {"nom":[w],"gen":[o],"dat":[o],"loc":[o],"ins":[o]}

FUSED_PREFIX = {1:["jedena","jedna"],2:["dvaa"],3:["třia"],4:["čtyřia"],
                5:["pěta"],6:["šesta"],7:["sedma"],8:["osma"],9:["devěta"]}

def tail(v, case):
    """1..99 -> set of tuples in `case` (nom includes all genders; the
    counted noun outside the run decides gender for a FINAL tail, so all
    are generated here — the scale-coefficient position filters later)."""
    out = set()
    if v in UNITS:
        for f in UNITS[v][case]:
            out.add((f,))
        if case != "nom" and v == 1:
            out.add(("jedna",))          # frozen: "sto jedna", "tisíc jedna"
    elif v in TEENS:
        for f in TEENS[v][case]:
            out.add((f,))
    elif v in TENS:
        for f in TENS[v][case]:
            out.add((f,))
    else:
        t, u = v - v % 10, v % 10
        for tf, uf in product(TENS[t][case], UNITS[u][case]):
            out.add((tf, uf))            # dvacet pět / dvaceti pěti
        if u == 1 and case != "nom":
            for tf in TENS[t][case]:
                out.add((tf, "jedna"))   # frozen: "před dvaceti jedna lety"
        for p, tf in product(FUSED_PREFIX[u], TENS[t][case]):
            out.add((p + tf,))           # pětadvacet / pětadvaceti
    return out

STO = {
    1: {"nom":[("sto",)],"gen":[("sta",)],"dat":[("stu",)],
        "loc":[("stě",),("stu",)],"ins":[("stem",)]},   # 'stem' filtered in gen()
    2: {"nom":[("dvě","stě")],"gen":[("dvou","set")],"dat":[("dvěma","stům")],
        "loc":[("dvou","stech")],"ins":[("dvěma","sty")]},
    3: {"nom":[("tři","sta")],"gen":[("tří","set"),("třech","set")],
        "dat":[("třem","stům")],"loc":[("třech","stech")],"ins":[("třemi","sty")]},
    4: {"nom":[("čtyři","sta")],"gen":[("čtyř","set"),("čtyřech","set")],
        "dat":[("čtyřem","stům")],"loc":[("čtyřech","stech")],"ins":[("čtyřmi","sty")]},
}
for h in range(5, 10):
    n = UNITS[h]["nom"][0]
    o = UNITS[h]["gen"][0]
    STO[h] = {"nom":[(n,"set")],"gen":[(o,"set")],"dat":[(o,"stům")],
              "loc":[(o,"stech")],"ins":[(o,"sty")]}

def hundreds_part(h, case):
    out = set(STO[h][case])
    out |= set(STO[h]["nom"])            # frozen head: "se sto padesáti"
    return out

def segment(v, case):
    """1..999 -> spellings in `case`."""
    out = set()
    h, t = divmod(v, 100)
    if v <= 99:
        return tail(v, case)
    parts_h = hundreds_part(h, case)
    if t == 0:
        out |= parts_h
    else:
        for hp, tp in product(parts_h, tail(t, case)):
            out.add(hp + tp)
    return out

# Scale units. GENDER decides which forms of 1 and 2 may modify them:
# tisíc and milion/milión are masculine inanimate, miliarda is feminine.
TISIC = {
    "one":  {"nom":["tisíc"],"gen":["tisíce"],"dat":["tisíci"],"loc":["tisíci"],"ins":["tisícem"]},
    "few":  {"nom":["tisíce"],"gen":["tisíc","tisíců"],"dat":["tisícům"],"loc":["tisících"],"ins":["tisíci"]},
    "many": {"nom":["tisíc"],"gen":["tisíc","tisíců"],"dat":["tisícům"],"loc":["tisících"],"ins":["tisíci"]},
}
MILION = {
    "one":  {"nom":["milion","milión"],"gen":["milionu","miliónu"],"dat":["milionu","miliónu"],
             "loc":["milionu","miliónu"],"ins":["milionem","miliónem"]},
    "few":  {"nom":["miliony","milióny"],"gen":["milionů","miliónů"],"dat":["milionům","miliónům"],
             "loc":["milionech","miliónech"],"ins":["miliony","milióny"]},
    "many": {"nom":["milionů","miliónů"],"gen":["milionů","miliónů"],"dat":["milionům","miliónům"],
             "loc":["milionech","miliónech"],"ins":["miliony","milióny"]},
}
MILIARDA = {
    "one":  {"nom":["miliarda","miliardu"],"gen":["miliardy"],"dat":["miliardě"],"loc":["miliardě"],"ins":["miliardou"]},
    "few":  {"nom":["miliardy"],"gen":["miliard"],"dat":["miliardám"],"loc":["miliardách"],"ins":["miliardami"]},
    "many": {"nom":["miliard"],"gen":["miliard"],"dat":["miliardám"],"loc":["miliardách"],"ins":["miliardami"]},
}

# Declined "1" by gender and case. Feminine nominative/accusative is PAIRED
# with its noun form (jedna miliarda / jednu miliardu), never crossed.
_ONE_M = {"nom":"jeden","gen":"jednoho","dat":"jednomu","loc":"jednom","ins":"jedním"}
_ONE_F_OBL = {"gen":"jedné","dat":"jedné","loc":"jedné","ins":"jednou"}
_ONE_FORMS = {"jeden","jedna","jedno","jednu","jednoho","jednomu","jednom",
              "jedním","jedné","jednou"}
_TWO_NOM = {"m":"dva","f":"dvě"}

def coeff_class(k):
    if k == 1:
        return "one"
    last2, last1 = k % 100, k % 10
    if last1 in (2, 3, 4) and last2 not in (12, 13, 14):
        return "few"
    return "many"

def _one_group(table, gender, case):
    """Spellings of coefficient-1 x unit: bare, declined-1, digit-1."""
    out = set()
    for u in table["one"][case]:
        out.add((u,))
        out.add(("1", u))
    if gender == "m":
        for u in table["one"][case]:
            out.add((_ONE_M[case], u))
    elif case == "nom":
        for cf, u in zip(("jedna", "jednu"), table["one"]["nom"], strict=False):
            out.add((cf, u))             # jedna miliarda / jednu miliardu
    else:
        for u in table["one"][case]:
            out.add((_ONE_F_OBL[case], u))
    return out

def scale_group(k, table, gender, case):
    """Coefficient k (1..999) + scale word, agreement enforced."""
    if k == 1:
        return _one_group(table, gender, case)
    out = set()
    cls = coeff_class(k)
    for coeff in segment(k, case) | {(str(k),)}:
        final = coeff[-1]
        if final.isdigit():
            classes = {cls} | ({"many"} if cls == "few" and k > 4 else set())
            for cl in classes:
                for u in table[cl][case]:
                    out.add((*coeff, u))
        elif final in _ONE_FORMS:
            # ...21/...31/...101: noun agrees with the declined "jeden"
            # (dvacet jeden milion), or frozen "jedna" takes the counted
            # plural (dvacet jedna milionů). Wrong-gender declined forms
            # are dropped, which is the whole point of v2.
            if final == "jedna":
                for u in table["many"][case]:
                    out.add((*coeff, u))
            if gender == "m":
                if final == _ONE_M[case]:
                    for u in table["one"][case]:
                        out.add((*coeff, u))
            elif case == "nom":
                for cf, u in zip(("jedna", "jednu"), table["one"]["nom"], strict=False):
                    if final == cf:
                        out.add((*coeff, u))
            elif final == _ONE_F_OBL[case]:
                for u in table["one"][case]:
                    out.add((*coeff, u))
        elif final in ("dva", "dvě"):
            if final != _TWO_NOM[gender]:
                continue                 # "dvě tisíce" / "dva miliardy": junk
            for u in table["few"][case]:
                out.add((*coeff, u))
            if k > 4:
                for u in table["many"][case]:
                    out.add((*coeff, u))
        else:
            classes = {cls} | ({"many"} if cls == "few" and k > 4 else set())
            for cl in classes:
                for u in table[cl][case]:
                    out.add((*coeff, u))
    return out

def gen(v, cases=CASES, year_style=True):
    """Complete spelling set for v (1..999_999_999_999)."""
    out = set()
    g, rest = divmod(v, 10**9)
    m, rest2 = divmod(rest, 10**6)
    k, r = divmod(rest2, 1000)
    for case in cases:
        heads = [()]
        if g:
            heads = [h + s for h in heads for s in scale_group(g, MILIARDA, "f", case)]
        if m:
            heads = [h + s for h in heads for s in scale_group(m, MILION, "m", case)]
        if k:
            heads = [h + s for h in heads for s in scale_group(k, TISIC, "m", case)]
        tails = [()] if r == 0 else list(segment(r, case))
        for h, t in product(heads, tails):
            if h + t:
                out.add(h + t)
    if year_style and 1100 <= v <= 1999:
        h, r = divmod(v, 100)
        for teen in TEENS[h]["nom"]:
            if r == 0:
                out.add((teen, "set"))
            else:
                for t in tail(r, "nom"):
                    out.add((teen, "set", *t))
    return {s for s in out if "stem" not in s}


# The eagerly-built tier. Every isolated segment (so every coefficient 1-999 in
# every case), every year in both styles, every round thousand and million, and
# half-round amounts to 99,500. What prose actually spells out is round numbers
# and years; a non-round six-figure value written in words is vanishingly rare,
# and an unmatched run keeps the suppression it has today — visible, and never a
# wrong integer.
_EAGER_VALUES = (
    list(range(1, 2101))
    + [k * 1000 for k in range(1, 1000)]
    + [k * 1000 + h * 100 for k in range(1, 100) for h in range(1, 10)]
    + [k * 10**6 for k in range(1, 1000)]
)

_PHRASES: dict[tuple[str, ...], int] | None = None


def phrases() -> dict[tuple[str, ...], int]:
    """Token sequence -> value, for sequences of two tokens or more.

    Built on first use, not at import: it is ~167k entries and ~0.1 s, and a
    caller rendering English never pays for it. Same reasoning as `render`'s
    deferred verifier.
    """
    global _PHRASES
    if _PHRASES is None:
        table: dict[tuple[str, ...], int] = {}
        for value in _EAGER_VALUES:
            for spelling in gen(value):
                if len(spelling) >= 2:
                    # A sequence that spells two different numbers would be a
                    # wrong integer waiting to happen. The audit asserts there
                    # are none; this keeps the first writer if one ever appears.
                    table.setdefault(spelling, value)
        _PHRASES = table
    return _PHRASES
