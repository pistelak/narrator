"""Independent ACCEPTOR for Czech cardinal spellings: sequence -> set of values.

This is the invariant the v1 generator lacked. It validates WELL-FORMEDNESS,
not just non-collision: every token is morphologically tagged (value, cases,
gender, number) and a linear walk enforces case unification, gender agreement
between 1/2-coefficients and their scale noun, and coefficient-class agreement
(jeden milion / dva miliony / pět milionů / dvacet jedna milionů). It shares
the FORM inventory with the generator (unavoidable — the forms are the data)
but none of the composition code: the generator crosses products, this
unifies features, so a cross-product leak in one is a rejection in the other.

Deliberately accepts a superset in a few paired-form corners (documented
inline)
the invariant direction is: everything GENERATED must be accepted
with exactly its value, and known junk must be rejected.
"""

ALLC = frozenset({"nom", "acc", "gen", "dat", "loc", "ins"})
NA = frozenset({"nom", "acc"})

def _t(v, cases, kind="plain", genders=None, frozen=False):
    return (v, frozenset(cases), kind, genders, frozen)

# --- token facts: small numerals -------------------------------------------
_FACTS = {}
def _add(tok, *entries):
    _FACTS.setdefault(tok, []).extend(entries)

_add("jeden", _t(1, NA, "decl1", frozenset("m")))
_add("jedna", _t(1, {"nom"}, "decl1", frozenset("f")),
              _t(1, ALLC, "frozen1"))
_add("jedno", _t(1, NA, "decl1", frozenset("n")))
_add("jednu", _t(1, {"acc"}, "decl1", frozenset("f")))
_add("jednoho", _t(1, {"gen"}, "decl1", frozenset("m")))
_add("jednomu", _t(1, {"dat"}, "decl1", frozenset("m")))
_add("jednom",  _t(1, {"loc"}, "decl1", frozenset("m")))
_add("jedním",  _t(1, {"ins"}, "decl1", frozenset("m")))
_add("jedné",   _t(1, {"gen", "dat", "loc"}, "decl1", frozenset("f")))
_add("jednou",  _t(1, {"ins"}, "decl1", frozenset("f")))
_add("dva", _t(2, NA, "two", frozenset("m")))
_add("dvě", _t(2, NA, "two", frozenset({"f", "n"})))
_add("dvou", _t(2, {"gen", "loc"}))
_add("dvěma", _t(2, {"dat", "ins"}))
_add("tři", _t(3, NA))
_add("tří", _t(3, {"gen"}))
_add("třech", _t(3, {"gen", "loc"}))
_add("třem", _t(3, {"dat"}))
_add("třemi", _t(3, {"ins"}))
_add("čtyři", _t(4, NA))
_add("čtyř", _t(4, {"gen"}))
_add("čtyřech", _t(4, {"gen", "loc"}))
_add("čtyřem", _t(4, {"dat"}))
_add("čtyřmi", _t(4, {"ins"}))
_OBL = frozenset({"gen", "dat", "loc", "ins"})
for v, w, o in [(5,"pět","pěti"),(6,"šest","šesti"),(7,"sedm","sedmi"),
                (8,"osm","osmi"),(9,"devět","devíti")]:
    _add(w, _t(v, NA))
    _add(o, _t(v, _OBL))
for v, w, o in [(10,"deset","deseti"),(11,"jedenáct","jedenácti"),
                (12,"dvanáct","dvanácti"),(13,"třináct","třinácti"),
                (14,"čtrnáct","čtrnácti"),(15,"patnáct","patnácti"),
                (16,"šestnáct","šestnácti"),(17,"sedmnáct","sedmnácti"),
                (18,"osmnáct","osmnácti"),(19,"devatenáct","devatenácti")]:
    _add(w, _t(v, NA))
    _add(o, _t(v, _OBL))
_add("desíti", _t(10, _OBL))
_TENS = [(20,"dvacet","dvaceti"),(30,"třicet","třiceti"),(40,"čtyřicet","čtyřiceti"),
         (50,"padesát","padesáti"),(60,"šedesát","šedesáti"),(70,"sedmdesát","sedmdesáti"),
         (80,"osmdesát","osmdesáti"),(90,"devadesát","devadesáti")]
_TEN_TOKENS = {}
for v, w, o in _TENS:
    _add(w, _t(v, NA))
    _add(o, _t(v, _OBL))
    _TEN_TOKENS[w] = (v, NA)
    _TEN_TOKENS[o] = (v, _OBL)
for u, prefixes in {1:["jedena","jedna"],2:["dvaa"],3:["třia"],4:["čtyřia"],
                    5:["pěta"],6:["šesta"],7:["sedma"],8:["osma"],9:["devěta"]}.items():
    for p in prefixes:
        for v, w, o in _TENS:
            _add(p + w, _t(v + u, NA))
            _add(p + o, _t(v + u, _OBL))

_UNIT_VALUES = frozenset(range(1, 10))

# --- hundreds: (tokens...) -> {(h, cases)} ---------------------------------
_HUNDREDS = {
    ("sto",): {(1, NA)}, ("sta",): {(1, frozenset({"gen"}))},
    ("stu",): {(1, frozenset({"dat", "loc"}))}, ("stě",): {(1, frozenset({"loc"}))},
    ("dvě", "stě"): {(2, NA)}, ("dvou", "set"): {(2, frozenset({"gen"}))},
    ("dvěma", "stům"): {(2, frozenset({"dat"}))},
    ("dvou", "stech"): {(2, frozenset({"loc"}))},
    ("dvěma", "sty"): {(2, frozenset({"ins"}))},
}
for h, n, g in [(3, "tři", ("tří", "třech")), (4, "čtyři", ("čtyř", "čtyřech"))]:
    _HUNDREDS[(n, "sta")] = {(h, NA)}
    for gf in g:
        _HUNDREDS[(gf, "set")] = {(h, frozenset({"gen"}))}
    d = "třem" if h == 3 else "čtyřem"
    loc = "třech" if h == 3 else "čtyřech"
    i = "třemi" if h == 3 else "čtyřmi"
    _HUNDREDS[(d, "stům")] = {(h, frozenset({"dat"}))}
    _HUNDREDS[(loc, "stech")] = {(h, frozenset({"loc"}))}
    _HUNDREDS[(i, "sty")] = {(h, frozenset({"ins"}))}
for h, n, o in [(5,"pět","pěti"),(6,"šest","šesti"),(7,"sedm","sedmi"),
                (8,"osm","osmi"),(9,"devět","devíti")]:
    _HUNDREDS[(n, "set")] = {(h, NA)}
    _HUNDREDS[(o, "set")] = {(h, frozenset({"gen"}))}
    _HUNDREDS[(o, "stům")] = {(h, frozenset({"dat"}))}
    _HUNDREDS[(o, "stech")] = {(h, frozenset({"loc"}))}
    _HUNDREDS[(o, "sty")] = {(h, frozenset({"ins"}))}

# --- scale nouns: token -> set of slots ------------------------------------
# slots: 'sg:<case>' | 'pl:nom' (counting plural: tisíce) | 'pl:gen'
# (counted genitive: tisíc/tisíců) | 'pl:dat/loc/ins'
_SCALES = [
    (10**9, "f", {
        "miliarda": {"sg:nom"}, "miliardu": {"sg:acc"},
        "miliardy": {"sg:gen", "pl:nom"}, "miliardě": {"sg:dat", "sg:loc"},
        "miliardou": {"sg:ins"}, "miliard": {"pl:gen"},
        "miliardám": {"pl:dat"}, "miliardách": {"pl:loc"}, "miliardami": {"pl:ins"},
    }),
    (10**6, "m", {
        "milion": {"sg:nom", "sg:acc"}, "milión": {"sg:nom", "sg:acc"},
        "milionu": {"sg:gen", "sg:dat", "sg:loc"}, "miliónu": {"sg:gen", "sg:dat", "sg:loc"},
        "milionem": {"sg:ins"}, "miliónem": {"sg:ins"},
        "miliony": {"pl:nom", "pl:ins"}, "milióny": {"pl:nom", "pl:ins"},
        "milionů": {"pl:gen"}, "miliónů": {"pl:gen"},
        "milionům": {"pl:dat"}, "miliónům": {"pl:dat"},
        "milionech": {"pl:loc"}, "miliónech": {"pl:loc"},
    }),
    (1000, "m", {
        "tisíc": {"sg:nom", "sg:acc", "pl:gen"}, "tisíce": {"sg:gen", "pl:nom"},
        "tisíci": {"sg:dat", "sg:loc", "pl:ins"}, "tisícem": {"sg:ins"},
        "tisíců": {"pl:gen"}, "tisícům": {"pl:dat"}, "tisících": {"pl:loc"},
    }),
]

def _coeff_class(k):
    if k == 1:
        return "one"
    if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14):
        return "few"
    return "many"

def _tail_parse(toks):
    """1..99 -> {(v, cases, kind, genders)}
    kind in decl1/frozen1/two/plain."""
    out = set()
    if len(toks) == 1:
        for v, cases, kind, genders, _ in _FACTS.get(toks[0], []):
            out.add((v, cases, kind, genders))
    elif len(toks) == 2 and toks[0] in _TEN_TOKENS:
        tv, tc = _TEN_TOKENS[toks[0]]
        for v, cases, kind, genders, _ in _FACTS.get(toks[1], []):
            if v not in _UNIT_VALUES:
                continue
            shared = tc & cases if kind != "frozen1" else tc
            if shared:
                out.add((tv + v, frozenset(shared), kind, genders))
    return out

def _seg_parse(toks):
    """1..999 -> {(v, cases, kind, genders)} with the frozen-nominative
    hundred-head licence ("dvě stě padesáti")."""
    out = set(_tail_parse(toks))
    for i in (1, 2):
        head = tuple(toks[:i])
        if head not in _HUNDREDS or len(toks) < i:
            continue
        for h, hcases in _HUNDREDS[head]:
            rest = toks[i:]
            if not rest:
                cases = ALLC if hcases & NA else hcases
                out.add((h * 100, cases, "plain", None))
                continue
            for v, cases, kind, genders in _tail_parse(rest):
                shared = hcases & cases
                if shared:
                    out.add((h * 100 + v, frozenset(shared), kind, genders))
                if hcases & NA:            # frozen head, declined tail
                    out.add((h * 100 + v, cases, kind, genders))
    return out

def _expected_slots(k, case, kind):
    cls = _coeff_class(k) if kind not in ("frozen1", "digit1") else \
          ("many" if kind == "frozen1" else "one")
    if kind in ("decl1", "empty", "digit1") or cls == "one":
        return {f"sg:{case}"}
    slots = set()
    def one(cl):
        if cl == "few":
            return {"pl:nom"} if case in ("nom", "acc") else \
                   {"pl:gen"} if case == "gen" else {f"pl:{case}"}
        return {"pl:gen"} if case in ("nom", "acc", "gen") else {f"pl:{case}"}
    slots |= one(cls)
    if cls == "few" and k > 4:
        slots |= one("many")               # dvacet dva tisíc beside ...tisíce
    return slots

def _group_parse(coeff, unit_tok, scale_gender, slots):
    """-> {(k, case)} for one scale group."""
    cands = []
    if not coeff:
        cands.append((1, ALLC, "empty", None))
    elif len(coeff) == 1 and coeff[0].isdigit():
        k = int(coeff[0])
        if str(k) == coeff[0] and 1 <= k <= 999:
            cands.append((k, ALLC, "digit1" if k == 1 else "digit", None))
    else:
        for v, cases, kind, genders in _seg_parse(list(coeff)):
            cands.append((v, cases, kind, genders))
    out = set()
    for k, cases, kind, genders in cands:
        if kind in ("decl1", "two") and genders and scale_gender not in genders:
            continue                       # "jedna tisíc", "dva miliardy"
        if kind == "decl1" and not (k % 10 == 1 and k % 100 != 11):
            continue
        if kind == "frozen1" and not (
                k > 1 and k % 10 == 1 and k % 100 != 11):
            continue                       # bare "jedna tisíc" is junk
        for case in cases:
            if _expected_slots(k, case, kind) & slots:
                out.add((k, "na" if case in ("nom", "acc") else case))
    return out

def accept(seq):
    """-> set of integer readings of a well-formed cardinal (empty if none)."""
    toks = list(seq)
    results = set()
    # year style: devatenáct set [osmdesát devět]
    if len(toks) >= 2 and toks[1] == "set":
        for v, cases, kind, _, _ in _FACTS.get(toks[0], []):
            if 11 <= v <= 19 and kind == "plain" and cases & NA:
                rest = toks[2:]
                if not rest:
                    results.add(v * 100)
                else:
                    for tv, tc, _, _ in _tail_parse(rest):
                        if tc & NA:
                            results.add(v * 100 + tv)
    # scale groups, strictly descending, each at most once
    positions = []
    ok = True
    prev = -1
    for scale, gender, forms in _SCALES:
        idx = [i for i, t in enumerate(toks) if t in forms]
        if len(idx) > 1:
            ok = False
            break
        if idx:
            if idx[0] <= prev:
                ok = False
                break
            positions.append((scale, gender, forms, idx[0]))
            prev = idx[0]
    if ok:
        groups = []            # each: {(value_contribution, caseset-per-parse)}
        start = 0
        for scale, gender, forms, i in positions:
            parses = _group_parse(tuple(toks[start:i]), toks[i], gender,
                                  forms[toks[i]])
            groups.append({(k * scale, case) for k, case in parses})
            start = i + 1
        rest = toks[start:]
        if rest:
            groups.append({(v, "na" if case in ("nom", "acc") else case)
                           for v, cases, _, _ in _seg_parse(rest)
                           for case in cases})
        if groups and all(groups):
            def combine(gs, case_set):
                if not gs:
                    return {0} if case_set else set()
                vals = set()
                for v, c in gs[0]:
                    shared = case_set & {c}
                    if shared:
                        vals |= {v + w for w in combine(gs[1:], shared)}
                return vals
            # a case must unify across the whole phrase (nom/acc as one)
            for case in ("na", "gen", "dat", "loc", "ins"):
                for total in combine(groups, {case}):
                    if total:
                        results.add(total)
    return results
