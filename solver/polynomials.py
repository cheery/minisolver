from fractions import Fraction
from functools import total_ordering
from itertools import combinations
from math import gcd
import bisect
import sys

first  = lambda x: x[0]
second = lambda x: x[1]

def lockstep(xs, ys, default):
    terminal = (None, default)
    x = next(xs, terminal)
    y = next(ys, terminal)
    while x[0] is not None and y[0] is not None:
        if x[0] < y[0]:
            yield x[0], x[1], default
            x = next(xs, terminal)
        elif y[0] < x[0]:
            yield y[0], default, y[1]
            y = next(ys, terminal)
        else:
            yield x[0], x[1], y[1]
            x = next(xs, terminal)
            y = next(ys, terminal)
    while x[0] is not None:
        yield x[0], x[1], default
        x = next(xs, terminal)
    while y[0] is not None:
        yield y[0], default, y[1]
        y = next(ys, terminal)

@total_ordering    # grevlex
class Monomial:
    __slots__ = ("exps",)

    def __init__(self, exps):
        self.exps = tuple(exps)
        assert all(exp > 0 for var, exp in self.exps), self.exps

    @classmethod
    def new(cls, x):
        if x is None:
            return cls([])
        return cls([(x, 1)])

    def __hash__(self):
        return hash(self.exps)

    def __eq__(self, other):
        return self.exps == other.exps

    @property
    def degree(self):
        return sum(degree for _, degree in self.exps)

    def common(self, other):
        return lockstep(iter(self.exps), iter(other.exps), 0)

    def __lt__(self, other):
        deg_self = self.degree
        deg_other = other.degree
        if deg_self != deg_other:
            return deg_self < deg_other
        lst = tuple(self.common(other))
        for _, a, b in reversed(lst):
            if a != b:
                return a > b
        return False

    def __mul__(self, other):
        return Monomial((x, ei + ej) for x, ei, ej in self.common(other))

    def __pow__(self, coeff):
        return Monomial((x, e * coeff) for x, e in self.exps)

    def divides(self, other):
        return all(ei <= ej for _, ei, ej in self.common(other))

    def __truediv__(self, other):
        return Monomial((x, ei - ej) for x, ei, ej in self.common(other) if ei != ej)

    def lcm(self, other):
        return Monomial((x, max(ei, ej)) for x, ei, ej in self.common(other))

    def pretty(self):
        return "".join(f"*{repr(x)}**{e}" if e > 1 else f"*{repr(x)}"
                       for x, e in self.exps)

    def __repr__(self):
        exps = ",".join(f"{repr(x)}**{e}" for x, e in self.exps)
        return f"Monomial({exps})"

class Polynomial:
    __slots__ = ("terms", "_lm_cache")
    def __init__(self, terms=None, prune=True):
        self.terms = {} if prune else terms
        if terms and prune:
            for m, c in terms.items():
                if c != 0:
                    self.terms[m] = Fraction(c)
        self._lm_cache = None

    def canonical(self):
        if not self.terms:
            return self
        coeffs = list(self.terms.values())
        # GCD of all numerators, LCM of all denominators
        num_gcd = abs(coeffs[0].numerator)
        den_lcm = abs(coeffs[0].denominator)
        for c in coeffs[1:]:
            num_gcd = gcd(num_gcd, abs(c.numerator))
            den_lcm = den_lcm * c.denominator // gcd(den_lcm, c.denominator)
        factor = Fraction(num_gcd, den_lcm)
        if factor == 0:
            return self
        return Polynomial({m: c / factor for m, c in self.terms.items()}, False)

    def monic(self):
        lc = self.lc()
        if lc == 1:
            return self
        return Polynomial({m: c / lc for m, c in self.terms.items()}, False)

    def __add__(self, other):
        res = self.terms.copy()
        for m, c in other.terms.items():
            res[m] = res.get(m, Fraction(0)) + c
            if res[m] == 0:
                del res[m]
        return Polynomial(res, False)

    def __sub__(self, other):
        res = self.terms.copy()
        for m, c in other.terms.items():
            res[m] = res.get(m, Fraction(0)) - c
            if res[m] == 0:
                del res[m]
        return Polynomial(res, False)

    def mul_by_monomial(self, coeff, mono):
        if coeff == 0:
            return Polynomial()
        res_terms = {}
        for m, c in self.terms.items():
            res_terms[m * mono] = c * coeff
        return Polynomial(res_terms, False)

    def mul(self, other):
        res_terms = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                mprod = m1 * m2
                res_terms[mprod] = res_terms.get(mprod, Fraction(0)) + c1 * c2
        return Polynomial(res_terms, False)

    def lm(self):
        if self._lm_cache is None and self.terms:
            self._lm_cache = max(self.terms.keys())
        return self._lm_cache

    def lc(self):
        lm = self.lm()
        return self.terms[lm] if lm else Fraction(0)

    def __bool__(self):
        return bool(self.terms)

    def __eq__(self, other):
        M = set(self.terms) | set(other.terms)
        if len(M) == len(self.terms) == len(other.terms):
            for m in M:
                if self.terms[m] != other.terms[m]:
                    return False
            return True
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.terms.items())))

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for m in sorted(self.terms.keys(), reverse=True):
            coeff = self.terms[m]
            parts.append(f"{coeff}{m.pretty()}")
        return " + ".join(parts)

    def copy(self):
        return Polynomial(self.terms.copy(), False)

    def substitute(self, env):
        c = Monomial(())
        poly = Polynomial()
        for m, coeff in self.terms.items():
            x = Polynomial({c: coeff})
            exps = []
            for var, exp in m.exps:
                if var in env:
                    y = env[var]
                    for _ in range(exp):
                        x = x.mul(y)
                else:
                    exps.append((var, exp))
            poly += x.mul_by_monomial(1, Monomial(exps))
        return poly

class MonomialTrie:
    __slots__ = ("children", "index")
    def __init__(self):
        self.index = -1
        self.children = {}

    def insert(self, exps, index):
        node = self
        for var, exp in exps:
            branch = node.children.setdefault(var, [])
            for e, child in branch:
                if e == exp:
                    node = child
                    break
            else:
                node = MonomialTrie()
                bisect.insort(branch, (exp, node), key=first)
        if node.index == -1:
            node.index = index

    def find(self, exps):
        if len(exps) == 0:
            return self.index
        var, exp = exps[0]
        exps     = exps[1:]
        index    = self.find(exps)
        if index >= 0:
            return index
        for e, node in self.children.get(var, ()):
            if e <= exp:
                index = node.find(exps)
                if index >= 0:
                    return index
        return -1

    def debug(self, depth=0):
        print(" "*depth, self.index)
        for var, block in self.children.items():
          print(" "*depth, "VARIABLES", var)
          for e, node in block:
              print(" "*depth, "EXPONENT", e)
              node.debug(depth+2)

class Polynomials:
    def __init__(self, polys, parent=None):
        self.parent = parent
        self.cutoff = 0 if parent is None else len(parent)
        self.polys = []
        self.trie = MonomialTrie()
        for poly in sorted(polys, key=lm):
            self.append(poly)

    def __getitem__(self, index):
        if index < self.cutoff:
            return self.parent[index]
        return self.polys[index - self.cutoff]

    def __iter__(self):
        if self.parent is not None:
            yield from self.parent
        yield from self.polys

    def __len__(self):
        return self.cutoff + len(self.polys)

    def append(self, poly):
        self.trie.insert(poly.lm().exps, len(self.polys))
        self.polys.append(poly)

    def covered(self, poly):
        return self.trie.find(poly.lm().exps) >= 0

    def find_index(self, m):
        if self.parent is not None:
            if p := self.parent.find_index(m):
                return p
        index = self.trie.find(m.exps)
        if index >= 0:
            return self.cutoff + index
        return -1

    def find(self, m):
        if self.parent is not None:
            if p := self.parent.find(m):
                return p
        index = self.trie.find(m.exps)
        if index >= 0:
            return self.polys[index]
        
    def rem_of(self, f, debug=False):
        r = Polynomial()
        p = f
        while p:
            mono_p = p.lm()
            if g := self.find(mono_p):
                mono_q = mono_p / g.lm()
                coeff_q = p.lc() / g.lc()
                if debug:
                    print("PPPP", p)
                    print("    ", coeff_q)
                    print("hmm ", g.mul_by_monomial(coeff_q, mono_q))
                p = p - g.mul_by_monomial(coeff_q, mono_q)
            else:
                c = Polynomial({p.lm(): p.lc()})
                r = r + c
                p = p - c
            if p: # TODO: not neede if everything works.
                assert p.lm() < mono_p, str((p.lm(), mono_p))
        return r

    def reduce(self):
        F = self
        while True:
            G = Polynomials([], F.parent)
            H = Polynomials([], F.parent)
            for f in sorted(F.polys, key=lm):
                if len(G) == 0:
                    H.append(f.monic())
                elif r := G.rem_of(f):
                    H.append(r.monic())
                G.append(f)
            if G.polys == H.polys:
                return G
            F = H

    def substitute(self, env):
        if not env:
            return self
        polys = []
        for poly in self.polys:
            poly = poly.substitute(env)
            if poly:
                polys.append(poly)
        return reduce(polys, self.parent)

    @property
    def empty(self):
        return self.trie.index >= 0

    @property
    def solved(self):
        solved = set()
        for poly in self.polys:
            lm = poly.lm()
            if len(lm.exps) == 1 and lm.exps[0][1] == 1:
                solved.add(lm.exps[0][0])
        return solved

#def div(f, G):
#    quotients = [Polynomial() for _ in G]
#    r = Polynomial()
#    p = f
#    while p:
#        for i, g in enumerate(G):
#            if g.lm().divides(p.lm()):
#                mono_q = p.lm() / g.lm()
#                coeff_q = p.lc() / g.lc()
#                q_term = Polynomial({mono_q: coeff_q})
#                quotients[i] = quotients[i] + q_term
#                p = p - g.mul_by_monomial(coeff_q, mono_q)
#                break
#        else:
#            c = Polynomial({p.lm(): p.lc()})
#            r = r + c
#            p = p - c
#    return quotients, r

def reduce(F, parent):
    G = Polynomials([], parent)
    H = Polynomials([], parent)
    for f in sorted(F, key=lm):
        if len(G) == 0:
            H.append(f.monic())
        elif r := G.rem_of(f):
            H.append(r.monic())
        G.append(f)
    if G.polys == H.polys:
        return G
    return H.reduce()

def lm(poly):
    lm = poly.lm()
    if lm is None:
        raise ValueError(f"polynomial is null")
    return lm

def degree(poly):
    return poly.lm().degree

def s_polynomial(f, g):
    lcm_m = f.lm().lcm(g.lm())
    mono_f = lcm_m / f.lm()
    mono_g = lcm_m / g.lm()
    coeff_f = Fraction(1) / f.lc()
    coeff_g = Fraction(1) / g.lc()
    return f.mul_by_monomial(coeff_f, mono_f) - g.mul_by_monomial(coeff_g, mono_g)

def buchberger(F, parent=None):
    G = reduce(F, parent)
    pairs = [(i, j) for i in range(G.cutoff, len(G)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        S = s_polynomial(G[i], G[j])
        if R := G.rem_of(S):
            G.append(R.monic())
            for k in range(len(G) - 1):
                pairs.append((len(G) - 1, k))
    return G.reduce()

class MonomialTable:
    def __init__(self): 
        self._to_idx = {}
        self._from_idx = []
    def ensure(self, mono):
        if mono in self._to_idx: return self._to_idx[mono]
        idx = len(self._from_idx)
        self._to_idx[mono] = idx
        self._from_idx.append(mono)
        return idx
    def get_index(self, mono): return self._to_idx[mono]
    def get_monomial(self, idx): return self._from_idx[idx]
    def sorted_monomials_desc(self):
        return sorted(self._from_idx, reverse=True)
    def __len__(self): return len(self._from_idx)

def collect_pair_rows(deg, pairs, G, Sig, computed, mon_table):
    """
    pairs: list of tuples (i,j) indices into G
    G: list of Polynomial basis
    mon_table: MonomialTable (will be filled)
    Returns:
      rows: list of dict{col_index: Fraction}
      columns: list of Monomial (ordered desc by monomial)
    """
    rows = []
    mono_set = set()
    def add(poly, sig, lm=None):
        if sig in computed:
            return
        for m in poly.terms.keys(): include(m)
        rows.append((sig, poly.terms))
    def include(m):
        if m not in mono_set:
            mono_set.add(m)
            k = G.find_index(m)
            if k >= 0:
                g = G[k]
                q = m / g.lm()
                h = g.mul_by_monomial(Fraction(1), q)
                for _m in h.terms.keys(): include(_m)
                sig = Sig[k][0], q*Sig[k][1]
                rows.append((sig, h.terms))

    for (i, j) in pairs:
        gi, gj = G[i], G[j]
        if gi.lm() is None or gj.lm() is None: 
            continue
        L = gi.lm().lcm(gj.lm())
        # u * LM(gi) = L  => u = L / LM(gi)
        ui = L / gi.lm()
        uj = L / gj.lm()
        # rows: ui*gi and uj*gj, but multiply polynomials directly
        rowi_poly = gi.mul_by_monomial(Fraction(1) / gi.lc(), ui)
        rowj_poly = gj.mul_by_monomial(Fraction(1) / gj.lc(), uj)
        index, w = Sig[i]
        add(rowi_poly, (index, w*ui))
        index, w = Sig[j]
        add(rowj_poly, (index, w*uj))
        #rows.append({m: c for m, c in rowi_poly.terms.items()})
        #meta.append( (i, ui) )
        #rows.append({m: c for m, c in rowj_poly.terms.items()})
        #meta.append( (j, uj) )
    # build monomial table indices for all monos, sorted descending
    sorted_monos = sorted(mono_set, reverse=True)
    for m in sorted_monos:
        mon_table.ensure(m)
    # convert rows from monomial->coeff to col_index->coeff
    indexed_rows = []
    for sig, r in sorted(rows, key=first, reverse=True):
        idx_row = {}
        for m, c in r.items():
            idx = mon_table.get_index(m)
            idx_row[idx] = Fraction(c)
        indexed_rows.append((sig, idx_row))
    # columns list: indices sorted ascending correspond to sorted_monos order
    columns = [mon_table.get_monomial(i) for i in range(len(mon_table._from_idx))]
    # sort by signature
    #_indexed_rows = []
    #_meta = []
    #for i in sorted(range(len(rows)), key=lambda x: meta[x]):
    #    _indexed_rows.append(indexed_rows[i])
    #    _meta.append(meta[i])
    return indexed_rows, columns #, meta

def sparse_eliminate(rows):
    """
    rows: list of dict {col_idx: Fraction}
    Returns list of reduced rows (dict col->coeff) in row-echelon form with unique pivot columns.
    Leftmost pivot rule: leftmost = min(col_index) because columns are ordered descending monomial -> index 0 is largest monomial.
    """
    pivot_for_col = {}      # col_idx -> pivot_row dict
    reduced_rows = []       # will accumulate pivot rows
    for sig, row in rows: #, sig in zip(rows, meta):
        # make a mutable copy
        r = dict(row)
        # eliminate using existing pivots
        while r:
            left = min(r.keys())  # leftmost nonzero column (smallest index)
            if left not in pivot_for_col:
                # normalize? optional. We'll store as-is
                #pivot_for_col[left] = [r, sig]
                #reduced_rows.append((r, sig))
                #if sig[0] >= 0:
                pivot_for_col[left] = sig, r
                reduced_rows.append((sig, r))
                break
            #pivot, other_sig = pivot_for_col[left]
            sg, pivot = pivot_for_col[left]
            sig = min(sig, sg)
            # multiplier = r[left] / pivot[left]
            factor = r[left] / pivot[left]
            # r = r - factor * pivot
            # iterate over pivot entries
            for cidx, pcoeff in pivot.items():
                r[cidx] = r.get(cidx, Fraction(0)) - factor * pcoeff
                if r[cidx] == 0:
                    del r[cidx]
            # continue: find new leftmost
    # Optionally, we can return reduced_rows; they each have distinct leftmost keys
    return reduced_rows

def rows_to_polys(reduced_rows, columns):
    """
    columns: list of Monomial indexed by column index (0..)
    reduced_rows: list of dict col->Fraction
    returns list of Polynomial
    """
    polys = []
    for sig, r in reduced_rows:
        terms = {}
        for idx, coeff in r.items():
            mono = columns[idx]
            if coeff != 0:
                terms[mono] = Fraction(coeff)
        polys.append((sig, Polynomial(terms, False)))
    return polys

def f4_algorithm(G, parent=None):
    G = reduce(G, parent)
    Sig = [(i, Monomial([])) for i in range(len(G))]
    computed = set()
    batches = []
    for i in range(G.cutoff, len(G)):
        for j in range(i):
            degree = G[i].lm().lcm(G[j].lm()).degree
            batches.append((degree, i, j))
    if not batches:
        return G
    batches.sort(key=first)
    while batches:
        deg = batches[0][0]
        for n in range(len(batches)):
            if deg < batches[n][0]:
                break
        else:
            n += 1
        batch, batches = [(i,j) for _, i, j in batches[:n]], batches[n:]
        mon_table = MonomialTable()
        rows, columns = collect_pair_rows(deg, batch, G, Sig, computed, mon_table)
        if not rows:
            continue
        reduced = sparse_eliminate(rows)
        new_polys = rows_to_polys(reduced, columns)
        for sig, p in new_polys:
            computed.add(sig)
            if not G.covered(p):
            #if not any(g.lm().divides(p.lm()) for g in tG):
                p = p.monic()
                G.append(p)
                Sig.append(sig)
                new_index = len(G) - 1
                for k in range(new_index):
                    degree = G[new_index].lm().lcm(G[k].lm()).degree
                    bisect.insort(batches, (degree, k, new_index), key=first)
    return G.reduce()

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass(eq=True, order=True, frozen=True)
    class Symbol:
        name : str
        def __repr__(self):
            return self.name
    
    c = Monomial.new(None)

    x1 = Monomial.new(Symbol("x1"))
    x2 = Monomial.new(Symbol("x2"))
    x3 = Monomial.new(Symbol("x3"))
    x4 = Monomial.new(Symbol("x4"))
    x5 = Monomial.new(Symbol("x5"))
    x6 = Monomial.new(Symbol("x6"))
    x7 = Monomial.new(Symbol("x7"))

    def cycle_n(n):
        mon = [x1,x2,x3,x4,x5,x6,x7]
        
        # helper for cyclic products (indices 1-based, wrapping around)
        def prod_vars(start, length):
            vars_list = [mon[((start+i-1)%n)] for i in range(length)]
            m = vars_list[0]
            for v in vars_list[1:]:
                m = m * v
            return m
        
        system = []
        for k in range(1, n):
            f = Polynomial({prod_vars(i, k): 1 for i in range(1, n+1)})  # degree 1 terms
            system.append(f)
        prod_all = prod_vars(1, n)
        f = Polynomial({prod_all: 1, c: -1})
        system.append(f)
        return system

    def surfaces():
        x = Monomial.new(Symbol("x"))
        y = Monomial.new(Symbol("y"))
        z = Monomial.new(Symbol("z"))
        xy = x*y
        x2 = x**2
        y2 = y**2
        z2 = z**2

        f1 = Polynomial({x2: 2, x: -4, y2: 1, y: -4, c:3})
        f2 = Polynomial({x2: 1, x: -2, y2: 3, y: -12, c:9})
        return [f1, f2]

    F = surfaces()
    print("Problemn:")
    for f in F:
        print(" ", f)

    G = buchberger(F)
    print("Grobner basis:")
    for g in G:
        print(" ", g)

    G = f4_algorithm(F)
    print("Grobner basis (F4):")
    for g in G:
        print(" ", g)

    system = cycle_n(3)
    print("Problem")
    for f in system:
        print(" ", f)

    G = buchberger(system)
    print("Grobner basis:")
    for g in G:
        print(" ", g)

    G = f4_algorithm(system)
    print("Grobner basis (F4):")
    for g in G:
        print(" ", g)
    print("x3 = ", G.rem_of(Polynomial({x3: 1})))

    h = Polynomial({x1: 1, c: 1})
    H = f4_algorithm([h], G)
    print("Grobner basis (constrained):")
    for h in H:
        print(" ", h)
    print("it's empty:", H.empty)

    q1 = Polynomial({x1: 4, x2: 15, c: 10})
    G = f4_algorithm([q1])
    print("grobner basis (F4):")
    for g in G:
        print(" ", g)
    subs = {}
    for s in G.solved:
        subs[s] = G.rem_of(Polynomial({Monomial.new(s): 1}))
        print(f"{s} = {subs[s]}")
    G = G.substitute(subs)
    print("it's empty:", G.empty)
    print("grobner basis (subs):")
    for g in G:
        print(" ", g)

    for i in range(5,7):
        q2 = subs[x1.exps[0][0]]
        q2 = q2 + Polynomial({c: Fraction(-i,9)})
        H = f4_algorithm([q2], G)
        print("Extended grobner basis (F4):")
        for h in H:
            print(" ", h)
        table = {}
        for s in H.solved:
            table[s] = H.rem_of(Polynomial({Monomial.new(s): 1}))
            print(f"{s} = {table[s]}")
        H = H.substitute(table)
        print("Extended grobner basis (subs):")
        for h in H:
            print(" ", h)
