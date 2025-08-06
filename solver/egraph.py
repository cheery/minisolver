from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union

# "egg: Fast and Extensible Equality Saturation"
# "Relational E-Matching"
# "Better Together: Unifying Datalog and Equality Saturation"

class Contradiction(Exception):
    pass

constant = object()
term     = object()
minima   = object()
maxima   = object()
unify    = object()
check    = object()

@dataclass(eq=True, frozen=True)
class ENode:
    kind     : type
    children : Tuple[Any]

    def split(self):
        children = []
        etuple = []
        for sub, child in zip(self.kind.sub(), self.children):
            if sub == term or sub == constant:
                children.append(child)
            else:
                etuple.append(child)
        return self.kind, tuple(children), tuple(etuple)

ETuple = Tuple[Any]

@dataclass(eq=False)
class EClass:
    parent : Optional['EClass']
    terms  : Dict[type, Dict[ETuple, ETuple]]
    uses   : List[ENode]
    forbid : Set['EClass']

    @classmethod
    def new(cls, enode):
        kind, ekey, etuple = enode.split()
        return cls(None, {kind: {ekey: etuple}}, [], set())

    def __repr__(self):
        return str(id(self))

class EGraph:
    def __init__(self):
        self.eclasses = set()
        self.hashcons : Dict[ENode, int] = {}
        self.worklist : List[int] = []
        self.new_terms : Dict(type, List[ENode]) = {}
        self.is_saturated = True

    def find(self, bunch):
        if not isinstance(bunch, EClass):
            return bunch
        if bunch.parent is None:
            return bunch
        while bunch.parent.parent is not None:
            bunch.parent = bunch.parent.parent
        return bunch.parent

    def canonicalize(self, enode):
        return ENode(enode.kind, tuple(self.find(obj) for obj in enode.children))

    def add(self, kind, children):
        node = self.canonicalize(ENode(kind, children))
        try:
            return self.find(self.hashcons[node])
        except KeyError:
            self.eclasses.add(eclass := EClass.new(node))
            for child in node.children:
                if isinstance(child, EClass):
                    child.uses.append(node)
            self.hashcons[node] = eclass
            self.new_terms.setdefault(node.kind, []).append(node)
            self.is_saturated = False
            return eclass

    def merge(self, eclass1, eclass2):
        eclass1 = self.find(eclass1)
        eclass2 = self.find(eclass2)
        if eclass1 is eclass2:
            return eclass1
        if len(eclass1.uses) >= len(eclass2.uses):
            eclass2.parent = eclass1
            self.worklist.append(eclass2)
            self.is_saturated = False
            return eclass1
        else:
            eclass1.parent = eclass2
            self.worklist.append(eclass1)
            self.is_saturated = False
            return eclass2

    def merge_etuples(self, kind, ekey, etuple1, etuple2):
        xs = iter(etuple1)
        ys = iter(etuple2)
        out = []
        changed = False
        for sub in kind.sub():
            if sub == term or sub == constant:
                continue
            x = next(xs)
            y = next(ys)
            changed |= x != y
            if sub == minima:
                out.append(min(x, y))
            elif sub == maxima:
                out.append(max(x, y))
            elif sub == unify:
                out.append(self.merge(x, y))
            elif sub == check:
                if hasattr(x, "align"):
                    out.append(x.align(y))
                else:
                    if x != y:
                        raise Contradiction(f"{x} == {y}")
                    out.append(x)
        if changed:
            node = self.rejoin_enode(kind, ekey, out)
            self.new_terms.setdefault(node.kind, []).append(node)
        return tuple(out)

    def rejoin_enode(self, kind, ekey, etuple):
        return ENode(kind, tuple(self._rejoin_enode_children(kind, ekey, etuple)))

    def _rejoin_enode_children(self, kind, ekey, etuple):
        a = iter(ekey)
        b = iter(etuple)
        for sub in kind.sub():
            yield next(a if sub == term or sub == constant else b)

    def rebuild(self):
        while self.worklist:
            todo, self.worklist = self.worklist, []
            for eclass in todo:
                self.evict(eclass)
        self.is_saturated = True

    def evict(self, eclass):
        self.eclasses.discard(eclass)

        for p_node in eclass.uses:
            p_eclass = self.find(self.hashcons.pop(p_node))
            p_node = self.canonicalize(p_node)
            self.hashcons[p_node] = self.merge(p_eclass, self.hashcons.pop(p_node, p_eclass))

        n_eclass = self.find(eclass)
        for kind, table in eclass.terms.items():
            n_table = n_eclass.terms.setdefault(kind, {})
            for ekey, etuple in table.items():
                ekey = tuple(self.find(a) for a in ekey)
                etuple1 = tuple(self.find(a) for a in etuple)
                etuple2 = n_table.pop(ekey, None)
                if etuple2 is None:
                    n_table[ekey] = etuple1
                else:
                    n_table[ekey] = self.merge_etuples(kind, ekey, etuple1, etuple2)

        n_eclass = self.find(eclass)
        for other in eclass.forbid:
            other = self.find(other)
            if other is n_eclass:
                raise Contradiction
            other.add(n_eclass)
            n_eclass.add(other)

    def match(self, bind, xs, eclass, node):
        if -1 in bind and bind[-1] in xs and eclass != xs[bind[-1]]:
            return
        for i, j in bind.items():
            if i >= 0 and j in xs and node.children[i] != xs[j]:
                break
        else:
            return {-1: eclass} | {i: child for i, child in enumerate(node.children)}

    def new_relation(self, kind, bind, xs):
        for node in self.new_terms.get(kind, ()):
            node = self.canonicalize(node)
            eclassid = self.find(self.hashcons[node])
            m = self.match(bind, xs, eclassid, node)
            if m is not None:
                yield m

    def relation(self, kind, bind, xs):
        if -1 in bind and bind[-1] in xs:
            eclasses = iter([xs[bind[-1]]])
        else:
            eclasses = iter(self.eclasses)
        for eclass in eclasses:
            table = eclass.terms.get(kind, None)
            if table is None:
                continue
            for ekey, etuple in table.items():
                node = self.rejoin_enode(kind, ekey, etuple)
                m = self.match(bind, xs, eclass, node)
                if m is not None:
                    yield m

    def query(self, q, argc):
        pend = []
        out = set()
        def _pending_(q, pivot):
            if len(q) != 0:
                kind, bind = q[0]
                for m in self.new_relation(kind, bind, {}):
                    pend.append((pivot, {k: m[i] for i, k in bind.items()}))
                return _pending_(q[1:], pivot+1)
        def _execute_(q, xs, pivot):
            if len(q) == 0:
                out.add(tuple(xs[i] for i in range(argc)))
            elif pivot == 0:
                _execute_(q[1:], xs, pivot-1)
            else:
                kind, bind = q[0]
                for m in self.relation(kind, bind, xs):
                    ys = xs | {k: m[i] for i, k in bind.items()}
                    _execute_(q[1:], ys, pivot-1)
        _pending_(q, 0)
        for pivot, xs in pend:
            _execute_(q, xs, pivot)
        return out

    def run(self, rules):
        while not self.is_saturated:
            self.rebuild()
            matches = []
            for pattern, arity, process in rules:
                matches.extend((process, xs) for xs in self.query(pattern, arity))
            self.new_terms.clear()
            for process, args in matches:
                process(self, *args)

@dataclass(eq=True, frozen=True)
class expr:
    @classmethod
    def sub(cls):
        out = []
        for fld in fields(cls):
            if "sub" in fld.metadata:
                out.append(fld.metadata["sub"])
            elif fld.type == expr:
                out.append(term)
            else:
                out.append(constant)
        return out

@dataclass(eq=True, frozen=True)
class const(expr):
    value : int = field(metadata={"sub":check})

@dataclass(eq=True, frozen=True)
class add(expr):
    lhs : expr
    rhs : expr

@dataclass(eq=True, frozen=True)
class mul(expr):
    lhs : expr
    rhs : expr

eg = EGraph()
i0 = eg.add(add, (eg.add(const, (100,)), eg.add(const, (40,))))
i1 = eg.add(add, (eg.add(const, (20,)), eg.add(const, (60,))))
i2 = eg.add(mul, (i0, i1))

Q = [ (add,    {-1: 0, 0: -2, 1: -1}),
      (const, {-1: -2, 0: 1}),
      (const, {-1: -1, 0: 2}) ]
def process_Q(eg, eclassid, a, b):
    print(f"{a}+{b}={a+b}")
    t = eg.add(const, (a+b,))
    eg.merge(eclassid, t)

R = [ (mul,    {-1: 0, 0: -2, 1: -1}),
      (const, {-1: -2, 0: 1}),
      (const, {-1: -1, 0: 2}) ]
def process_R(eg, eclassid, a, b):
    print(f"{a}*{b}={a*b}")
    t = eg.add(const, (a*b,))
    eg.merge(eclassid, t)

rules = [(Q, 3, process_Q),
         (R, 3, process_R)]

eg.run(rules)

for eclass in eg.eclasses:
    print("EC", eclass)
    for kind, table in eclass.terms.items():
        for ekey, etuple in table.items():
            print(" ", eg.rejoin_enode(kind, ekey, etuple))
