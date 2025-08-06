from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union

# "egg: Fast and Extensible Equality Saturation"
# "Relational E-Matching"
# "Better Together: Unifying Datalog and Equality Saturation"

class DisjointSet:
    def __init__(self):
        self.parent = []
        self.bunches = {}

    def __getitem__(self, index):
        return self.bunches[self.find(index)]

    def __len__(self):
        return len(self.bunches)

    def __iter__(self):
        return iter(self.bunches)

    def items(self):
        return self.bunches.items()

    def new(self, bunch):
        self.parent.append(i := len(self.parent))
        self.bunches[i] = bunch
        return i

    def find(self, index):
        while index != self.parent[index]:
            self.parent[index] = index = self.parent[self.parent[index]]
        return index

    def union(self, i, j):
        i = self.find(i)
        j = self.find(j)
        if i != j:
            self.parent[i] = j
            self.bunches[j] |= self.bunches.pop(i)
            return True, j
        else:
            return False, j

constant = object()
term     = object()
minima   = object()
maxima   = object()
unify    = object()

@dataclass(eq=True, frozen=True)
class ENode:
    kind     : type
    children : Tuple[Any]

    def masked(self):
        out = []
        a = iter(self.children)
        for sub in self.kind.sub():
            if sub == term or sub == constant:
                out.append((sub == term, next(a)))
        return out

    def split(self):
        children = []
        etuple = []
        for sub, child in zip(self.kind.sub(), self.children):
            if sub == term or sub == constant:
                children.append(child)
            else:
                etuple.append(child)
        return ENode(self.kind, tuple(children)), tuple(etuple)

    def join(self, etuple):
        children = []
        a = iter(self.children)
        b = iter(etuple)
        for sub in self.kind.sub():
            children.append(next(a if sub == term or sub == constant else b))
        return ENode(self.kind, tuple(children))

    def merge(self, egraph, left, right):
        a = iter(left)
        b = iter(right)
        out = []
        changed = False
        for sub in self.kind.sub():
            if sub == term or sub == constant:
                continue
            x = next(a)
            y = next(b)
            changed |= x != y
            if sub == minima:
                out.append(min(x, y))
            elif sub == maxima:
                out.append(max(x, y))
            elif sub == unify:
                out.append(egraph.merge(x, y))
        return changed, tuple(out)

ETuple = Tuple[Any]

@dataclass(eq=False)
class EClass:
    nodes : Set[ENode]
    #data : Any
    uses : Dict[ENode, 'EClass']
    def __or__(self, other : 'EClass'):
        #return EClass(self.nodes | other.nodes, self.data | other.data, self.uses | other.uses)
        return EClass(self.nodes | other.nodes, self.uses | other.uses)

class EGraph:
    def __init__(self):
        #self.analysis = Analysis(self)
        self.eclasses = DisjointSet()
        self.hashcons : Dict[ENode, int] = {}
        self.etuples : Dict[ENode, ETuple] = {}
        self.worklist : List[int] = []
        self.pending_nodes    : Set[ENode] = set()
        #self.pending_eclasses : Set[int] = set()
        self.is_saturated = True

    def canonicalize(self, node):
        return ENode(node.kind, tuple(self.eclasses.find(obj) if p else obj for p, obj in node.masked()))

    def add(self, node):
        node, etuple = node.split()
        node = self.canonicalize(node)
        if node in self.hashcons:
            made, etuple = node.merge(self, etuple, self.etuples[node])
            if made:
                self.etuples[node] = etuple
                self.is_saturated = False
                self.pending_nodes.add(node)
            return self.eclasses.find(self.hashcons[node])
        else:
            #eclassid = self.eclasses.new(EClass({node}, self.analysis.make(node), {}))
            eclassid = self.eclasses.new(EClass({node}, {}))
            for p, child in node.masked():
                if p:
                    self.eclasses[child].uses[node] = eclassid
            self.hashcons[node] = eclassid
            self.etuples[node] = etuple
            #eclassid = self.analysis.modify(eclassid)
            self.is_saturated = False
            self.pending_nodes.add(node)
            return eclassid

    def merge(self, i, j):
        made, k = self.eclasses.union(i, j)
        if made:
            self.is_saturated = False
            #self.pending_eclasses.add(k)
            self.worklist.append(k)

    def rebuild(self):
        while self.worklist:
            todo = set(self.eclasses.find(i) for i in self.worklist)
            self.worklist.clear()
            for eclassid in todo:
                self.repair(eclassid)
        self.is_saturated = True

    def repair(self, eclassid):
        eclass = self.eclasses[self.eclasses.find(eclassid)]
        uses, eclass.uses = eclass.uses, {}
        etuple_sets = []

        for p_node, p_eclassid in uses.items():
            p_eclassid = self.eclasses.find(p_eclassid)
            p_eclass = self.eclasses[p_eclassid]
            p_eclass.nodes.discard(p_node)
            self.hashcons.pop(p_node)
            p_etuple = self.etuples.pop(p_node)
            p_node = self.canonicalize(p_node)
            self.hashcons[p_node] = p_eclassid
            p_eclass.nodes.add(p_node)
            etuple_sets.append((p_node, p_etuple))

        merges = []
        for p_node, p_eclassid in uses.items():
            p_node = self.canonicalize(p_node)
            if p_node in eclass.uses:
                merges.append((p_eclassid, uses[p_node]))
            eclass.uses[p_node] = self.eclasses.find(p_eclassid)

        for p_node, p_etuple in etuple_sets:
            p_node = self.canonicalize(p_node)
            if p_node in self.etuples:
                _, p_etuple = p_node.merge(self, p_etuple, self.etuples[p_node])
            self.etuples[p_node] = p_etuple

        for i, j in merges:
            self.merge(i, j)

        #eclassid = self.analysis.modify(eclassid)
        #for p_node, p_eclassid in self.eclasses[self.eclasses.find(eclassid)].uses.items():
        #    p_eclass = self.eclasses[self.eclasses.find(p_eclassid)]
        #    new_data = p_eclass.data | self.analysis.make(p_node)
        #    if new_data != p_eclass.data:
        #        p_eclass.data = new_data
        #        self.worklist.append(p_eclassid)

    def examine(self, kind, bind, xs, eclassid, node):
        if node.kind != kind:
            return
        if -1 in bind and bind[-1] in xs and eclassid != xs[bind[-1]]:
            return
        node = node.join(self.etuples[node])
        for i, j in bind.items():
            if i >= 0 and j in xs and node.children[i] != xs[j]:
                break
        else:
            return {-1: eclassid} | {i: child for i, child in enumerate(node.children)}

    def pending_relation(self, kind, bind, xs):
        #if -1 in bound:
        #    if bound[-1] not in self.pending_eclasses:
        #        eclasses = iter(())
        #    eclasses = iter([bound[-1]])
        #else:
        #    eclasses = iter(self.pending_eclasses)
        for node in self.pending_nodes:
            node = self.canonicalize(node)
            eclassid = self.hashcons[node]
            m = self.examine(kind, bind, xs, eclassid, node)
            if m is not None:
                yield m
        #for eclassid in eclasses:
        #    for node in self.eclasses[eclassid].nodes:
        #        m = self.examine(kind, bound, bind, eclassid, node)
        #        if m is not None:
        #            yield m

    def relation(self, kind, bind, xs):
        if -1 in bind and bind[-1] in xs:
            eclasses = iter([xs[bind[-1]]])
        else:
            eclasses = iter(self.eclasses)
        for eclassid in eclasses:
            for node in self.eclasses[eclassid].nodes:
                m = self.examine(kind, bind, xs, eclassid, node)
                if m is not None:
                    yield m

    def query(self, q, argc):
        pend = []
        out = set()
        def _pending_(q, pivot):
            if len(q) != 0:
                kind, bind = q[0]
                for m in self.pending_relation(kind, bind, {}):
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

    def reset(self):
        self.pending_nodes.clear()
        #self.pending_eclasses.clear()

# class DummyAnalysis:
#     def __init__(self, egraph):
#         self.egraph = egraph
# 
#     def make(self, node):
#         return set()
# 
#     def modify(self, eclassid):
#         return eclassid

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
    value : int #= field(metadata={"sub":minima})

@dataclass(eq=True, frozen=True)
class add(expr):
    lhs : expr
    rhs : expr

@dataclass(eq=True, frozen=True)
class mul(expr):
    lhs : expr
    rhs : expr

eg = EGraph()
i0 = eg.add(ENode(add, (eg.add(ENode(const, (100,))), eg.add(ENode(const, (100,))))))
i1 = eg.add(ENode(mul, (i0, i0)))

while not eg.is_saturated:
    eg.rebuild()
    
    Q = [ (add,    {-1: 0, 0: -2, 1: -1}),
          (const, {-1: -2, 0: 1}),
          (const, {-1: -1, 0: 2}) ]

    R = [ (mul,    {-1: 0, 0: -2, 1: -1}),
          (const, {-1: -2, 0: 1}),
          (const, {-1: -1, 0: 2}) ]
    
    matches = []
    matches.extend((0, xs) for xs in eg.query(Q, 3))
    matches.extend((1, xs) for xs in eg.query(R, 3))
    eg.reset()
        
    for rule, xs in matches:
        if rule == 0:
            t = eg.add(ENode(const, (xs[1] + xs[2],)))
            eg.merge(xs[0], t)
        if rule == 1:
            t = eg.add(ENode(const, (xs[1] * xs[2],)))
            eg.merge(xs[0], t)

print("EC")
for i, ec in eg.eclasses.items():
    print(i, ec)
