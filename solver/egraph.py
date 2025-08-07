from collections import deque, defaultdict
from functools import cache
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from typing import get_type_hints

# "egg: Fast and Extensible Equality Saturation"
# "Relational E-Matching"
# "Better Together: Unifying Datalog and Equality Saturation"

class Contradiction(Exception):
    pass

class EMode:
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
            if sub == EMode.term or sub == EMode.constant:
                children.append(child)
            else:
                etuple.append(child)
        return self.kind, tuple(children), tuple(etuple)

ETuple = Tuple[Any]

@dataclass(eq=False)
class EClass:
    parent : Optional['EClass']
    terms  : Dict[type, Dict[ETuple, ETuple]]
    uses   : Set[ENode]
    forbid : Set['EClass']

    @classmethod
    def new(cls, enode):
        kind, ekey, etuple = enode.split()
        return cls(None, {kind: {ekey: etuple}}, set(), set())

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
                    child.uses.add(node)
            self.hashcons[node] = eclass
            self.new_terms.setdefault(node.kind, []).append(node)
            self.is_saturated = False
            return eclass

    def different(self, eclass1, eclass2):
        eclass1 = self.find(eclass1)
        eclass2 = self.find(eclass2)
        return eclass1 is not eclass2

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
            if sub == EMode.term or sub == EMode.constant:
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
            yield next(a if sub == EMode.term or sub == EMode.constant else b)

    def rebuild(self):
        while self.worklist:
            todo, self.worklist = self.worklist, []
            for eclass in todo:
                self.evict(eclass)
        self.is_saturated = True

    def evict(self, eclass):
        self.eclasses.discard(eclass)

        for p_node in list(eclass.uses):
            p_eclass = self.hashcons.pop(p_node)
            for child in p_node.children:
                if isinstance(child, EClass):
                    child.uses.discard(p_node)
            p_node = self.canonicalize(p_node)
            for child in p_node.children:
                if isinstance(child, EClass):
                    child.uses.add(p_node)
            o_eclass = self.hashcons.pop(p_node, p_eclass)
            self.hashcons[p_node] = self.merge(p_eclass, o_eclass)

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

    def enodes(self, eclass):
        for kind, table in eclass.terms.items():
            for ekey, etuple in table.items():
                yield self.rejoin_enode(kind, ekey, etuple)

    def extract(self, *eclasses):
        xtr = self.extract_eclasses(eclasses)
        res = tuple(xtr[eclass] for eclass in eclasses)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def extract_eclasses(self, eclasses):
        ec_costs = {}
        def best(eclass):
            if eclass in ec_costs:
                return ec_costs[eclass]
            min_cost = float('inf')
            best_enode = None
            ec_costs[eclass] = min_cost, best_enode
            for enode in self.enodes(eclass):
                costs = [best(child)[0] for child in enode.children if isinstance(child, EClass)]
                ecs   = [best(child)[1] for child in enode.children if isinstance(child, EClass)]
                cost = sum(costs) + enode.kind.cost(*(e.kind for e in ecs))
                if cost < min_cost:
                    min_cost = cost
                    best_enode = enode
            ec_costs[eclass] = min_cost, best_enode
            return min_cost, best_enode
        @cache
        def construct(eclass):
            enode = best(eclass)[1]
            args = []
            for child in enode.children:
                if isinstance(child, EClass):
                    args.append(construct(child))
                else:
                    args.append(child)
            return enode.kind(*args)
        out = {}
        for eclass in eclasses:
            out[eclass] = construct(eclass)
        return out

    def topological_order(self, eclasses):
        visited = set()
        output = []
        def visit(eclass):
            if eclass in visited:
                return
            visited.add(eclass)
            for enode in self.enodes(eclass):
                for child in enode.children:
                    if isinstance(child, EClass):
                        visit(child)
            output.append(eclass)
        for eclass in eclasses:
            visit(eclass)
        return output

    def extract_eclasses_dag(self, eclasses, temp):
        tail  = {}
        costs = {}
        for eclass in self.topological_order(eclasses):
            best_cost = float('inf')
            best_enode = None
            best_tail = None
            for enode in self.enodes(eclass):
                child_costs = 0
                enode_tail = set([])
                for child in enode.children:
                    if isinstance(child, EClass) and child not in costs:
                        break # cycle detected
                    if isinstance(child, EClass):
                        child_costs += costs[child][0]
                        enode_tail.add(child)
                        enode_tail.update(tail[child])
                else:
                    enode_cost = enode.kind.simple_cost + child_costs + len(enode_tail)
                    if enode_cost < best_cost:
                        best_cost = enode_cost
                        best_enode = enode
                        best_tail = enode_tail
            costs[eclass] = best_cost, best_enode
            tail[eclass] = best_tail
        created = {}
        shared = []
        def construct(eclass):
            if eclass in created:
                shared.append(created[eclass])
            enode = costs[eclass][1]
            children = []
            for child in enode.children:
                if isinstance(child, EClass):
                    children.append(construct(child))
                else:
                    children.append(child)
            created[eclass] = term = enode.kind(*children)
            return term
        terms = {}
        for eclass in eclasses:
            terms[eclass] = construct(eclass)
        return terms

    def __call__(self, u):
        if isinstance(u, EId):
            return u.eclass
        children = []
        for mode, field in zip(type(u).sub(), fields(u)):
            child = getattr(u, field.name)
            if mode == EMode.term or mode == EMode.unify:
                children.append(self(child))
            else:
                children.append(child)
        return self.add(type(u), children)

class EVariable:
    def __repr__(self):
        return f"{self.name}"

@dataclass(eq=False)
class EId:
    eclass : EClass
    def __repr__(self):
        return f"EId({self.eclass})"

def evariable(Type, name):
    ty = type(f"evariable({Type.__name__})", (EVariable, Type),
        {'name': name, 'base_type': Type})
    return ty()

def eid(Type, eclass):
    ty = type(f"eid({Type.__name__})", (EId, Type), {})
    return ty(eclass)

def pat_typeof(pat):
    if isinstance(pat, EVariable):
        return pat.base_type
    else:
        return type(pat)

def compile_pattern(inputs, pat):
    types = [pat_typeof(p) for p in pat] + list(inputs.values())
    slots = {name:len(pat)+i for i, name in enumerate(inputs.keys())}
    temp  = -1
    query = []
    def _compile_(p, i):
        nonlocal temp
        bind = {-1: i}
        k = 0
        for mode, field in zip(type(p).sub(), fields(p)):
            child = getattr(p, field.name)
            if isinstance(child, EVariable):
                bind[k] = slots[child.name]
            elif mode == EMode.term or mode == EMode.unify:
                bind[k] = temp
                _compile_(child, temp)
                temp -= 1
            else:
                raise TypeError
            k += 1
        query.insert(0, (type(p), bind))
    for i, p in enumerate(pat):
        assert not isinstance(p, EVariable)
        _compile_(p, i)
    return types, query

def ruleset(*functions, debug=False):
    rules = []
    def make(inputs):
        def _decorator_(*pat):
            def _decorator_(fn):
                types, pattern = compile_pattern(inputs, pat)
                def processor(eg, *args):
                    args = (eid(t,a) for t,a in zip(types, args))
                    for a, b in fn(*args):
                        a = eg(a)
                        b = eg(b)
                        if debug and eg.different(a, b):
                            x, y = eg.extract(a, b)
                            print(f"{x} = {y}")
                        eg.merge(a, b)
                rules.append((pattern, len(types), processor))
                return fn
            return _decorator_
        return _decorator_
    for function in functions:
        env = {}
        inputs = {}
        index = 0
        for name, ty in get_type_hints(function).items():
            env[name] = evariable(ty, name)
            inputs[name] = ty
            index += 1
        function(make(inputs), **env)
    return rules
