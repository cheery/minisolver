from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from solver.egraph import EMode, EGraph, ruleset

@dataclass(eq=True, frozen=True)
class expr:
    @classmethod
    def sub(cls):
        out = []
        for fld in fields(cls):
            if "sub" in fld.metadata:
                out.append(fld.metadata["sub"])
            elif fld.type == expr:
                out.append(EMode.term)
            else:
                out.append(EMode.constant)
        return out

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __str__(self):
        return repr(self)

    def __repr__(self):
       args = []
       for fld in fields(self):
           args.append(repr(getattr(self, fld.name)))
       return type(self).__name__ + "(" + ", ".join(args) + ")"

@dataclass(eq=True, frozen=True, repr=False)
class const(expr):
    value : int = field(metadata={"sub":EMode.check})
    simple_cost = 1
    @classmethod
    def cost(self):
        return 1

    def __repr__(self):
        return f"{self.value}"

@dataclass(eq=True, frozen=True, repr=False)
class symbol(expr):
    index : int
    simple_cost = 1

    @classmethod
    def cost(cls):
        return 1

    def __repr__(self):
        return f"s{self.index}"

@dataclass(eq=True, frozen=True, repr=False)
class add(expr):
    lhs : expr
    rhs : expr
    simple_cost = 1

    @classmethod
    def cost(cls, lhs, rhs):
        if const == lhs:
            return 2
        elif lhs == add:
            return 0
        else:
            return 1

    def __repr__(self):
        return f"({self.lhs}+{self.rhs})"

@dataclass(eq=True, frozen=True, repr=False)
class mul(expr):
    lhs : expr
    rhs : expr
    simple_cost = 2

    @classmethod
    def cost(cls, lhs, rhs):
        if lhs == add or rhs == add:
            return 10
        elif const == lhs:
            return 2
        elif lhs == mul:
            return 0
        return 1

    def __repr__(self):
        return f"({self.lhs}*{self.rhs})"

@dataclass(eq=True, frozen=True, repr=False)
class temp(expr):
    index : int
    simple_cost = 1

    def __repr__(self):
        return f"temp{self.index}"

eg = EGraph()

s0 = symbol(0)
s1 = symbol(1)
a = s0 + const(40)
b = const(20) + s1
term = eg((a * b) + (a * b))

def evaluable(m, x : float, y : float):
    @m(const(x) + const(y))
    def addition(a, x, y):
        yield a, const(x + y)
    @m(const(x) * const(y))
    def multiplication(a, x, y):
        yield a, const(x * y)

def commutative(m, x : expr, y : expr):
    @m(x + y)
    def addition(a, x, y):
        yield a, y + x
    @m(x * y)
    def multiplication(a, x, y):
        yield a, y * x

def rule3(m, x : expr, y : expr, z : expr):
    @m((x+y)*z)
    def multiplication(a, x, y, z):
        yield a, x*z + y*z

# This last rule makes it explode in complexity
def rule4(m, x : expr, y : expr, z : expr):
    @m(x+(y+z))
    def multiplication(a, x, y, z):
        yield a, (x+y)+z

rules = ruleset(evaluable, commutative, rule3, debug=True)

eg.run(rules)

root = eg.find(term)
print("=", eg.extract(root))

result = eg.extract_eclasses_dag([root], temp)
print("[2] =", result[root])



import sys
sys.exit()

vertextype = {}
vertexterm = {}
vertex_eclass = {}
numbering = {}
def num(u):
    assert u is None or isinstance(u, (EClass, ENode))
    if u in numbering:
        return numbering[u]
    else:
        numbering[u] = i = len(numbering)
        vertextype[i] = 'eclass' if isinstance(u, EClass) else 'enode'
        vertexterm[i] = u
        return i
graph = defaultdict(set), defaultdict(set)

def visit(eclass):
    if eclass in numbering:
        return num(eclass)
    else:
        a = num(eclass)
        for enode in eg.enodes(eclass):
            b = num(enode)
            graph[0][a].add(b)
            graph[1][b].add(a)
            vertex_eclass[b] = a
            for child in enode.children:
                if isinstance(child, EClass):
                    c = visit(child)
                    graph[0][b].add(c)
                    graph[0][c].add(b)
        return a

vertexroots = set([visit(root)])

#vertexroot = num(None)
#graph[0][vertexroot].add(visit(root))
#graph[1][visit(root)].add(vertexroot)

undirected = {}
for n in vertextype:
    undirected[n] = graph[0][n] | graph[1][n]

def min_fill_treewidth(graph):
    graph = {node: set(neighs) for node, neighs in graph.items()}
    bags = []
    elim = []
    elim_in = {}
    tw = 0
    while graph:
        # Choose node with minimal fill-in edges
        min_fill = None
        best_node = None
        for node in graph:
            neighbors = graph[node]
            fill_in = 0
            neigh_list = list(neighbors)
            for i in range(len(neigh_list)):
                for j in range(i + 1, len(neigh_list)):
                    u, v = neigh_list[i], neigh_list[j]
                    if u not in graph[v]:
                        fill_in += 1

            if min_fill is None or fill_in < min_fill:
                min_fill = fill_in
                best_node = node
        # Eliminate best_node
        neighbors = graph[best_node]
        tw = max(tw, len(neighbors))
        # Add fill edges (turn neighbors into a clique)
        neigh_list = list(neighbors)
        for i in range(len(neigh_list)):
            for j in range(i + 1, len(neigh_list)):
                u, v = neigh_list[i], neigh_list[j]
                graph[u].add(v)
                graph[v].add(u)
        # Create bag
        elim.append(bag_id := len(bags))
        elim_in[best_node] = bag_id
        bags.append(set([best_node] + neigh_list))
        # Remove best_node
        for neighbor in neighbors:
            graph[neighbor].remove(best_node)
        del graph[best_node]
    output = {i:set() for i in range(len(bags))}
    for i, (node, bag) in enumerate(zip(elim, bags)):
        j = max(elim_in[v] for v in bag if v != node)
        if i != j:
            output[i].add(j)
            output[j].add(i)
    return tw, bags, output

def make_nice_decomposition(bags, graph):
    n = len(bags)
    # 1) pick a root bag containing root_vertex
    rt = 0
    #roots = [i for i,b in enumerate(bags) if root_vertex in b]
    #if not roots:
    #    raise ValueError(f"No bag contains root vertex {root_vertex}")
    #rt = roots[0]

    # 2) root the tree: build parent/children relationships
    parent = {rt: None}
    children = {i: [] for i in range(n)}
    q = [rt]
    while q:
        u = q.pop()
        for v in graph[u]:
            if v == parent[u]: continue
            parent[v] = u
            children[u].append(v)
            q.append(v)

    nice_nodes = []
    def join(a, b, bag):
        self_id = len(nice_nodes)
        nice_nodes.append({'type': 'join', 'bag': bag, 'children': [a, b]})
        return self_id
    def leaf(bag):
        self_id = len(nice_nodes)
        nice_nodes.append({'type': 'leaf', 'bag': set(), 'children': []})
        return delta(self_id, set(), bag)
    def delta(self_id, current, target):
        for v in target - current:
            intr_id = len(nice_nodes)
            bag = current | {v}
            nice_nodes.append({'type': 'insert', 'bag': bag, 'children': [self_id], 'v': v})
            self_id = intr_id
            current = bag
        for v in current - target:
            intr_id = len(nice_nodes)
            bag = current - {v}
            nice_nodes.append({'type': 'forget', 'bag': bag, 'children': [self_id], 'v': v})
            self_id = intr_id
            current = bag
        return self_id
    def build(u, target):
        bag = bags[u]
        subbags = [build(c, bag) for c in children[u]]
        while len(subbags) > 1:
            m = len(subbags)-2
            while m >= 0:
                a, b = subbags[m:m+2]
                subbags[m:m+2] = [join(a, b, bag)]
                m -= 2
        if not subbags:
            self_id = leaf(bag)
        else:
            self_id = subbags[0]
        return delta(self_id, bag, target)
    return nice_nodes, build(rt, set())

w, bags, tgraph = min_fill_treewidth(undirected)
nice_nodes, root_id = make_nice_decomposition(bags, tgraph)

from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#@dataclass(frozen=True)
#class Data:
#    cost : float
#    e_costs : Dict[int, float]
#
#def calculate_cost(inc, u, data):
#    if isinstance(vertexterm[u], EClass):
#        return data
#    else:
#        if u in inc:
#            cost = data.cost + data.e_costs[vertex_eclass[u]]
#        else:
#            cost = data.cost
#    return Data(cost, data.e_costs)

count = 0
ans = {}

def answer(b):
    global count
    this = nice_nodes[b]
    B = this['bag']
    for m in powerset(B):
        m = frozenset(m)
        for s in powerset(x for x in m if vertextype[x] == 'eclass'):
            s = frozenset(s)
            ans[b,encode(B,m),encode(B,s)] = compute_dp(b, m, s)
    count += 1
    print("PROGRESS", count / len(nice_nodes) * 100, "%")
    return ans

def encode(bag, s):
    i = 0
    k = 1
    for x in bag:
        if x in s:
            i |= k
        k <<= 1
    return i

def compute_dp(b, m, s):
    this = nice_nodes[b]
    def gans(b,m,s):
        B = nice_nodes[b]['bag']
        return ans[b, encode(B, m), encode(B, s)]

    kind = this['type']
    for u in m:
        if vertextype[u] == 'enode':
            for v in graph[0][u]:
                if v in this['bag']:
                    if v not in m:
                        return float('inf')
    if not (this['bag'] & vertexroots).issubset(m):
        return float('inf')
    elif kind == 'leaf':
        return 0.0
    elif kind == 'insert':
        u = this['v']
        c = this['children'][0]
        if u not in m:
            return gans(c, m, s)
        if vertextype[u] == 'eclass' and u not in s and len(m & graph[0][u]) == 0:
            return float('inf')
        Obu = frozenset(x for x in graph[1][u] if vertextype[x] == 'eclass' and x in m)
        U = frozenset([u])
        return gans(c, m - U, (s | Obu) - U) + calculate_cost(U)
    elif kind == 'forget':
        u = this['v']
        c = this['children'][0]
        U = frozenset([u])
        return min(gans(c, m, s), gans(c, m | U, s))
    elif kind == 'join':
        c, d = this['children']
        return gans(c,m,s) + gans(d,m,s) + calculate_cost(m)

def calculate_cost(us):
    return len(us)
    
#    if kind == "leaf":
#        return {frozenset(): Data(0.0, {})}
#    elif kind == "forget":
#        p = answer(nice_nodes[b]['children'][0])
#        u = nice_nodes[b]['v']
#        U = frozenset([u])
#        q = {}
#        for inc, data in p.items():
#            data = calculate_cost(inc, u, data)
#            inc = inc - U
#            if inc not in q or data.cost < q[inc].cost:
#                q[inc] = data
#        return q
#    elif kind == "insert":
#        p = answer(nice_nodes[b]['children'][0])
#        u = nice_nodes[b]['v']
#        U = frozenset([u])
#        q = {}
#        if isinstance(vertexterm[u], EClass):
#            for inc, data in p.items():
#                data = Data(data.cost, data.e_costs | {u: 0.0})
#                q[inc | U] = data
#                q[inc]     = data
#            return q
#        else:
#            eclass = vertex_eclass[u]
#            for inc, data in p.items():
#                if eclass not in set(vertex_eclass[i] for i in inc if i in vertex_eclass):
#                    ecost = 1.0
#                    for v in graph[0][u]:
#                        if v in data.e_costs[v]:
#                        ecost += data.e_costs[v]
#                    ecost += data.e_costs.get(eclass, 0.0)
#                    q[inc | U] = Data(data.cost, data.e_costs | {eclass: ecost})
#                else:
#                    print("Q")
#                    assert False
#                q[inc] = data
#            return q
##        q = {}
##        if isinstance(vertexterm[u], EClass):
##            for inc, data in p.items():
##                q[inc]     = data
##        else:
##            for inc, data in p.items():
##                    q[inc | U] = data
##                q[inc]     = data
##        return q
#    elif kind == "join":
#        p = answer(nice_nodes[b]['children'][0])
#        q = answer(nice_nodes[b]['children'][1])
#    assert False, "TODO " + kind

for b in range(len(nice_nodes)):
    answer(b)

print("ANS", ans[root_id,0,0])

print("MIN FILL", w)

#import graphviz
#dot = graphviz.Digraph('G', 'tree decomposition')
#for i, edges in graph.items():
#    dot.node(str(i))#, repr(bags[i]))
#    for j in edges:
#        dot.edge(str(i), str(j))
#dot.view()

#print("L", len(nice_nodes))
#dot = graphviz.Digraph('G', 'tree decomposition')
#for i, node in enumerate(nice_nodes):
#    l = ",".join(str(n) for n in node['bag'])
#    dot.node(str(i), "{" + l + "}")
#for i, node in enumerate(nice_nodes):
#    for j in node['children']:
#        dot.edge(str(i), str(j))
#dot.view()

#root_vertex = (eg.find(term),)
#
#transitive = {}
#backward   = {}
#edges = {}
#for eclass in root_vertex:
#    transitive.setdefault(root_vertex, set())
#    transitive.setdefault(eclass, set()).add(root_vertex)
#    edges.setdefault(eclass, set()).add(root_vertex)
#    edges.setdefault(root_vertex, set()).add(eclass)
#    backward.setdefault(root_vertex, set()).add(eclass)
#
#for node, eclass in eg.hashcons.items():
#    eclass = eg.find(eclass)
#    transitive.setdefault(eclass, set())
#    transitive.setdefault(node, set()).add(eclass)
#    edges.setdefault(node, set()).add(eclass)
#    edges.setdefault(eclass, set()).add(node)
#    backward.setdefault(node, set())
#    backward.setdefault(eclass, set()).add(node)
#    for child in node.children:
#        if isinstance(child, EClass):
#            transitive.setdefault(child, set()).add(node)
#            edges.setdefault(child, set()).add(node)
#            edges.setdefault(node, set()).add(child)
#            backward.setdefault(node, set()).add(child)
#
#for v, e in transitive.items():
#    if len(e) == 0 and v != root_vertex:
#        transitive.pop(v)
#
#forward    = {}
#for v, e in transitive.items():
#    forward[v] = e.copy()
#
#changed = True
#while changed:
#    changed = False
#    for v, e in transitive.items():
#        n = len(e)
#        for w in list(e):
#            e.update(transitive[w])
#        m = len(e)
#        changed |= (n < m)
#
#print("ORIGINAL", len(edges))
#
#def tree_decomposition(G):
#
#def make_nice_decomposition(bags, dag, root_vertex):
#    """
#    Inputs:
#      - bags: list of sets, the original decomposition bags
#      - dag: dict mapping bag_idx -> list of neighboring bag_idxs (may be a DAG)
#      - root_vertex: a graph vertex that must lie in the root bag
#
#    Returns:
#      - nice_nodes: list of dicts (leaf/introduce/forget/join)
#      - root_id: index of the root node in nice_nodes
#    """
#
#    # 1) pick a root bag containing root_vertex
#    roots = [i for i,b in enumerate(bags) if root_vertex in b]
#    if not roots:
#        raise ValueError(f"No bag contains root vertex {root_vertex!r}")
#    rt = roots[0]
#
#    # 2) do a BFS from rt to build a spanning tree of the DAG
#    parent = {rt: None}
#    children = {i: [] for i in range(len(bags))}
#    q = deque([rt])
#    while q:
#        u = q.popleft()
#        for v in dag[u]:
#            if v in parent:
#                continue               # already visited ⇒ skip
#            parent[v] = u
#            children[u].append(v)
#            q.append(v)
#
#    # 3) recursively build nice decomposition over that tree
#    nice_nodes = []
#    def build(u, is_root=False):
#        # build nice subtree for bag‐node u
#        # first process all children in the spanning tree
#        child_ids = [build(v) for v in children[u]]
#
#        # if no children ⇒ start with a leaf
#        if not child_ids:
#            leaf_id = len(nice_nodes)
#            nice_nodes.append({
#                'type': 'leaf',
#                'bag': set(),
#                'children': []
#            })
#            cur_id = leaf_id
#        else:
#            # if one child ⇒ we'll massage that child's bag to bags[u]
#            if len(child_ids) == 1:
#                cur_id = child_ids[0]
#            else:
#                # join them pairwise
#                ids = child_ids[:]
#                while len(ids) > 1:
#                    a = ids.pop()
#                    b = ids.pop()
#                    join_id = len(nice_nodes)
#                    nice_nodes.append({
#                        'type': 'join',
#                        'bag': bags[u].copy(),
#                        'children': [a, b]
#                    })
#                    ids.append(join_id)
#                cur_id = ids[0]
#
#        # now transform cur_id's bag → bags[u]
#        B_cur = nice_nodes[cur_id]['bag']
#        target = set() if is_root else bags[u]
#
#        # introduce any missing vertices
#        for v in sorted(target - B_cur, key=id):
#            nid = len(nice_nodes)
#            nice_nodes.append({
#                'type': 'insert',
#                'bag': B_cur | {v},
#                'v': v,
#                'children': [cur_id]
#            })
#            cur_id = nid
#            B_cur = nice_nodes[cur_id]['bag']
#
#        # forget any extra vertices
#        for v in sorted(B_cur - target, key=id):
#            nid = len(nice_nodes)
#            nice_nodes.append({
#                'type': 'forget',
#                'bag': B_cur - {v},
#                'v': v,
#                'children': [cur_id]
#            })
#            cur_id = nid
#            B_cur = nice_nodes[cur_id]['bag']
#
#        # safety check
#        if B_cur != target:
#            raise RuntimeError(f"Bag mismatch at u={u}")
#
#        return cur_id
#
#    root_id = build(rt, True)
#    return nice_nodes, root_id
#
#
#bags, tree = tree_decomposition(edges)
#nice_nodes, root_id = make_nice_decomposition(bags, tree, root_vertex)
#for node in nice_nodes:
#    bag = node['bag']
#    node['edges'] = e = {}
#    for v in bag:
#        e[v] = transitive[v] & bag
#
#t = {root_vertex}
#ans = {}
#done = set()
#def process(b):
#    if b in done:
#        return
#    done.add(b)
#    for child in nice_nodes[b]['children']:
#        process(child)
#    for m in powerset(nice_nodes[b]['bag']):
#        for s in powerset(c for c in m if isinstance(c, EClass)):
#            computedp(b, frozenset(m), frozenset(s))
#
#def gans(b, m, s):
#    return ans.get((b,m,s), float('inf'))
#
#def computedp(b, m, s):
#    for u in m:
#        if not isinstance(u, EClass):
#            for v in backward[u]:
#                if v not in m:
#                    ans[b,m,s] = float('inf')
#                    return
#    Vb = nice_nodes[b]['bag']
#    if not (Vb & t).issubset(m):
#        ans[b,m,s] = float('inf')
#        return
#    kind = nice_nodes[b]['type']
#    if kind == 'leaf':
#        ans[b,m,s] = 0.0
#    elif kind == 'insert':
#        u = nice_nodes[b]['v']
#        c = nice_nodes[b]['children'][0]
#        if u not in m:
#            ans[b,m,s] = ans[c,m,s]
#        elif isinstance(u, EClass) and u not in s and len(m & backward[u]) == 0:
#            ans[b,m,s] = float('inf')
#        else:
#            Obu = frozenset(v for v in forward[u] if isinstance(v, EClass))
#            ans[b,m,s] = gans(c, m - frozenset({u}), (s | Obu) - frozenset({u})) + (u.kind.cost if isinstance(u, ENode) else 0)
#    elif kind == 'forget':
#        u = nice_nodes[b]['v']
#        c = nice_nodes[b]['children'][0]
#        ans[b,m,s] = min(ans[c,m,s], ans[c, m | frozenset({u}), s])
#    elif kind == 'join':
#        c1, c2 = nice_nodes[b]['children']
#        ans[b,m,s] = gans(c1,m,s) + gans(c2,m,s) - sum(x.kind.cost for x in m if isinstance(x, ENode))
#process(root_id)
#
#print("RESULT", ans[root_id, frozenset(), frozenset()])
#
#renamings = {}
#for v in transitive:
#    if v not in renamings:
#        renamings[v] = len(renamings)
#
##print("treewidth", max(len(node['bag'])-1 for node in nice_nodes))
##print(len(nice_nodes))
#
#import graphviz
#dot = graphviz.Digraph('G', 'tree decomposition')
