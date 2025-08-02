import numpy as np
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union

@dataclass(eq=False)
class variables:
    mapping : Dict['variable', float]
    memo    : Dict['expr', Any]
    def __getitem__(self, o):
        if isinstance(o, variable):
            return self.mapping[o]
        elif o in self.memo:
            return self.memo[o]
        else:
            self.memo[o] = v = o.evaluate(self)
            return v

    def has_var(self, var):
        return var in self.mapping

@dataclass(eq=False)
class variablesvector:
    mapping : Dict['variable', int]
    memo    : Dict['expr', Any]
    vector  : List[float]
    def __getitem__(self, o):
        if isinstance(o, variable):
            return self.vector[self.mapping[o]]
        elif o in self.memo:
            return self.memo[o]
        else:
            self.memo[o] = v = o.evaluate(self)
            return v

def setup(constraints, p):
    constraints = expand(constraints, expr)
    x0 = collect_variables(constraints, p)
    mangle = lambda x: variablesvector(x0.mapping, {}, x)
    def f(x):
        x = mangle(x)
        out = []
        for expr in constraints:
            out.extend(expr.hard(x, x0))
        return np.array(out, float)
    def g(x):
        x = mangle(x)
        out = []
        for expr in constraints:
            out.extend(expr.soft(x, x0))
        return np.array(out, float)
    g_w = []
    for a in constraints:
        g_w.extend(a.soft_weights)
    g_w = np.array(g_w, float)
    return f, g, g_w, x0.vector.copy(), mangle

def collect_variables(constraints, p):
    mapping = {}
    vector  = []
    for a in expand(constraints, (expr, scalar)):
        if isinstance(a, variable):
            mapping[a] = len(vector)
            vector.append(p[a] if p.has_var(a) else 0.0)
    return variablesvector(mapping, {}, np.array(vector, float))

def expand(args, ty):
    emitted = set()
    out = []
    def visit(a):
        if a not in emitted:
            emitted.add(a)
            out.append(a)
            for field in fields(a):
                o = getattr(a, field.name)
                if isinstance(o, ty):
                    visit(o)
    for a in args:
        visit(a)
    return out

@dataclass(eq=False, frozen=True)
class scalar:
    pass

@dataclass(eq=False, frozen=True)
class variable(scalar):
    pass

@dataclass(eq=False, frozen=True)
class constant(scalar):
    value : float = 0.0
    def evaluate(self, x):
        return self.value

@dataclass(eq=False)
class expr:
    def hard(self, x, x0):
        return iter(())

    soft_weights = ()
    def soft(self, x, x0):
        return iter(())

class constraint(expr):
    pass

zero = constant(0.0)
one  = constant(1.0)
