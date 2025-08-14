from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from solver.egraph import EClass, EMode, EGraph, ruleset

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
        return f"temp[{self.index}]"

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

rules = ruleset(evaluable, commutative, rule3, debug=False)

eg.run(rules)

root = eg.find(term)

print("=", eg.extract(root))

temps, terms = eg.extract_dag(temp, root)
for i, t in enumerate(temps):
    print(f"[{i}] =", t)
for t in terms:
    print(t)
