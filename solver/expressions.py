from collections import Counter
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
import math

class Unsatisfiable(Exception):
    pass

@dataclass(eq=False)
class Expr:
    precedence = 1000
    def __str__(self):
        return self.stringify(default_repr)

    def __post_init__(self):
        return validate(self)

    def __add__(self, other):
        other = convert(other)
        return distribute(self, other, Add)

    def __radd__(self, other):
        other = convert(other)
        return distribute(other, self, Add)

    def __sub__(self, other):
        other = convert(other)
        return distribute(self, other, lambda lhs, rhs: Add(lhs, Neg(rhs)))

    def __rsub__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: Add(lhs, Neg(rhs)))

    def __mul__(self, other):
        other = convert(other)
        return distribute(self, other, Mul)

    def __rmul__(self, other):
        other = convert(other)
        return distribute(other, self, Mul)

    def __truediv__(self, other):
        other = convert(other)
        return distribute(self, other, lambda lhs, rhs: Mul(lhs, Inv(rhs)))

    def __rtruediv__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: Mul(lhs, Inv(rhs)))

@dataclass(eq=False)
class Scalar(Expr):
    def __neg__(self):
        return Neg(self)

@dataclass(eq=False)
class Symbol(Scalar):
    def stringify(self, s):
        return f"Symbol()"

    def evaluate(self, context):
        return context[self]

@dataclass(eq=False)
class Constant(Scalar):
    value : float
    def stringify(self, s):
        return str(self.value)

    def evaluate(self, context):
        return self.value

@dataclass(eq=False)
class Unary(Scalar):
    scalar : Scalar
    def evaluate(self, context):
        scalar = context.compute(self.scalar)
        return self.op(scalar)

@dataclass(eq=False)
class Previous(Unary):
    def stringify(self, s):
        return f"Previous({s(self.scalar)})"

    def evaluate(self, context):
        return context.previous.compute(self.scalar)

@dataclass(eq=False)
class Neg(Unary):
    precedence = 10
    def op(self, scalar):
        return -scalar

    def stringify(self, s):
        return f"-{s(self.scalar)}"

@dataclass(eq=False)
class Inv(Unary):
    precedence = 20
    def stringify(self, s):
        return f"1 / {s(self.scalar)}"

    def op(self, scalar):
        if scalar > 1e-12:
            return 1 / scalar
        else:
            raise Unsatisfiable

@dataclass(eq=False)
class Sqr(Unary):
    def stringify(self, s):
        return f"Sqr({s(self.scalar)})"

    def op(self, scalar):
        return scalar*scalar

@dataclass(eq=False)
class Sqrt(Unary):
    def stringify(self, s):
        return f"Sqrt({s(self.scalar)})"

    def op(self, scalar):
        return math.sqrt(scalar)

@dataclass(eq=False)
class Abs(Unary):
    def stringify(self, s):
        return f"Abs({s(self.scalar)})"

    def op(self, scalar):
        return abs(scalar)

@dataclass(eq=False)
class Acos(Unary):
    def stringify(self, s):
        return f"Acos({s(self.scalar)})"

    def op(self, scalar):
        return math.acos(scalar)

@dataclass(eq=False)
class Binary(Scalar):
    lhs : Scalar
    rhs : Scalar
    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        return self.op(lhs, rhs)

@dataclass(eq=False)
class Add(Binary):
    precedence = 10
    def stringify(self, s):
        if isinstance(self.rhs, Neg):
            return f"{s(self.lhs, 9)} - {s(self.rhs.scalar, 10)}"
        else:
            return f"{s(self.lhs, 9)} + {s(self.rhs, 10)}"

    def op(self, lhs, rhs):
        return lhs + rhs

@dataclass(eq=False)
class Mul(Binary):
    precedence = 20
    def stringify(self, s):
        if isinstance(self.rhs, Inv):
            return f"{s(self.lhs, 19)} / {s(self.rhs.scalar, 20)}"
        else:
            return f"{s(self.lhs, 19)} * {s(self.rhs, 20)}"

    def op(self, lhs, rhs):
        return lhs * rhs

def convert(obj):
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, float):
        return Constant(obj)

def distribute(lhs, rhs, operand):
    if isinstance(lhs, Scalar) and isinstance(rhs, Scalar):
        return operand(lhs, rhs)
    elif isinstance(lhs, Compound) and isinstance(rhs, Compound):
        return lhs.compound(rhs, operand)
    elif isinstance(lhs, Compound) and isinstance(rhs, Scalar):
        return lhs.distribute(lambda a: operand(a, rhs))
    elif isinstance(lhs, Scalar) and isinstance(rhs, Compound):
        return rhs.distribute(lambda a: operand(lhs, a))
    else:
        raise TypeError

def default_repr(expr, precedence=0):
    if precedence < expr.precedence:
        return str(expr)
    else:
        return "(" + str(expr) + ")"

def validate(obj):
    for f in fields(obj):
        a = getattr(obj, f.name)
        if f.type == float:
            setattr(obj, f.name, float(a))
        if f.type == Constant:
            setattr(obj, f.name, Constant(a))
        assert isinstance(getattr(obj, f.name), f.type)

@dataclass(eq=False)
class NonZero(Expr):
    scalar : Scalar
    def stringify(self, s):
        return f"NonZero({s(self.scalar)})"

    def evaluate(self, context):
        scalar = context.compute(self.scalar)
        if scalar >= 1e-12:
            return 0
        else:
            raise Unsatisfiable

@dataclass(eq=False)
class Relation(Expr):
    lhs : Scalar
    rhs : Scalar

@dataclass(eq=False)
class Eq(Relation):
    def soft(self, weight):
        return SoftEq(self.lhs, self.rhs, weight)

    def stringify(self, s):
        return f"Eq({s(self.lhs)}, {s(self.rhs)})"

    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        return lhs - rhs

@dataclass(eq=False)
class SoftEq(Relation):
    weight : float

    def stringify(self, s):
        return f"Eq({s(self.lhs)}, {s(self.rhs)}).soft({self.weight})"

    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        return lhs - rhs

@dataclass(eq=False)
class Entity(Expr):
    def __post_init__(self):
        self.equations = list(self.constraints())

    def constraints(self):
        return iter(())

@dataclass(eq=False)
class Compound(Entity):
    def __neg__(self):
        return self.distribute(Neg)

@dataclass(eq=False)
class EvaluationContext:
    previous : Optional['EvaluationContext']

    def __getitem__(self, variable):
        raise NotImplemented

    def compute(self, scalar):
        raise NotImplemented

@dataclass(eq=False)
class JustContext(EvaluationContext):
    variables : Dict[Symbol, float]
    memo      : Dict[Scalar, float]
    def compute(self, scalar):
        if isinstance(scalar, Symbol):
            return self[scalar]
        try:
            return self.memo[scalar]
        except KeyError:
            self.memo[scalar] = value = scalar.evaluate(self)
            return value

    def __getitem__(self, variable):
        return self.variables[variable]

@dataclass(eq=False)
class VectoredContext(EvaluationContext):
    variables : Dict[Symbol, int]
    memo      : Dict[Scalar, float]
    x         : List[float]
    def compute(self, scalar):
        if isinstance(scalar, Symbol):
            return self[scalar]
        try:
            return self.memo[scalar]
        except KeyError:
            self.memo[scalar] = value = scalar.evaluate(self)
            return value

    def __getitem__(self, variable):
        return self.x[self.variables[variable]]

def all_exprs(system):
    visited = set()
    def visit(expr):
        if expr not in visited:
            visited.add(expr)
            for f in fields(expr):
                a = getattr(expr, f.name)
                if isinstance(a, Expr):
                    visit(a)
    for expr in system:
        visit(expr)
    return visited

def all_entities(system):
    return set(x for x in all_exprs(system) if isinstance(x, Entity))

def all_variables(system):
    out : Dict[Expr, Set[Symbol]] = {}
    def get_variables(expr):
        if isinstance(expr, Symbol):
            return set([expr])
        elif expr in out:
            return out[expr]
        else:
            out[expr] = s = set()
            for f in fields(expr):
                a = getattr(expr, f.name)
                if isinstance(a, Expr):
                    s.update(get_variables(a))
            return s
    total = set()
    for expr in system:
        total.update(get_variables(expr))
    return out, total

def print_system(system):
    counter = Counter()
    postorder = []
    def visit(expr):
        counter[expr] += 1
        if counter[expr] <= 1:
            for f in fields(expr):
                a = getattr(expr, f.name)
                if isinstance(a, Expr) and not isinstance(a, Constant):
                    visit(a)
            postorder.append(expr)
    for expr in system:
        visit(expr)
    i = 1
    names = {}
    def system_repr(expr, precedence=0):
        if expr in names:
            return names[expr]
        elif precedence < expr.precedence:
            return expr.stringify(system_repr)
        else:
            return "(" + str(expr) + ")"
    print("SYSTEM:")
    for expr in postorder:
        if isinstance(expr, Symbol):
            names[expr] = f"v{i}"
            i += 1
        elif counter[expr] > 1:
            print(f"  v{i} =", expr.stringify(system_repr))
            names[expr] = f"v{i}"
            i += 1
        elif expr in system:
            print(f" ", expr.stringify(system_repr))

zero = Constant(0.0)
one  = Constant(1.0)

