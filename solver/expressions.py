from collections import Counter
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
import math

class Unsatisfiable(Exception):
    pass

class Nondifferentiable(Exception):
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

    def evaluate(self, context):
        attrs = {}
        for f in fields(self):
            attrs[f.name] = context.compute(getattr(self, f.name))
        return type(self)(**attrs)

# TODO: subexpressions(self)

@dataclass(eq=False)
class Scalar(Expr):
    def __neg__(self):
        return Neg(self)

@dataclass(eq=False)
class Symbol(Scalar):
    def stringify(self, s):
        return f"Symbol()"

    def evaluate(self, context):
        try:
            return context[self]
        except KeyError:
            return self

@dataclass(eq=False)
class Constant(Scalar):
    value : float
    def stringify(self, s):
        return str(self.value)

    def evaluate(self, context):
        return self.value

    def __float__(self):
        return self.value

@dataclass(eq=False)
class Unary(Scalar):
    scalar : Scalar
    def evaluate(self, context):
        scalar = context.compute(self.scalar)
        if isinstance(scalar, Dual):
            result = self.op(scalar.scalar)
            partials = {}
            for sym, d in scalar.partials.items():
                partials[sym] = self.dop(scalar.scalar, d, result)
            return Dual(result, partials)
        elif isinstance(scalar, Expr):
            return type(self)(scalar)
        else:
            return self.op(scalar)

@dataclass(eq=False)
class Previous(Unary):
    def stringify(self, s):
        return f"Previous({s(self.scalar)})"

    def evaluate(self, context):
        if context.previous is not None:
            scalar = context.previous.compute(self.scalar)
            if isinstance(scalar, Expr):
                return type(self)(scalar)
            else:
                return scalar
        else:
            return self

@dataclass(eq=False)
class Neg(Unary):
    precedence = 10
    def op(self, scalar):
        return -scalar

    def stringify(self, s):
        return f"-{s(self.scalar)}"

    def dop(self, scalar, dscalar, result):
        return -dscalar

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

    def dop(self, scalar, dscalar, result):
        if scalar < 1e-12:
            raise Nondifferentiable
        return -dscalar / sqr(scalar)

@dataclass(eq=False)
class Sqr(Unary):
    def stringify(self, s):
        return f"Sqr({s(self.scalar)})"

    def op(self, scalar):
        return scalar*scalar

    def dop(self, scalar, dscalar, result):
        return 2 * scalar * dscalar

def sqr(obj):
    if isinstance(obj, Scalar):
        return Sqr(obj)
    elif isinstance(obj, Compound):
        return obj.distribute(sqr)
    else:
        return float(obj)**2

@dataclass(eq=False)
class Sqrt(Unary):
    def stringify(self, s):
        return f"Sqrt({s(self.scalar)})"

    def op(self, scalar):
        return math.sqrt(scalar)

    def dop(self, scalar, dscalar, result):
        return (result * dscalar) / (2 * scalar)

def sqrt(obj):
    if isinstance(obj, Scalar):
        return Sqrt(obj)
    elif isinstance(obj, Compound):
        return obj.distribute(sqrt)
    else:
        return math.sqrt(float(obj))

@dataclass(eq=False)
class Abs(Unary):
    def stringify(self, s):
        return f"Abs({s(self.scalar)})"

    def op(self, scalar):
        return abs(scalar)

    def dop(self, scalar, dscalar, result):
        if isinstance(scalar, Expr):
            raise NotImplemented
        if scalar < 0.0:
            return -dscalar
        else:
            return dscalar

@dataclass(eq=False)
class Acos(Unary):
    def stringify(self, s):
        return f"Acos({s(self.scalar)})"

    def op(self, scalar):
        return math.acos(scalar)

    def dop(self, scalar, dscalar, result):
        return -dscalar / sqrt(1 - sqr(scalar))

@dataclass(eq=False)
class Cos(Unary):
    def stringify(self, s):
        return f"Cos({s(self.scalar)})"

    def op(self, scalar):
        return math.cos(scalar)

    def dop(self, scalar, dscalar, result):
        return -math.sin(scalar) * dscalar

@dataclass(eq=False)
class Binary(Scalar):
    lhs : Scalar
    rhs : Scalar
    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        if isinstance(lhs, Dual) or isinstance(rhs, Dual):
            lhs = lhs if isinstance(lhs, Dual) else Dual(lhs, {})
            rhs = rhs if isinstance(rhs, Dual) else Dual(rhs, {})
            result = self.op(lhs.scalar, rhs.scalar)
            partials = {}
            for sym in set(lhs.partials) | set(rhs.partials):
                dlhs = lhs.partials.get(sym, 0.0)
                drhs = rhs.partials.get(sym, 0.0)
                partials[sym] = self.dop(lhs.scalar, rhs.scalar, dlhs, drhs, result)
            return Dual(result, partials)
        elif isinstance(lhs, Expr) or isinstance(rhs, Expr):
            return type(self)(lhs, rhs)
        else:
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

    def dop(self, lhs, rhs, dlhs, drhs, result):
        return dlhs + drhs

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

    def dop(self, lhs, rhs, dlhs, drhs, result):
        return dlhs*rhs + lhs*drhs

def convert(obj):
    if isinstance(obj, Expr):
        return obj
    else:
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
    if not isinstance(expr, Expr):
        return str(expr)
    elif precedence < expr.precedence:
        return str(expr)
    else:
        return "(" + str(expr) + ")"

def validate(obj):
    for f in fields(obj):
        a = getattr(obj, f.name)
        if f.type == float:
            setattr(obj, f.name, float(a))
        elif f.type == Scalar:
            setattr(obj, f.name, convert(a))
        a = getattr(obj, f.name)
        if isinstance(f.type, type):
            assert isinstance(a, f.type), f".{f.name} = {a} ? {f.type.__name__}"

@dataclass(eq=False)
class NonZero(Expr):
    scalar : Scalar
    def stringify(self, s):
        return f"NonZero({s(self.scalar)})"

    def evaluate(self, context):
        scalar = context.compute(self.scalar)
        if isinstance(scalar, Expr):
            return NonZero(scalar)
        elif scalar >= 1e-12:
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
        if isinstance(lhs, Expr) or isinstance(rhs, Expr):
            return Eq(lhs, rhs)
        return lhs - rhs

@dataclass(eq=False)
class SoftEq(Relation):
    weight : float

    def stringify(self, s):
        return f"Eq({s(self.lhs)}, {s(self.rhs)}).soft({self.weight})"

    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        if isinstance(lhs, Expr) or isinstance(rhs, Expr):
            return SoftEq(lhs, rhs, self.weight)
        return lhs - rhs

@dataclass(eq=False)
class Entity(Expr):
    def __post_init__(self):
        super().__post_init__()
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
    variables : Dict[Symbol, float | Scalar]
    memo      : Dict[Scalar, float | Scalar]
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

    def abstract(self, value):
        self.variables[sym := Symbol()] = value
        return sym

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
        if not isinstance(expr, Expr):
            return str(expr)
        elif expr in names:
            return names[expr]
        elif precedence < expr.precedence:
            return expr.stringify(system_repr)
        else:
            return "(" + expr.stringify(system_repr) + ")"
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

@dataclass(eq=False)
class Dual(Scalar):
    scalar   : float | Scalar
    partials : Dict[Symbol, float | Scalar]

# TODO: evaluate & subexpressions

    def stringify(self, s):
        components = []
        if self.scalar != 0.0:
            components.append(s(self.scalar))
        for sym, val in self.partials.items():
            if val == 0.0:
                continue
            if val == 1.0:
                components.append(f"dual({s(sym)})")
            else:
                components.append(f"dual({s(sym)})*{s(val)}")
        return " + ".join(components)

def dual(symbol, value=0.0):
    return Dual(value, {symbol: 1.0})
