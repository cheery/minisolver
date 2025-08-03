from collections import Counter, defaultdict
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
        return distribute(self, other, add)

    def __radd__(self, other):
        other = convert(other)
        return distribute(other, self, add)

    def __sub__(self, other):
        other = convert(other)
        return distribute(self, other, lambda lhs, rhs: add(lhs, neg(rhs)))

    def __rsub__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: add(lhs, neg(rhs)))

    def __mul__(self, other):
        other = convert(other)
        return distribute(self, other, mul)

    def __rmul__(self, other):
        other = convert(other)
        return distribute(other, self, mul)

    def __truediv__(self, other):
        other = convert(other)
        return distribute(self, other, lambda lhs, rhs: mul(lhs, inv(rhs)))

    def __rtruediv__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: mul(lhs, inv(rhs)))

    def evaluate(self, context):
        attrs = {}
        for f in fields(self):
            attrs[f.name] = context.compute(getattr(self, f.name))
        return type(self)(**attrs)

    def subexpressions(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, Expr):
                yield a

@dataclass(eq=False)
class Scalar(Expr):
    def __neg__(self):
        return neg(self)

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
class PowN(Unary):
    power : float
    precedence = 20
    def stringify(self, s):
        if self.power < 0:
            return f"1 / {s(PowN(self.scalar, -self.power), 19)}"
        elif self.power == 1:
            return f"{s(self.scalar, 20)}"
        else:
            return f"{s(self.scalar, 20)}**{self.power}"

    def op(self, scalar):
        if self.power < 0 and scalar < 1e-12:
            raise Unsatisfiable
        return scalar ** self.power

    def dop(self, scalar, dscalar, result):
        if self.power <= 0 and scalar < 1e-12:
            raise Nondifferentiable
        return dscalar * self.power * (scalar ** (self.power-1))

def inv(obj):
    if isinstance(obj, Expr):
        return PowN(obj, -1)
    elif float(obj) < 1e-12:
        return PowN(Product(0, []), -1)
    else:
        return 1.0 / float(obj)

def sqr(obj):
    if isinstance(obj, Expr):
        return PowN(obj, 2)
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
class Sum(Scalar):
    terms : List[Scalar]
    precedence = 10

    def evaluate(self, context):
        terms = [context.compute(term) for term in self.terms]
        if any(isinstance(term, Dual) for term in terms):
            result = []
            partials = defaultdict(float)
            for term in terms:
                if isinstance(term, Dual):
                    result.append(term.scalar)
                    for sym, value in term.partials.items():
                        partials[sym] += value
                else:
                    result.append(term)
            return Dual(sum(result), partials)
        else:
            return sum(terms)

    def subexpressions(self):
        yield from self.terms

    def stringify(self, s):
        out = []
        for term in self.terms:
            if isinstance(term, Product) and term.constant < 0.0:
                if out:
                    out.append(" - ")
                else:
                    out.append("-")
                out.append(term.stringify_factors(s))
            else:
                if out:
                    out.append(" + ")
                out.append(s(term, 10))
        return "".join(out)

@dataclass(eq=False)
class Product(Scalar):
    constant : float
    factors  : List[Scalar]
    precedence = 20
    def __float__(self):
        if self.factors:
            raise TypeError
        return self.constant

    def evaluate(self, context):
        factors = [context.compute(factor) for factor in self.factors]
        if any(isinstance(factor, Dual) for factor in factors):
            result = self.constant
            factors = []
            pfactors = defaultdict(list)
            for i, factor in enumerate(factors):
                if isinstance(factor, Dual):
                    result *= factor.scalar
                    factors.append(factor.scalar)
                    for sym, value in factor.partials.items():
                        pfactors[sym].append((i, value))
                else:
                    result *= factor
                    factors.append(factor)
            partials = {}
            for sym, xs in pfactors.items():
                partials[sym] = product_derivative(factors, xs)
            return Dual(result, partials)
        else:
            result = self.constant
            for factor in factors:
                result *= factor
            return result

    def subexpressions(self):
        yield from self.factors

    def stringify(self, s):
        prefix = "-" if self.constant < 0.0 else ""
        return prefix + self.stringify_factors(s)

    def stringify_factors(self, s):
        out = []
        if abs(self.constant) != 1:
            out.append(str(abs(self.constant)))
        for factor in self.factors:
            out.append(s(factor, 20))
        if not out:
            out.append("1.0")
        return "*".join(out)

def product_derivative(xs, p):
    total = 0.0
    for i,x in p:
        for j in range(len(xs)):
            if i != j:
                x *= xs[j]
        total += x
    return total

def add(lhs, rhs):
    terms = []
    if isinstance(lhs, Sum):
        terms.extend(lhs.terms)
    else:
        terms.append(lhs)
    if isinstance(rhs, Sum):
        terms.extend(rhs.terms)
    else:
        terms.append(rhs)
    constant = 0
    for term in terms[:]:
        if isinstance(term, Product) and not term.factors:
            constant += term.constant
            terms.remove(term)
    if constant != 0.0:
        terms.append(Product(constant, []))
    return Sum(terms)

def neg(term):
    if isinstance(term, Sum):
        return Sum(neg(t) for t in term.terms)
    elif isinstance(term, Product):
        return Product(-term.constant, term.factors)
    else:
        return Product(-1, [term])

def mul(lhs, rhs):
    constant = 1.0
    factors  = []
    if isinstance(lhs, Product):
        constant *= lhs.constant
        factors.extend(lhs.factors)
    else:
        factors.append(lhs)
    if isinstance(rhs, Product):
        constant *= rhs.constant
        factors.extend(rhs.factors)
    else:
        factors.append(rhs)
    return Product(constant, factors)

def convert(obj):
    if isinstance(obj, Expr):
        return obj
    else:
        return Product(float(obj), [])

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
        return self.distribute(neg)

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
            for a in expr.subexpressions():
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
            for a in expr.subexpressions():
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
            for a in expr.subexpressions():
                if not (isinstance(a, Product) and not a.factors):
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

zero = Product(0.0, [])
one  = Product(1.0, [])

@dataclass(eq=False)
class Dual(Scalar):
    scalar   : float | Scalar
    partials : Dict[Symbol, float | Scalar]

    def evaluate(self, context):
        scalar = context.compute(self.scalar)
        partials = {}
        for sym, value in self.partials.items():
            partials[sym] = context.compute(value)
        return Dual(scalar, partials)

    def subexpressions(self):
        if isinstance(self.scalar, Expr):
            yield self.scalar
        for sym, value in self.partials.items():
            yield sym
            if isinstance(value, Expr):
                yield value

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
