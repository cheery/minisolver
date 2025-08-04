from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from fractions import Fraction
import numpy as np
import math

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
        return distribute(self, other, lambda lhs, rhs: add(lhs, -rhs))

    def __rsub__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: add(lhs, -rhs))

    def __mul__(self, other):
        other = convert(other)
        return distribute(self, other, mul)

    def __rmul__(self, other):
        other = convert(other)
        return distribute(other, self, mul)

    def __truediv__(self, other):
        other = convert(other)
        return distribute(self, other, lambda lhs, rhs: mul(lhs, rhs**-1))

    def __rtruediv__(self, other):
        other = convert(other)
        return distribute(other, self, lambda lhs, rhs: mul(lhs, rhs**-1))

    def __pow__(self, other):
        other = convert(other)
        return distribute(self, other, power)

    def __rpow__(self, other):
        other = convert(other)
        return distribute(other, self, power)

    def apply(self, fn):
        attrs = {}
        for f in fields(self):
            a = getattr(self, f.name)
            attrs[f.name] = fn(a) if isinstance(a, Expr) else a
        return type(self)(**attrs)

    def evaluate(self, context):
        return self.apply(context.compute)

    def subexpressions(self):
        for f in fields(self):
            a = getattr(self, f.name)
            if isinstance(a, Expr):
                yield a

    def diff(self, memo, derive=None):
        return self.apply(lambda x: x.diff(memo, derive))

@dataclass(eq=False)
class Scalar(Expr):
    def __neg__(self):
        return neg(self)

    def __hash__(self):
        return hash((type(self),) + tuple(self.subexpressions()))

    def __eq__(self, other):
        if isinstance(other, Scalar) and self.kind == other.kind:
            return self._eq(other)
        return False

    def __lt__(self, other):
        if isinstance(other, Scalar):
            if self.kind < other.kind:
                return True
            if self.kind > other.kind:
                return False
            return self._lt(other)
        else:
            return super().__lt__(other)

@dataclass(eq=False)
class Floating(Scalar):
    value : float
    kind = 0
    def stringify(self, s):
        return str(self.value)

    def diff(self, memo, derive=None):
        return Dual(self, {})

    def evaluate(self, context):
        return self

    def __float__(self):
        return self.value

    def __hash__(self):
        return hash(float(self))

    def _eq(self, other):
        return float(self) == float(other)

    def _lt(self, other):
        return float(self) < float(other)

@dataclass(eq=False)
class Rational(Scalar):
    value : Fraction
    kind = 0
    def stringify(self, s):
        return str(self.value)

    def diff(self, memo, derive=None):
        return Dual(self, {})

    def evaluate(self, context):
        return self

    def __float__(self):
        return float(self.value)

    def __hash__(self):
        return hash(float(self))

    def _eq(self, other):
        return float(self) == float(other)

    def _lt(self, other):
        return float(self) < float(other)

@dataclass(eq=False)
class Product(Scalar):
    factors : List[Scalar]
    kind = 1
    precedence = 20

    def diff(self, memo, derive=None):
        if self in memo:
            return memo[self]
        in_factors = [factor.diff(memo, derive) for factor in self.factors]
        result = one
        factors = []
        pfactors = defaultdict(list)
        for i, factor in enumerate(in_factors):
            result *= factor.scalar
            factors.append(factor.scalar)
            for sym, value in factor.partials.items():
                pfactors[sym].append((i, value))
        partials = {}
        for sym, xs in pfactors.items():
            partials[sym] = product_derivative(factors, xs)
        memo[self] = result = Dual(result, partials)
        return result

    def evaluate(self, context):
        factors = [context.compute(factor) for factor in self.factors]
        result = one
        for factor in factors:
            result *= factor
        return result

    def apply(self, fn):
        return Product([fn(x) for x in self.factors])

    def subexpressions(self):
        yield from self.factors

    def stringify(self, s):
        return self.stringify_factors(s)

    def stringify_factors(self, s):
        out = []
        for factor in self.factors:
            out.append(s(factor, 20))
        if not out:
            out.append("1")
        return "*".join(out)

    def _eq(self, other):
        return self.factors == other.factors

    def _lt(self, other):
        m = len(self.factors)
        n = len(other.factors)
        for i in range(1, min(m, n)+1):
            if self.factors[m-i] < other.factors[n-i]:
                return True
            if self.factors[m-i] > other.factors[n-i]:
                return False
        return m < n

def product_derivative(xs, p):
    total = 0.0
    for i,x in p:
        for j in range(len(xs)):
            if i != j:
                x *= xs[j]
        total += x
    return total

@dataclass(eq=False)
class Power(Scalar):
    lhs : Scalar
    rhs : Scalar
    kind = 2
    precedence = 30
    def stringify(self, s):
        return f"{s(self.lhs, 30)}**{s(self.rhs, 30)}"

    def diff(self, memo, derive=None):
        if self in memo:
            return memo[self]
        lhs = self.lhs.diff(memo, derive)
        rhs = self.rhs.diff(memo, derive)
        scalar = lhs.scalar ** rhs.scalar
        ln_lhs = ln(lhs.scalar)
        partials = {}
        for sym in set(lhs.partials) | set(rhs.partials):
            x = lhs.partials.get(sym, zero)
            y = rhs.partials.get(sym, zero)
            partials[sym] = scalar * (y * ln_lhs + rhs.scalar * x / lhs.scalar)
        memo[self] = result = Dual(scalar, partials)
        return result

    def evaluate(self, context):
        lhs = context.compute(self.lhs)
        rhs = context.compute(self.rhs)
        return lhs ** rhs

    def _eq(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs

    def _lt(self, other):
        if self.lhs < other.lhs:
            return True
        elif self.lhs == other.lhs and self.rhs < other.rhs:
            return True
        else:
            return False

@dataclass(eq=False)
class Sum(Scalar):
    terms : List[Scalar]
    kind = 3
    precedence = 10

    def diff(self, memo, derive=None):
        if self in memo:
            return memo[self]
        terms = [term.diff(memo, derive) for term in self.terms]
        result = []
        partials = defaultdict(float)
        for term in terms:
            result.append(term.scalar)
            for sym, value in term.partials.items():
                partials[sym] += value
        memo[self] = result = Dual(sum(result), dict(partials.items()))
        return result

    def evaluate(self, context):
        return sum(context.compute(term) for term in self.terms)

    def apply(self, fn):
        return Sum([fn(x) for x in self.terms])

    def subexpressions(self):
        yield from self.terms

    def stringify(self, s):
        out = []
        for term in self.terms:
            out.append(s(term, 10))
        return " + ".join(out)

    def _eq(self, other):
        return self.terms == other.terms

    def _lt(self, other):
        m = len(self.terms)
        n = len(other.terms)
        for i in range(1, min(m, n)+1):
            if self.terms[m-i] < other.terms[n-i]:
                return True
            if self.terms[m-i] > other.terms[n-i]:
                return False
        return m < n
 
@dataclass(eq=False)
class Function(Scalar):
    kind = 4
    def evaluate(self, context):
        return self.op(*[context.compute(x) for x in self.subexpressions()])

    def stringify(self, s):
        args = [s(expr) for expr in self.subexpressions()]
        return self.name + "(" + "".join(args) + ")"

    def _eq(self, other):
        return type(self) == type(other) and tuple(self.subexpressions()) == tuple(other.subexpressions())

    def _lt(self, other):
        if self.name < other.name:
            return True
        if id(type(self)) < id(type(other)):
            return True
        return type(self) == type(other) and tuple(self.subexpressions()) < tuple(other.subexpressions())

@dataclass(eq=False)
class UnaryFunction(Function):
    scalar : Scalar

    def diff(self, memo, derive=None):
        if self in memo:
            return memo[self]
        x = self.scalar.diff(memo, derive)
        result = self.op(x.scalar)
        partials = {}
        for sym, dx in x.partials.items():
            partials[sym] = self.dop(x.scalar, dx, result)
        memo[self] = result = Dual(result, partials)
        return result

@dataclass(eq=False)
class Symbol(Scalar):
    kind = 5
    def stringify(self, s):
        return f"Symbol()"

    def diff(self, memo, derive=None):
        if self in memo:
            return memo[self]
        if derive is None or self in derive:
            memo[self] = result = dual(self, self)
        else:
            memo[self] = result = self
        return result

    def evaluate(self, context):
        try:
            return context[self]
        except KeyError:
            return self

    def _eq(self, other):
        return id(self) == id(other)

    def _lt(self, other):
        return id(self) < id(other)

def base(u):
    if isinstance(u, Power):
        return u.lhs
    else:
        return u

def exponent(u):
    if isinstance(u, Power):
        return u.rhs
    else:
        return one

def term(u):
    if isinstance(u, (Symbol, Sum, Power, Function)):
        return u
    elif isinstance(u, Product):
        if isinstance(u.factors[0], (Rational,Floating)):
            if len(u.factors) == 2:
                return u.factors[1]
            else:
                return Product(u.factors[1:])
        else:
            return u
    else:
        return undef

def const(u):
    if isinstance(u, (Symbol, Sum, Power, Function)):
        return one
    elif isinstance(u, Product):
        if isinstance(u.factors[0], (Rational,Floating)):
            return u.factors[0]
        else:
            return one
    else:
        return undef

def power(v, w):
    if v == undef or w == undef:
        return undef
    elif v == zero:
        if isinstance(w, (Rational,Floating)) and float(w) > 0:
            return zero
        else:
            return undef # TODO: This is probably not correct.
    elif v == one:
        return one
    elif isinstance(w, Rational) and w.value.denominator == 1:
        return power_integer(v, w)
    else:
        return Power(v, w)

def power_integer(v, n):
    if isinstance(v, Rational):
        return Rational(v.value ** n.value)
    elif isinstance(v, Floating):
        return Floating(v.value ** n.value)
    elif n.value == 0:
        return one
    elif n.value == 1:
        return v
    elif isinstance(v, Power):
        r = v.lhs
        s = v.rhs
        p = product([s, n])
        if isinstance(p, Rational):
            return power_integer(r,p)
        else:
            return Power(r, p)
    elif isinstance(v, Product):
        r = v.apply(lambda x: power(x, n))
        return product(r.factors)
    else:
        return Power(v, n)

def product(factors):
    factors.sort()
    if zero in factors:
        return zero
    if undef in factors:
        return undef
    if len(factors) == 1:
        return factors[0]
    v = product_fold(factors)
    if len(v) == 0:
        return one
    elif len(v) == 1:
        return v[0]
    else:
        return Product(v)

def product_fold(factors):
    changed = True
    while changed:
        factors.sort()
        changed = False
        i = 0
        while i+1 < len(factors):
            result = product_merge(factors[i], factors[i+1])
            if result is None:
                i += 1
            else:
                changed = True
                factors[i:i+2] = result
    return factors

def product_merge(u1, u2):
    if isinstance(u1, Product) and isinstance(u2, Product):
        return u1.factors + u2.factors
    elif isinstance(u1, Product):
        return u1.factors + [u2]
    elif isinstance(u2, Product):
        return [u1] + u2.factors
    elif isinstance(u1, Rational) and isinstance(u2, Rational):
        rat = Rational(u1.value * u2.value)
        if rat == one:
            return []
        else:
            return [rat]
    elif isinstance(u1, (Rational, Floating)) and isinstance(u2, (Rational,Floating)):
        flo = Floating(u1.value * u2.value)
        if flo == one:
            return []
        else:
            return [flo]
    elif u1 == one:
        return [u2]
    elif u2 == one:
        return [u1]
    elif base(u1) == base(u2):
        s = sum([exponent(u1), exponent(u2)])
        p = power(base(u1), s)
        if p == one:
            return []
        else:
            return [p]
    else:
        return None

def summation(terms):
    terms.sort()
    if undef in terms:
        return undef
    if len(terms) == 1:
        return terms[0]
    v = summation_fold(terms)
    if len(v) == 0:
        return zero
    elif len(v) == 1:
        return v[0]
    else:
        return Sum(v)

def summation_fold(terms):
    changed = True
    while changed:
        terms.sort()
        changed = False
        i = 0
        while i+1 < len(terms):
            result = summation_merge(terms[i], terms[i+1])
            if result is None:
                i += 1
            else:
                changed = True
                terms[i:i+2] = result
    return terms

def summation_merge(u1, u2):
    if isinstance(u1, Sum) and isinstance(u2, Sum):
        return u1.terms + u2.terms
    elif isinstance(u1, Sum):
        return u1.terms + [u2]
    elif isinstance(u2, Sum):
        return [u1] + u2.terms
    elif isinstance(u1, Rational) and isinstance(u2, Rational):
        rat = Rational(u1.value + u2.value)
        if rat == zero:
            return []
        else:
            return [rat]
    elif isinstance(u1, (Rational, Floating)) and isinstance(u2, (Rational,Floating)):
        flo = Floating(u1.value + u2.value)
        if flo == zero:
            return []
        else:
            return [flo]
    elif u1 == zero:
        return [u2]
    elif u2 == zero:
        return [u1]
    elif term(u1) == term(u2):
        s = summation([const(u1), const(u2)])
        p = product([s, term(u1)])
        if p == zero:
            return []
        else:
            return [p]
    else:
        return None

@dataclass(eq=False)
class Previous(Function):
    name = "Previous"
    scalar : Scalar

    def diff(self, memo, derive=None):
        raise RuntimeError

    def evaluate(self, context):
        if context.previous is not None:
            expr = context.previous.compute(self.scalar)
            if isinstance(expr, (Floating, Rational)):
                return expr
            else:
                return Previous(expr)
        else:
            return self

def sqrt(obj):
    if isinstance(obj, Compound):
        return obj.distribute(sqrt)
    else:
        return Sqrt.op(convert(obj))

@dataclass(eq=False)
class Sqrt(UnaryFunction):
    name = "sqrt"

    @classmethod
    def dop(cls, x, dx, y):
        return (y * dx) / (2 * x)

    @classmethod
    def op(cls, scalar):
        if isinstance(scalar, (Floating, Rational)):
            return Floating(math.sqrt(scalar.value))
        else:
            return cls(scalar)

def xabs(obj):
    if isinstance(obj, Compound):
        return obj.distribute(abs)
    else:
        return Abs.op(convert(obj))

@dataclass(eq=False)
class Abs(UnaryFunction):
    name = "xabs"

    @classmethod
    def dop(cls, x, dx, y):
        return sign(x) * dx

    @classmethod
    def op(cls, scalar):
        if isinstance(scalar, (Floating, Rational)):
            return Floating(abs(scalar.value))
        else:
            return cls(scalar)

def acos(obj):
    if isinstance(obj, Compound):
        return obj.distribute(xabs)
    else:
        return Acos.op(convert(obj))

@dataclass(eq=False)
class Acos(UnaryFunction):
    name = "acos"

    def dop(self, x, dx, y):
        return -dx / sqrt(1 - x**2)

    @classmethod
    def op(cls, scalar):
        if isinstance(scalar, (Floating, Rational)):
            return Floating(math.acos(scalar.value))
        else:
            return cls(scalar)

def ln(obj):
    if isinstance(obj, Compound):
        return obj.distribute(ln)
    else:
        return Ln.op(convert(obj))

@dataclass(eq=False)
class Ln(UnaryFunction):
    name = "ln"

    def dop(self, x, dx, y):
        return dx / x

    @classmethod
    def op(cls, scalar):
        if isinstance(scalar, (Floating, Rational)):
            return Floating(math.ln(scalar.value))
        else:
            return cls(scalar)

def sign(obj):
    if isinstance(obj, Compound):
        return obj.distribute(sign)
    else:
        return Sign.op(convert(obj))

@dataclass(eq=False)
class Sign(UnaryFunction):
    name = "sign"

    @classmethod
    def dop(cls, x, dx, y):
        if x == zero:
            return 2*x
        return zero

    @classmethod
    def op(cls, scalar):
        if isinstance(scalar, (Floating, Rational)):
            return one if float(scalar) >= 0 else -one
        else:
            return cls(scalar)

def add(lhs, rhs):
    return summation([lhs, rhs])

def neg(term):
    return product([Rational(Fraction(-1)), term])

def mul(lhs, rhs):
    return product([lhs, rhs])

def convert(obj):
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, int):
        return Rational(Fraction(obj))
    elif isinstance(obj, Fraction):
        return Rational(obj)
    elif isinstance(obj, float):
        return Floating(obj)
    else:
        return Floating(float(obj))

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
        raise TypeError(f"distribute({lhs} : {type(lhs).__name__}, {rhs} : {type(rhs).__name__}, ...)")

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

@dataclass(eq=True)
class NonZero(Expr):
    scalar : Scalar
    def stringify(self, s):
        return f"NonZero({s(self.scalar)})"

    def evaluate(self, context):
        return NonZero(context.compute(self.scalar))

    def __hash__(self):
        return hash(self.scalar)

@dataclass(eq=True)
class Relation(Expr):
    objective : Scalar

@dataclass(eq=True)
class Eq(Relation):
    def soft(self, weight):
        return SoftEq(self.objective, weight)

    def stringify(self, s):
        return f"Eq({s(self.objective)})"

    def evaluate(self, context):
        return Eq(context.compute(self.objective))

    def __hash__(self):
        return hash(self.objective)

def eq(lhs, rhs):
    return Eq(convert(lhs - rhs))

@dataclass(eq=True)
class SoftEq(Relation):
    weight : float

    def stringify(self, s):
        return f"Eq({s(self.objective)}).soft({self.weight})"

    def evaluate(self, context):
        return SoftEq(context.compute(self.objective), self.weight)

    def __hash__(self):
        return hash((self.objective, self.weight))

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

@dataclass(eq=False)
class EmptyContext(EvaluationContext):
    memo : Dict[Scalar, Scalar]

    def compute(self, scalar):
        if isinstance(scalar, Symbol):
            return scalar
        try:
            return self.memo[scalar]
        except KeyError:
            self.memo[scalar] = value = scalar.evaluate(self)
            return value

    def __getitem__(self, variable):
        return variable

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

@dataclass(eq=False)
class CellGet:
    k : int
    i : int
    def __call__(self, xs):
        return xs[self.k][self.i]

@dataclass(eq=False)
class CellSet:
    i : int
    j : int
    arg : Any
    def __call__(self, xs):
        xs[4][self.i, self.j] = v = self.arg(xs)
        return v

@dataclass(eq=False)
class CellValue:
    value : float
    def __call__(self, xs):
        return self.value

@dataclass(eq=False)
class Cell:
    op : Any
    args : List[Any]
    def __call__(self, xs):
        x = [a(xs) for a in self.args]
        return self.op(*x)

def cells(system, variables, known):
    common = {x: i for i, (x, _) in enumerate(exprs_postorder(system))}
    def build(expr, deep=True):
        if expr in known:
            return CellGet(3, known[expr])
        if isinstance(expr, Symbol):
            return CellGet(0, variables[expr])
        if deep and expr in common:
            return CellGet(2, common[expr])
        if isinstance(expr, Abs):
            return Cell(abs, [build(expr.scalar)])
        if isinstance(expr, Sqrt):
            return Cell(math.sqrt, [build(expr.scalar)])
        if isinstance(expr, Acos):
            f = lambda x: math.acos(np.clip(x, -1, +1))
            return Cell(f, [build(expr.scalar)])
        if isinstance(expr, Sign):
            f = lambda x: 1 if x >= 0 else -1
            return Cell(f, [build(expr.scalar)])
        if isinstance(expr, Sum):
            return Cell(lambda *xs: sum(xs), list(map(build, expr.terms)))
        if isinstance(expr, Product):
            return Cell(prod, list(map(build, expr.factors)))
        if isinstance(expr, Power):
            return Cell(pow, [build(expr.lhs), build(expr.rhs)])
        if isinstance(expr, (Floating, Rational)):
            return CellValue(float(expr))
        assert False, expr
    out = [build(x, False) for x in common if x not in system]
    for col, expr in enumerate(system):
        if isinstance(expr, NonZero):
            if not isinstance(expr.scalar, Dual):
                out.append(Cell(nonzero, [build(expr.scalar)]))
        if isinstance(expr, (Eq, SoftEq)):
            if isinstance(expr.objective, Dual):
                for sym, value in expr.objective.partials.items():
                    if sym in variables:
                        out.append(CellSet(col, variables[sym], build(value)))
            else:
                out.append(build(expr.objective))
    return out

def nonzero(s):
    if s < 1e-12:
        return np.inf
    else:
        return 0

def prod(*s):
    total = 1.0
    for x in s:
        total *= x
    return total

def exprs_postorder(system, include_symbols=False):
    counter = Counter()
    postorder = []
    def visit(expr):
        counter[expr] += 1
        if counter[expr] <= 1:
            for a in expr.subexpressions():
                if not isinstance(a, (Floating, Rational)):
                    visit(a)
            postorder.append(expr)
    for expr in system:
        visit(expr)
    for x in postorder:
        if include_symbols and isinstance(x, Symbol):
            yield x, True
        elif counter[x] > 1 and not isinstance(x, Symbol):
            yield x, True
    for x in system:
        yield x, False

def print_system(system):
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
    for expr, vari in exprs_postorder(system, include_symbols=True):
        if isinstance(expr, Symbol):
            names[expr] = f"v{i}"
            i += 1
        elif vari:
            print(f"  v{i} =", expr.stringify(system_repr))
            names[expr] = f"v{i}"
            i += 1
        else:
            print(f" ", expr.stringify(system_repr))

undef = Symbol()
zero  = Rational(Fraction(0))
one   = Rational(Fraction(1))

@dataclass(eq=False)
class Dual(Scalar):
    kind = -1
    scalar   : Scalar
    partials : Dict[Symbol, Scalar]

    def apply(self, fn):
        scalar = fn(self.scalar)
        partials = {}
        for sym, value in self.partials.items():
            partials[sym] = fn(value)
        return Dual(scalar, partials)

    def subexpressions(self):
        yield self.scalar
        for sym, value in self.partials.items():
            yield sym
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

    def __hash__(self):
        return id(self)

    def _eq(self, other):
        return self is other

    def _lt(self, other):
        return id(self) < id(other)

def dual(symbol, value=zero):
    return Dual(value, {symbol: one})
