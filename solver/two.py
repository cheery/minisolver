from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from .expressions import *
#from .common import expr, scalar, variable, constraint, zero, one, abs, acos, prev, sqr, sqrt
import numpy as np
import math

@dataclass(eq=False)
class Vector(Compound):
    x : Scalar
    y : Scalar

    def compound(self, other, operand):
        assert isinstance(other, Vector)
        return Vector(
            operand(self.x, other.x),
            operand(self.y, other.y))

    def distribute(self, operand):
        return Vector(
            operand(self.x),
            operand(self.y))

    def stringify(self, s):
        return f"two.Vector({s(self.x)}, {s(self.y)})"

    def __matmul__(self, other):
        assert isinstance(other, Vector)
        return dot(self, other)
        
    def __rmatmul__(self, other):
        assert isinstance(other, Vector)
        return dot(other, self)

def dot(a, b):
    return a.x*b.x + a.y*b.y

def mag(v):
    return Sqrt(Sqr(v.x) + Sqr(v.y))

@dataclass(eq=False)
class Point(Vector):
    def stringify(self, s):
        return f"two.Point({s(self.x)}, {s(self.y)})"

@dataclass(eq=False)
class Normal(Vector):
    def constraints(self):
        yield Eq(self @ self, one)

    def stringify(self, s):
        return f"two.Normal({s(self.x)}, {s(self.y)})"

@dataclass(eq=False)
class Line(Entity):
    vector : Vector
    scalar : Scalar

    def stringify(self, s):
        return f"two.Line({s(self.normal)}, {s(self.distance)})"

def point_line_distance(point, vector, scalar):
    mag = np.linalg.norm(vector)
    if mag >= 1e-12:
        return abs(point @ vector + scalar) / mag
    return 0.0

@dataclass(eq=False)
class Drag(Entity):
    a : Point
    b : Point

    def constraints(self):
        yield Eq(mag(self.a - self.b), zero).soft(100)

@dataclass(eq=False)
class Coincident(Entity):
    a : Point
    t : Line

    def constraints(self):
        vector = self.t.vector
        scalar = self.t.scalar
        yield NonZero(vector @ vector)
        yield Eq(self.a @ vector, -scalar)

@dataclass(eq=False)
class Distance(Entity):
    d : Scalar
    a : Point
    b : Point
    along : Vector

    def constraints(self):
        n = self.along / mag(self.along)
        yield Eq(Abs(n @ self.a - n @ self.b), self.d)

@dataclass(eq=False)
class Phi(Entity):
    a : Scalar
    n : Vector
    m : Vector

    def constraints(self):
        mag_n = mag(self.n)
        mag_m = mag(self.m)
        yield Eq(Acos(dot(self.n / mag_n, self.m / mag_m)), self.a)
        yield Eq(mag_n, Previous(mag_n)).soft(0.1)
        yield Eq(mag_m, Previous(mag_m)).soft(0.1)

def angle_of(n, m):
    magn = np.linalg.norm(n)
    magm = np.linalg.norm(m)
    if magn + magm >= 1e-12:
        d = np.clip((n / magn) @ (m / magm), -1, +1)
        return np.acos(d)
    else:
        return 0
