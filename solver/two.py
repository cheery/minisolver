from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from .expressions import *
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
    return sqrt(v.x**2 + v.y**2)

@dataclass(eq=False)
class Point(Vector):
    def stringify(self, s):
        return f"two.Point({s(self.x)}, {s(self.y)})"

@dataclass(eq=False)
class Normal(Vector):
    def constraints(self):
        yield eq(self @ self, one)

    def stringify(self, s):
        return f"two.Normal({s(self.x)}, {s(self.y)})"

@dataclass(eq=False)
class Line(Entity):
    vector : Vector
    scalar : Scalar

    def stringify(self, s):
        return f"two.Line({s(self.vector)}, {s(self.scalar)})"

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
        yield eq(mag(self.a - self.b), zero).soft(100)

    def stringify(self, s):
        return f"two.Drag({s(self.a)}, {s(self.b)})"

@dataclass(eq=False)
class Coincident(Entity):
    a : Point
    t : Line

    def constraints(self):
        vector = self.t.vector
        scalar = self.t.scalar
        #yield NonZero(vector @ vector)
        yield eq(self.a @ vector, -scalar)

    def stringify(self, s):
        return f"two.Coincident({s(self.a)}, {s(self.t)})"

@dataclass(eq=False)
class SoftCoincident(Entity):
    a : Point
    t : Line

    def constraints(self):
        vector = self.t.vector
        scalar = self.t.scalar
        #yield NonZero(vector @ vector)
        yield eq(self.a @ vector, -scalar).soft(100.0)

    def stringify(self, s):
        return f"two.SoftCoincident({s(self.a)}, {s(self.t)})"

@dataclass(eq=False)
class Distance(Entity):
    d : Scalar
    a : Point
    b : Point
    along : Vector

    def constraints(self):
        yield eq((self.along @ (self.a - self.b))**2 / (self.along @ self.along), self.d**2)

    def stringify(self, s):
        return f"two.Distance({s(self.d)}, {s(self.a)}, {s(self.b)}, {s(self.along)})"

@dataclass(eq=False)
class Phi(Entity):
    a : Scalar
    n : Vector
    m : Vector

    def constraints(self):
        n2 = self.n@self.n
        m2 = self.m@self.m
        yield eq(acos(dot(self.n, self.m) / (sqrt(n2) * sqrt(m2))), self.a)
        yield eq(sqrt(n2), Previous(sqrt(n2))).soft(0.01)
        yield eq(sqrt(m2), Previous(sqrt(m2))).soft(0.01)

    def stringify(self, s):
        return f"two.Phi({s(self.a)}, {s(self.n)}, {s(self.m)})"

def angle_of(n, m):
    magn = np.linalg.norm(n)
    magm = np.linalg.norm(m)
    if magn + magm >= 1e-12:
        d = np.clip((n / magn) @ (m / magm), -1, +1)
        return np.acos(d)
    else:
        return 0
