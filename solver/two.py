from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any, Set, Union
from .common import expr, scalar, variable, constraint, zero, one
import numpy as np
import math

@dataclass(eq=False)
class point(expr):
    x : scalar = zero
    y : scalar = zero

    def evaluate(self, x):
        return np.array([x[self.x], x[self.y]])

@dataclass(eq=False)
class normal(expr):
    pass

@dataclass(eq=False)
class free_normal(normal):
    x : variable
    y : variable

    def evaluate(self, x):
        return np.array([x[self.x], x[self.y]])

    def hard(self, x, _):
        n = x[self]
        yield n @ n - 1

@dataclass(eq=False)
class transformed_normal(normal):
    source : normal
    matrix : np.array

    def evaluate(self, x):
        return self.matrix @ x[self.source]

@dataclass(eq=False)
class normal_between(normal):
    source : point
    target : point

    def evaluate(self, x):
        source = x[self.source]
        target = x[self.target]
        return target - source

@dataclass(eq=False)
class line(expr):
    orient   : normal
    distance : scalar

    def evaluate(self, x):
        return x[self.orient], x[self.distance]

def point_line_distance(point, orient, distance):
    mag = np.linalg.norm(orient)
    if mag >= 1e-12:
        return abs(point @ orient + distance) / mag
    return 0.0

@dataclass(eq=False)
class drag(constraint):
    a : point
    b : point

    def soft(self, x, _):
        a = x[self.a]
        b = x[self.b]
        d = a - b
        yield (d @ d) * 1000.0

@dataclass(eq=False)
class coincident(constraint):
    a : point
    t : line

    def hard(self, x, _):
        a = x[self.a]
        orient, distance = x[self.t]
        n2 = orient @ orient
        if n2 >= 1e-12:
            yield - distance - a @ orient
        else:
            yield 1

@dataclass(eq=False)
class distance(constraint):
    d : scalar
    a : point
    b : point
    along : normal
    mode : Callable[float, float] = abs

    def hard(self, x, _):
        d = x[self.d]
        a = x[self.a]
        b = x[self.b]
        n = x[self.along]
        mag = np.linalg.norm(n)
        if mag >= 1e-12:
            n = n / mag
            yield self.mode(n @ b - n @ a) - d
        else:
            yield 1

@dataclass(eq=False)
class phi(constraint):
    a : scalar
    n : normal
    m : normal

    def hard(self, x, _):
        a = x[self.a]
        n = x[self.n]
        m = x[self.m]
        magn = np.linalg.norm(n)
        magm = np.linalg.norm(m)
        if magn + magm >= 1e-12:
            d = np.clip((n / magn) @ (m / magm), -1, +1)
            yield np.acos(d) - a
        else:
            yield 1

    def soft(self, x, x0):
        n0 = x0[self.n]
        m0 = x0[self.m]
        n = x[self.n]
        m = x[self.m]
        yield np.linalg.norm(n0) - np.linalg.norm(n)
        yield np.linalg.norm(m0) - np.linalg.norm(m)

def angle_of(n, m):
    magn = np.linalg.norm(n)
    magm = np.linalg.norm(m)
    if magn + magm >= 1e-12:
        d = np.clip((n / magn) @ (m / magm), -1, +1)
        return np.acos(d)
    else:
        return 0
