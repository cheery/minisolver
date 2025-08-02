from solver.solve import approximate_jacobian, analyze, solve_soft
import pygame
import numpy as np
import math
import weakref

from solver.common import variables, setup, scalar, constant, variable, zero, one, expand, expr
from solver import two

def solve_constraints(dragging=None):
    constrs = constraints[:]
    mx, my = pygame.mouse.get_pos()
    if isinstance(dragging, two.point):
        p = two.point(constant(mx), constant(my))
        constrs.append(two.drag(dragging, p))
    elif isinstance(dragging, two.line):
        p = two.point(constant(mx), constant(my))
        q = two.point(variable(), variable())
        constrs.append(two.drag(p, q))
        constrs.append(two.coincident(q, dragging))
    f, g, g_w, x0, interp = setup(constrs, vari)
    jac = approximate_jacobian(f)
    soft_jac = approximate_jacobian(g)
    try:
        sol = solve_soft(f, jac, g, soft_jac, g_w, x0)
        variables = interp(sol)
        for var in variables.mapping:
            vari.mapping[var] = variables[var]
        vari.memo.clear()
        return analyze(jac(sol))[0] == 0
    except:
        import traceback
        traceback.print_exc()
        return False

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

pygame.init()

pygame.init()
font   = pygame.font.SysFont('Arial', 16)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock  = pygame.time.Clock()

vari   = variables(weakref.WeakKeyDictionary(), {})
sketch = []
constraints = []

def erase_derived(entity):
    for xx in constraints[:]:
        if entity in expand([xx], expr):
            constraints.remove(xx)
    for yy in sketch[:]:
        if entity in expand([yy], expr):
            sketch.remove(yy)

def line_rect_intersection(orient, distance, rect):
    A, B = orient
    C = distance
    xmin, ymin = rect.topleft
    xmax, ymax = rect.bottomright
    pts = []
    def try_add(x: float, y: float):
        if xmin - 1e-9 <= x <= xmax + 1e-9 and ymin - 1e-9 <= y <= ymax + 1e-9:
            pts.append((x, y))
    if B != 0:
        y = -(A * xmin + C) / B
        try_add(xmin, y)
    if B != 0:
        y = -(A * xmax + C) / B
        try_add(xmax, y)
    if A != 0:
        x = -(B * ymin + C) / A
        try_add(x, ymin)
    if A != 0:
        x = -(B * ymax + C) / A
        try_add(x, ymax)
    unique = []
    for p in pts:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in unique):
            unique.append(p)
    return unique

green = True
alight = None
blight = None
highlight = None
running = True
while running:
    dt = clock.tick(30) / 1000.0
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mods = pygame.key.get_mods()
            shift_held = mods & pygame.KMOD_SHIFT
            if ev.button == 1 and shift_held:
                x = constant(ev.pos[0])
                y = constant(ev.pos[1])
                sketch.append(two.point(x, y))
            elif ev.button == 1:
                x = variable()
                y = variable()
                vari.mapping[x] = ev.pos[0]
                vari.mapping[y] = ev.pos[1]
                sketch.append(two.point(x, y))
            elif ev.button == 2:
                x = variable()
                y = variable()
                z = variable()
                vari.mapping[x] = 1
                vari.mapping[y] = 0
                vari.mapping[z] = -ev.pos[0]
                n = two.free_normal(x, y)
                sketch.append(two.line(n, z))
            elif ev.button == 3 and highlight is not None:
                sketch.remove(highlight)
                erase_derived(highlight)
                highlight = alight = blight = None
        elif ev.type == pygame.MOUSEMOTION:
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE]:
                green = solve_constraints(highlight)
            elif len(sketch):
                m = np.array(ev.pos)
                def dfn(p):
                    if isinstance(p, two.line):
                        return two.point_line_distance(m, *vari[p]) * 2.0
                    elif isinstance(p, two.point):
                        return np.linalg.norm(vari[p] - m) * 1.0
                    return np.inf
                highlight = min(sketch, key=dfn)
            else:
                highlight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_a:
            blight = alight
            alight = highlight
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
            if isinstance(highlight, two.line) and isinstance(alight, two.point):
                constraints.append(two.coincident(alight, highlight))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_w:
            if isinstance(highlight, two.point) and isinstance(alight, two.point):
                d = constant(np.linalg.norm(vari[highlight] - vari[alight]))
                if isinstance(blight, two.line):
                    along = blight.orient
                else:
                    along = two.normal_between(highlight, alight)
                constraints.append(two.distance(d, highlight, alight, along))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_e:
            if isinstance(highlight, two.point) and isinstance(alight, two.point) and isinstance(blight, two.point):
                n = two.normal_between(highlight, alight)
                m = two.normal_between(highlight, blight)
                s = constant(value=two.angle_of(vari[n], vari[m]))
                constraints.append(two.phi(s, n, m))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
            green = solve_constraints(highlight)

    screen.fill((30, 30, 30))
    for entity in sketch:
        color = (255, 255, 255)
        if highlight is entity:
            color = (255, 255, 0)
        if alight is entity:
            color = (0, 255, 255)
        if blight is entity:
            color = (255, 0, 255)
        if isinstance(entity, two.point):
            pos = vari[entity]
            pygame.draw.circle(screen, color, pos, 2)
            #if entity.group == None and type(entity) == two.point:
            #    x, y = entity.pos
            #    pygame.draw.line(screen, color, (x-4, y-4), (x+4, y+4))
            #    pygame.draw.line(screen, color, (x-4, y+4), (x+4, y-4))
        #if isinstance(entity, two.point_on_line):
        #    pygame.draw.circle(screen, (100, 100, 100), entity.pos, 6, 1)
        if isinstance(entity, two.line):
            orient, distance = vari[entity]
            s = line_rect_intersection(orient, distance, screen.get_rect())
            if len(s) == 2:
                #if entity.group == None and type(entity) == two.line:
                #    pygame.draw.line(screen, color, s[0], s[1], 3)
                #else:
                pygame.draw.line(screen, color, s[0], s[1], 1)

    if green:
        pygame.draw.line(screen, (0, 255, 0), (0, 0), (SCREEN_WIDTH, 0), 5)

#    for group, flavor, amount in constraints:
#        points = tuple(sketch[i] for i in group)
#        if flavor == 'dist':
#            center = (points[0] + points[1]) / 2
#            delta = (points[0] - points[1])
#            dist = math.sqrt(np.dot(delta,delta))
#            norm = delta / dist if dist > 0 else 0
#            pygame.draw.line(screen, (200,200,200), center + norm*amount, center - norm*amount, 2)
#       
#        if flavor == 'angle':
#            if abs(angle(*points) - amount) < 1e-4:
#                pygame.draw.line(screen, (255,255,0), (points[0] + points[1])/2, points[1], 2)
#                pygame.draw.line(screen, (255,255,0), points[1], (points[1] + points[2])/2, 2)
#            else:
#                pygame.draw.line(screen, (255,0,0), (points[0] + points[1])/2, points[1], 2)
#                pygame.draw.line(screen, (255,0,0), points[1], (points[1] + points[2])/2, 2)
    pygame.display.flip()
