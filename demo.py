from solver.expressions import *
from solver.solve import setup, approximate_jacobian, analyze, solve_soft
from solver import two
import pygame
import numpy as np
import math
import weakref

last_call = None

def solve_constraints(dragging=None):
    global last_call
    if not (last_call and last_call[2] == dragging):
        entities = list(all_entities(sketch))
        knownvec = np.zeros(2)
        mx = Symbol()
        my = Symbol()
        known = {mx: 0, my: 1}
        if isinstance(dragging, two.Point):
            p = two.Point(mx, my)
            entities.append(two.Drag(dragging, p))
        elif isinstance(dragging, two.Line):
            p = two.Point(mx, my)
            entities.append(two.SoftCoincident(p, dragging))
        last_call = setup(entities, context, known, knownvec), knownvec, dragging

    last_call[1][:] = pygame.mouse.get_pos()
    f, jac, g, soft_jac, g_w, x0, wrap = last_call[0]
    try:
        sol = solve_soft(f, jac, g, soft_jac, g_w, x0)
        ctx = wrap(sol)
        for var in ctx.variables:
            context.variables[var] = ctx[var]
        context.memo.clear()
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

context = JustContext(None, weakref.WeakKeyDictionary(), {})
sketch  = []

def erase_derived(entity):
    for yy in sketch[:]:
        if entity in all_entities([yy]):
            sketch.remove(yy)

def line_rect_intersection(orient, distance, rect):
    A, B = orient
    C = distance
    A = float(A)
    B = float(B)
    C = float(C)
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
                sketch.append(two.Point(*ev.pos))
            elif ev.button == 1:
                x = context.abstract(ev.pos[0])
                y = context.abstract(ev.pos[1])
                sketch.append(two.Point(x, y))
            elif ev.button == 2:
                x = context.abstract(1)
                y = context.abstract(0)
                z = context.abstract(-ev.pos[0])
                sketch.append(two.Line(two.Normal(x, y), z))
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
                    if isinstance(p, two.Line):
                        x = float(context.compute(p.vector.x))
                        y = float(context.compute(p.vector.y))
                        d = float(context.compute(p.scalar))
                        return float(two.point_line_distance(m, np.array((x,y)), d) * 2.0)
                    elif isinstance(p, two.Point):
                        x = float(context.compute(p.x))
                        y = float(context.compute(p.y))
                        return np.linalg.norm((x,y) - m) * 1.0
                    return np.inf
                highlight = min(sketch, key=dfn)
            else:
                highlight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_a:
            blight = alight
            alight = highlight
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
            if isinstance(highlight, two.Line) and isinstance(alight, two.Point):
                sketch.append(two.Coincident(alight, highlight))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_w:
            if isinstance(highlight, two.Point) and isinstance(alight, two.Point):
                if isinstance(blight, two.Line):
                    along = blight.vector
                else:
                    along = highlight - alight
                sketch.append(two.Distance(100, highlight, alight, along))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_e:
            if isinstance(highlight, two.Point) and isinstance(alight, two.Point) and isinstance(blight, two.Point):
                n = highlight - alight
                m = highlight - blight
                #s = constant(value=two.angle_of(vari[n], vari[m]))
                sketch.append(two.Phi(math.pi * 0.25, n, m))
                highlight = alight = blight = None
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
            last_call = None
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
        if isinstance(entity, two.Point):
            x = float(context.compute(entity.x))
            y = float(context.compute(entity.y))
            pygame.draw.circle(screen, color, (x,y), 2)
        if isinstance(entity, two.Line):
            line = context.compute(entity)
            x = float(line.vector.x)
            y = float(line.vector.y)
            d = float(line.scalar)
            s = line_rect_intersection((x,y), d, screen.get_rect())
            if len(s) == 2:
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
