"""
physics_engine.py  -  GPU physics engine + state manager + State Viewer + GGUF model runner + Destruction lab (one file)
====================================================================================================
PART 1  2D rigid bodies on NVIDIA Warp.  Everything runs on the GPU (there is no CPU physics).
PART 2  The state manager: a character that perceives, remembers, believes and wonders, plus the
        State Viewer window that shows what it is thinking.
PART 3  A GGUF language-model runner (no llama.cpp): loads a .gguf file and generates text on the GPU (CuPy) or CPU.

Run:    python physics_engine.py [state.json]            open the State Viewer
        python physics_engine.py --gguf model.gguf ...   chat with a model  (--prompt, --tokens, --temp, --top-p,
                                                          --dense, --stats, --cpu, --info; see PART 3)
Import: import physics_engine as pe       (the physics, state manager and runner are all plain APIs)
Nothing here draws a scene; a UI sits on top and only calls the APIs.

PART 1 - the physics engine  (PART 4, the Destruction lab, is described in its own header:  python physics_engine.py --lab)

    import physics_engine as pe
    gpu = pe.WarpPhysics()                 # needs an NVIDIA GPU; WarpPhysics("cpu") = Warp's slow emulation (testing)
    world = pe.World()
    ground = world.add(pe.Body(pe.BOX, 450, 580, hw=450, hh=20, static=True))
    crate  = world.add(pe.Body(pe.BOX, 450, 300, hw=20, hh=20, density=0.01))
    gpu.load(world.static + world.dyn, tether=None, noglue=[], dt=pe.STEP_DT)
    for _ in range(600):
        gpu.step_bundle()                  # one call = 4 substeps = 1/60 s of simulated time
        gpu.readback()                     # -> px, py, angle, asleep, active, glue_flags (numpy)

API summary (see the class for the details)
    World / Body            plain Python description of the bodies (the GPU holds the live state)
    WarpPhysics.load        send the bodies to the GPU          .add_body / .remove_body   add or remove one live
    .step_bundle            advance 1/60 s                       .readback                  positions etc. back to Python
    .set_inputs             mouse, grab, joint strength, tether (rope), sideways acceleration (earthquake / wind),
                            gravitation=G (Newton's attraction between bodies, off by default)
    .kinetic_energy         total (and per-body) kinetic energy, 1/2 m v^2 + 1/2 I w^2
    .weld / .weld_all       join pieces with cohesive joints of a chosen strength (nail / weld / anchor)
    .add_joint / .weld_joints   anchored breakable joints (a structural graph): each joint holds two bodies at a point and
                            carries tension, shear and bending-moment limits (+ fatigue); JOINT_MATERIALS has presets
                            (nail / wood / concrete / glass / weld / steel / anchor).  A joint that overloads snaps and
                            the pieces then collide normally.   .take_broken_joints() -> break events, .joint_loads()
    Body(material=...) / .fracture_step(world)   bodies that break: wood splinters, glass shards, concrete / stone
                            chunks, brick chunks; MATERIALS has the limits.  build_wall() makes a brick wall.
    .kick                   give a body an instant velocity (explosions)
    .peak_host              hardest hit each body took since the last readback (for breaking things)
    WarpParticles           a separate GPU particle / fluid solver

Needs:  pip install warp-lang numpy      (a recent NVIDIA driver)
"""
import argparse
import heapq
import math
import os
import random
import re
import queue
import struct
import sys
import threading
import time
from collections import deque
from functools import lru_cache

try:
    import tkinter as tk          # only the State Viewer window needs this
except ImportError:
    tk = None

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is missing.  Run:  pip install numpy")
try:
    import warp as wp
except ImportError:
    sys.exit("NVIDIA Warp is missing.  Run:  pip install warp-lang")

STEP_DT = 1 / 240        # the GPU physics always steps at this fixed rate (4 steps per 1/60 s bundle)
CEMENT_FORCE = 3.0e6     # strength of one full cement contact (mass * px/s^2)
BOX, CIRCLE = 0, 1


# ####################################################################################################
# GPU ENGINE
# ####################################################################################################

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
CAP = int(os.environ.get("PE_CAP", 320))  # body slots on the GPU (static and dynamic bodies)
MAXC = int(os.environ.get("PE_MAXC", 8192))  # contact capacity
BUNDLE = 4                    # substeps recorded in one graph (4 x 1/240 s = 1/60 s)
ITERATIONS = 8                # solver sweeps per substep (each sweeps every colour)
SLEEP_PASSES = 16             # island propagation passes after each bundle

GRAVITY = wp.constant(700.0)
BETA = wp.constant(0.2)
SLOP = wp.constant(0.5)
REST_THRESHOLD = wp.constant(60.0)
MAX_SPEED = wp.constant(1200.0)
SLEEP_SPEED = wp.constant(10.0)
SLEEP_SPIN = wp.constant(0.12)
SLEEP_TIME = wp.constant(0.6)
GLUE_MARGIN = wp.constant(3.0)
CEMENT_CURE = wp.constant(3.0)
GRAB_K = wp.constant(90.0)
GRAB_C = wp.constant(16.0)
GRAB_MAX = wp.constant(6000.0)
GRAB_REACH = wp.constant(60.0)
WORLD_W = wp.constant(6000.0)
WORLD_H = wp.constant(4000.0)
KCAP = wp.constant(CAP)
KMAXC = wp.constant(MAXC)
NCOLORS = wp.constant(24)       # contact colours (max contacts a body can have in one colour class set)
COLOR_ROUNDS = 12

# layout of the small `params` array (uploaded by the host every frame)
P_CEMENT, P_GRAB_ON, P_GRAB_BODY, P_GRAB_LX, P_GRAB_LY, P_MX, P_MY = range(7)
P_TETHER, P_PIVX, P_PIVY, P_ROPE = 7, 8, 9, 10
P_AX = 11                     # sideways acceleration of the whole world (earthquake / wind)
P_G = 12                      # gravitational constant for body-to-body attraction (0 = off)
NPARAMS = 13
GRAV_SOFT = wp.constant(8.0)  # softening length (px): stops the pull blowing up when two bodies overlap

# ---- anchored joints (rigid bodies + breakable constraints, stored as an adjacency list) ----
MAXJ = int(os.environ.get("PE_MAXJ", 1024))  # joint slots
MAXDEG = 12                   # joints per body in the adjacency list (jointed pairs do not collide)
JOINT_COLORS = 32             # edge colours for the parallel joint solve
JOINT_BETA = wp.constant(0.25)        # position correction rate of a joint (Baumgarte)
JOINT_BIAS_MAX = wp.constant(600.0)   # px/s: the most a joint will try to close a gap in one step
FATIGUE_U0 = wp.constant(0.3)         # load ratio (0..1) above which a joint starts to wear
JOINT_GRACE = wp.constant(0.3)       # s after creation during which a joint cannot break (settling transient)
JOINT_WARM = 0.7   # share of last step's joint impulse re-applied at the start of a step
JOINT_SMOOTH = wp.constant(0.35)      # weight of the newest substep in the load that is checked against the limits
JOINT_EXTRA_SWEEPS = 6        # extra joint-only sweeps per substep, so load travels further along a chain
KMAXDEG = wp.constant(MAXDEG)


# ----------------------------------------------------------------------------
# narrow phase (device functions).  All return (count, nx, ny, x0, y0, d0, x1, y1, d1):
# the normal points from A to B, depth < 0 means "not quite touching yet".
# ----------------------------------------------------------------------------
@wp.func
def _circle_circle(ax: float, ay: float, ar: float, bx: float, by: float, br: float, margin: float):
    dx = bx - ax
    dy = by - ay
    rr = ar + br
    d2 = dx * dx + dy * dy
    if d2 >= (rr + margin) * (rr + margin):
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    d = wp.sqrt(d2)
    nx = float(0.0)
    ny = float(1.0)
    if d >= 1.0e-9:
        nx = dx / d
        ny = dy / d
    depth = rr - d
    px = ax + nx * (ar - 0.5 * depth)
    py = ay + ny * (ar - 0.5 * depth)
    return 1, nx, ny, px, py, depth, 0.0, 0.0, 0.0


@wp.func
def _circle_box(cx: float, cy: float, cr: float, bx: float, by: float, bc: float, bs: float,
                bhw: float, bhh: float, margin: float):
    """Normal points from the box to the circle."""
    dx = cx - bx
    dy = cy - by
    lx = dx * bc + dy * bs
    ly = -dx * bs + dy * bc
    qx = wp.max(-bhw, wp.min(bhw, lx))
    qy = wp.max(-bhh, wp.min(bhh, ly))
    ex = lx - qx
    ey = ly - qy
    d2 = ex * ex + ey * ey
    nlx = float(0.0)
    nly = float(0.0)
    depth = float(0.0)
    if d2 > 1.0e-12:
        if d2 >= (cr + margin) * (cr + margin):
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        d = wp.sqrt(d2)
        nlx = ex / d
        nly = ey / d
        depth = cr - d
    else:
        ox = bhw - wp.abs(lx)
        oy = bhh - wp.abs(ly)
        if ox < oy:
            nlx = 1.0
            if lx < 0.0:
                nlx = -1.0
            depth = cr + ox
            qx = nlx * bhw
            qy = ly
        else:
            nly = 1.0
            if ly < 0.0:
                nly = -1.0
            depth = cr + oy
            qx = lx
            qy = nly * bhh
    nx = nlx * bc - nly * bs
    ny = nlx * bs + nly * bc
    px = bx + qx * bc - qy * bs
    py = by + qx * bs + qy * bc
    return 1, nx, ny, px, py, depth, 0.0, 0.0, 0.0


@wp.func
def _clip(x1: float, y1: float, s1: float, x2: float, y2: float, s2: float, sign: float, limit: float):
    """Clip segment to sign*s <= limit. Returns (ok, x1, y1, s1, x2, y2, s2)."""
    d1 = sign * s1 - limit
    d2 = sign * s2 - limit
    if d1 <= 0.0 and d2 <= 0.0:
        return 1, x1, y1, s1, x2, y2, s2
    if d1 > 0.0 and d2 > 0.0:
        return 0, x1, y1, s1, x2, y2, s2
    t = d1 / (d1 - d2)
    qx = x1 + t * (x2 - x1)
    qy = y1 + t * (y2 - y1)
    qs = s1 + t * (s2 - s1)
    if d1 > 0.0:
        return 1, qx, qy, qs, x2, y2, s2
    return 1, x1, y1, s1, qx, qy, qs


@wp.func
def _box_box(ax: float, ay: float, ac: float, as_: float, ahw: float, ahh: float,
             bx: float, by: float, bc: float, bs: float, bhw: float, bhh: float, margin: float):
    dx = bx - ax
    dy = by - ay
    a0x = ac
    a0y = as_
    a1x = -as_
    a1y = ac
    b0x = bc
    b0y = bs
    b1x = -bs
    b1y = bc
    dA0 = dx * a0x + dy * a0y
    dA1 = dx * a1x + dy * a1y
    dB0 = dx * b0x + dy * b0y
    dB1 = dx * b1x + dy * b1y
    k00 = wp.abs(a0x * b0x + a0y * b0y)
    k01 = wp.abs(a0x * b1x + a0y * b1y)
    k10 = wp.abs(a1x * b0x + a1y * b0y)
    k11 = wp.abs(a1x * b1x + a1y * b1y)
    m = -margin
    oA0 = ahw + bhw * k00 + bhh * k01 - wp.abs(dA0)
    if oA0 <= m:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    oA1 = ahh + bhw * k10 + bhh * k11 - wp.abs(dA1)
    if oA1 <= m:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    oB0 = bhw + ahw * k00 + ahh * k10 - wp.abs(dB0)
    if oB0 <= m:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    oB1 = bhh + ahw * k01 + ahh * k11 - wp.abs(dB1)
    if oB1 <= m:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    best = oA0
    axis = int(0)
    if oA1 < best:
        best = oA1
        axis = 1
    ob = oB1
    kb = int(3)
    if oB0 < oB1:
        ob = oB0
        kb = 2
    if ob < best - 0.05 * wp.abs(best) - 0.01:
        best = ob
        axis = kb

    # reference box (rx.. ) and incident box (ix..)
    rcx = float(0.0)
    rcy = float(0.0)
    ux = float(0.0)
    uy = float(0.0)
    hn = float(0.0)
    tx = float(0.0)
    ty = float(0.0)
    ht = float(0.0)
    sgn = float(1.0)
    flip = float(1.0)
    icx = float(0.0)
    icy = float(0.0)
    ic = float(0.0)
    is_ = float(0.0)
    ihw = float(0.0)
    ihh = float(0.0)
    if axis < 2:
        rcx = ax
        rcy = ay
        icx = bx
        icy = by
        ic = bc
        is_ = bs
        ihw = bhw
        ihh = bhh
        dproj = float(0.0)
        if axis == 0:
            ux = a0x
            uy = a0y
            dproj = dA0
            hn = ahw
            tx = a1x
            ty = a1y
            ht = ahh
        else:
            ux = a1x
            uy = a1y
            dproj = dA1
            hn = ahh
            tx = a0x
            ty = a0y
            ht = ahw
        if dproj < 0.0:
            sgn = -1.0
        flip = 1.0
    else:
        rcx = bx
        rcy = by
        icx = ax
        icy = ay
        ic = ac
        is_ = as_
        ihw = ahw
        ihh = ahh
        dproj = float(0.0)
        if axis == 2:
            ux = b0x
            uy = b0y
            dproj = dB0
            hn = bhw
            tx = b1x
            ty = b1y
            ht = bhh
        else:
            ux = b1x
            uy = b1y
            dproj = dB1
            hn = bhh
            tx = b0x
            ty = b0y
            ht = bhw
        sgn = 1.0
        if dproj >= 0.0:
            sgn = -1.0
        flip = -1.0
    nrx = sgn * ux
    nry = sgn * uy

    i0x = ic
    i0y = is_
    i1x = -is_
    i1y = ic
    d0 = nrx * i0x + nry * i0y
    d1 = nrx * i1x + nry * i1y
    nix = float(0.0)
    niy = float(0.0)
    hn_i = float(0.0)
    tix = float(0.0)
    tiy = float(0.0)
    ht_i = float(0.0)
    if wp.abs(d0) > wp.abs(d1):
        s2 = float(1.0)
        if d0 > 0.0:
            s2 = -1.0
        nix = s2 * i0x
        niy = s2 * i0y
        hn_i = ihw
        tix = i1x
        tiy = i1y
        ht_i = ihh
    else:
        s2 = float(1.0)
        if d1 > 0.0:
            s2 = -1.0
        nix = s2 * i1x
        niy = s2 * i1y
        hn_i = ihh
        tix = i0x
        tiy = i0y
        ht_i = ihw
    fcx = icx + nix * hn_i
    fcy = icy + niy * hn_i
    v1x = fcx + tix * ht_i
    v1y = fcy + tiy * ht_i
    v2x = fcx - tix * ht_i
    v2y = fcy - tiy * ht_i
    s1 = (v1x - rcx) * tx + (v1y - rcy) * ty
    s2v = (v2x - rcx) * tx + (v2y - rcy) * ty
    ok, c1x, c1y, c1s, c2x, c2y, c2s = _clip(v1x, v1y, s1, v2x, v2y, s2v, 1.0, ht)
    if ok == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ok2, e1x, e1y, e1s, e2x, e2y, e2s = _clip(c1x, c1y, c1s, c2x, c2y, c2s, -1.0, ht)
    if ok2 == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    depth1 = hn - ((e1x - rcx) * nrx + (e1y - rcy) * nry)
    depth2 = hn - ((e2x - rcx) * nrx + (e2y - rcy) * nry)
    keep1 = depth1 > -margin
    keep2 = depth2 > -margin
    n = int(0)
    x0 = float(0.0)
    y0 = float(0.0)
    dd0 = float(0.0)
    x1 = float(0.0)
    y1 = float(0.0)
    dd1 = float(0.0)
    if keep1:
        n = 1
        x0 = e1x
        y0 = e1y
        dd0 = depth1
        if keep2:
            n = 2
            x1 = e2x
            y1 = e2y
            dd1 = depth2
    elif keep2:
        n = 1
        x0 = e2x
        y0 = e2y
        dd0 = depth2
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return n, nrx * flip, nry * flip, x0, y0, dd0, x1, y1, dd1


@wp.func
def _collide(i: int, j: int, margin: float,
             kind: wp.array(dtype=int), px: wp.array(dtype=float), py: wp.array(dtype=float),
             cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
             hw: wp.array(dtype=float), hh: wp.array(dtype=float), rad: wp.array(dtype=float)):
    """Dispatch on body types.  Normal points from body i to body j."""
    if kind[i] == 0:
        if kind[j] == 0:
            return _box_box(px[i], py[i], cosa[i], sina[i], hw[i], hh[i],
                            px[j], py[j], cosa[j], sina[j], hw[j], hh[j], margin)
        # box i, circle j: circle_box already points box -> circle
        return _circle_box(px[j], py[j], rad[j], px[i], py[i], cosa[i], sina[i], hw[i], hh[i], margin)
    if kind[j] == 1:
        return _circle_circle(px[i], py[i], rad[i], px[j], py[j], rad[j], margin)
    # circle i, box j: circle_box points box -> circle (j -> i), so flip
    n, nx, ny, x0, y0, d0, x1, y1, d1 = _circle_box(px[i], py[i], rad[i], px[j], py[j], cosa[j], sina[j],
                                                    hw[j], hh[j], margin)
    return n, -nx, -ny, x0, y0, d0, x1, y1, d1


@wp.func
def _bound(i: int, kind: wp.array(dtype=int), hw: wp.array(dtype=float), hh: wp.array(dtype=float),
           rad: wp.array(dtype=float)):
    if kind[i] == 1:
        return rad[i]
    return wp.sqrt(hw[i] * hw[i] + hh[i] * hh[i])


# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------
@wp.kernel
def k_attract(dt: float, params: wp.array(dtype=float), active: wp.array(dtype=int), fixed: wp.array(dtype=int),
              asleep: wp.array(dtype=int), sleep_t: wp.array(dtype=float), px: wp.array(dtype=float),
              py: wp.array(dtype=float), inv_m: wp.array(dtype=float), vx: wp.array(dtype=float),
              vy: wp.array(dtype=float)):
    """Newton's law of universal gravitation between every pair of movable bodies:
    F = G m1 m2 / r^2, so body i accelerates by G m_j / r^2 towards each j (softened at short range).
    Fixed bodies neither move nor attract (they have no inverse mass), so the ground stays out of it."""
    i = wp.tid()
    g = params[P_G]
    if g == 0.0 or active[i] == 0 or fixed[i] == 1:
        return
    ax = float(0.0)
    ay = float(0.0)
    for j in range(KCAP):
        if j == i or active[j] == 0 or fixed[j] == 1 or inv_m[j] <= 0.0:
            continue
        dx = px[j] - px[i]
        dy = py[j] - py[i]
        r2 = dx * dx + dy * dy + GRAV_SOFT * GRAV_SOFT
        inv_r = 1.0 / wp.sqrt(r2)
        a = g / (inv_m[j] * r2)                    # G * m_j / r^2
        ax += a * dx * inv_r
        ay += a * dy * inv_r
    if asleep[i] == 1:
        if ax * ax + ay * ay < 1.0:
            return
        asleep[i] = 0                              # a real pull: wake up
        sleep_t[i] = 0.0
    vx[i] = vx[i] + ax * dt
    vy[i] = vy[i] + ay * dt


@wp.kernel
def k_kinetic(active: wp.array(dtype=int), fixed: wp.array(dtype=int), inv_m: wp.array(dtype=float),
              inv_i: wp.array(dtype=float), vx: wp.array(dtype=float), vy: wp.array(dtype=float),
              w: wp.array(dtype=float), ke: wp.array(dtype=float), total: wp.array(dtype=float)):
    """Kinetic energy per body: 1/2 m v^2 (moving) + 1/2 I w^2 (spinning)."""
    i = wp.tid()
    e = float(0.0)
    if active[i] == 1 and fixed[i] == 0:
        if inv_m[i] > 0.0:
            e += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]) / inv_m[i]
        if inv_i[i] > 0.0:
            e += 0.5 * w[i] * w[i] / inv_i[i]
    ke[i] = e
    wp.atomic_add(total, 0, e)


@wp.kernel
def k_forces(dt: float, params: wp.array(dtype=float), active: wp.array(dtype=int), fixed: wp.array(dtype=int),
             asleep: wp.array(dtype=int), sleep_t: wp.array(dtype=float), kind: wp.array(dtype=int),
             vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float),
             clock: wp.array(dtype=float)):
    i = wp.tid()
    if i == 0:
        clock[0] = clock[0] + dt
    if active[i] == 0 or fixed[i] == 1:
        return
    ax = params[P_AX]
    if asleep[i] == 1:
        if ax * ax < 1.0:
            return
        asleep[i] = 0                                   # the ground is shaking: nothing sleeps
        sleep_t[i] = 0.0
    vy[i] = (vy[i] + GRAVITY * dt) * 0.9997
    vx[i] = (vx[i] + ax * dt) * 0.9997
    if kind[i] == 0:
        w[i] = w[i] * 0.998
    else:
        w[i] = w[i] * 0.994


@wp.kernel
def k_grab(dt: float, params: wp.array(dtype=float), px: wp.array(dtype=float), py: wp.array(dtype=float),
           cosa: wp.array(dtype=float), sina: wp.array(dtype=float), inv_m: wp.array(dtype=float),
           inv_i: wp.array(dtype=float), vx: wp.array(dtype=float), vy: wp.array(dtype=float),
           w: wp.array(dtype=float), asleep: wp.array(dtype=int), sleep_t: wp.array(dtype=float)):
    if params[P_GRAB_ON] < 0.5:
        return
    b = int(params[P_GRAB_BODY])
    lx = params[P_GRAB_LX]
    ly = params[P_GRAB_LY]
    c = cosa[b]
    s = sina[b]
    rx = lx * c - ly * s
    ry = lx * s + ly * c
    ax = px[b] + rx
    ay = py[b] + ry
    asleep[b] = 0
    sleep_t[b] = 0.0
    vax = vx[b] - w[b] * ry
    vay = vy[b] + w[b] * rx
    ex = params[P_MX] - ax
    ey = params[P_MY] - ay
    dist = wp.sqrt(ex * ex + ey * ey)
    if dist > GRAB_REACH:
        ex = ex * GRAB_REACH / dist
        ey = ey * GRAB_REACH / dist
    fx = GRAB_K * ex - GRAB_C * vax
    fy = GRAB_K * ey - GRAB_C * vay
    mag = wp.sqrt(fx * fx + fy * fy)
    if mag > GRAB_MAX:
        fx = fx * GRAB_MAX / mag
        fy = fy * GRAB_MAX / mag
    ix = fx * dt
    iy = fy * dt
    m = 1.0 / inv_m[b]
    vx[b] = vx[b] + ix
    vy[b] = vy[b] + iy
    w[b] = w[b] + m * inv_i[b] * (rx * iy - ry * ix)


@wp.kernel
def k_reset_counts(ccount: wp.array(dtype=int), used: wp.array(dtype=int)):
    i = wp.tid()
    if i == 0:
        ccount[0] = 0
    used[i] = 0


@wp.kernel
def k_narrow(dt: float, params: wp.array(dtype=float), clock: wp.array(dtype=float),
             active: wp.array(dtype=int), fixed: wp.array(dtype=int), asleep: wp.array(dtype=int),
             sleep_t: wp.array(dtype=float), kind: wp.array(dtype=int),
             px: wp.array(dtype=float), py: wp.array(dtype=float),
             cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
             vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float),
             hw: wp.array(dtype=float), hh: wp.array(dtype=float), rad: wp.array(dtype=float),
             inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=float),
             mu: wp.array(dtype=float), rest: wp.array(dtype=float),
             glue_born: wp.array(dtype=float), glue_str: wp.array(dtype=float),
             warm_pn: wp.array(dtype=float), warm_pt: wp.array(dtype=float),
             ccount: wp.array(dtype=int),
             c_a: wp.array(dtype=int), c_b: wp.array(dtype=int), c_key: wp.array(dtype=int),
             c_nx: wp.array(dtype=float), c_ny: wp.array(dtype=float),
             c_rax: wp.array(dtype=float), c_ray: wp.array(dtype=float),
             c_rbx: wp.array(dtype=float), c_rby: wp.array(dtype=float),
             c_mn: wp.array(dtype=float), c_mt: wp.array(dtype=float), c_bias: wp.array(dtype=float),
             c_pn: wp.array(dtype=float), c_pt: wp.array(dtype=float), c_mu: wp.array(dtype=float),
             c_coh: wp.array(dtype=float), adj: wp.array(dtype=int)):
    tid = wp.tid()
    i = tid // KCAP
    j = tid - i * KCAP
    if j <= i:
        return
    if active[i] == 0 or active[j] == 0:
        return
    if asleep[i] == 1 and asleep[j] == 1:        # two sleepers (statics count as asleep)
        return
    key = i * KCAP + j
    for kk in range(KMAXDEG):                    # bodies held together by a live joint do not collide with each other
        if adj[i * KMAXDEG + kk] == j:
            warm_pn[key * 2] = 0.0
            warm_pn[key * 2 + 1] = 0.0
            warm_pt[key * 2] = 0.0
            warm_pt[key * 2 + 1] = 0.0
            return
    glued = glue_born[key]
    margin = float(0.0)
    if glued >= 0.0:
        margin = GLUE_MARGIN
    dx = px[j] - px[i]
    dy = py[j] - py[i]
    reach = _bound(i, kind, hw, hh, rad) + _bound(j, kind, hw, hh, rad) + margin
    if dx * dx + dy * dy > reach * reach:
        warm_pn[key * 2] = 0.0
        warm_pn[key * 2 + 1] = 0.0
        warm_pt[key * 2] = 0.0
        warm_pt[key * 2 + 1] = 0.0
        return
    n, nx, ny, x0, y0, d0, x1, y1, d1 = _collide(i, j, margin, kind, px, py, cosa, sina, hw, hh, rad)
    if n == 0:
        warm_pn[key * 2] = 0.0
        warm_pn[key * 2 + 1] = 0.0
        warm_pt[key * 2] = 0.0
        warm_pt[key * 2 + 1] = 0.0
        return
    if asleep[i] == 1 and fixed[i] == 0:          # touched by something awake: wake up
        asleep[i] = 0
        sleep_t[i] = 0.0
    if asleep[j] == 1 and fixed[j] == 0:
        asleep[j] = 0
        sleep_t[j] = 0.0
    friction = wp.sqrt(mu[i] * mu[j])
    e = wp.min(rest[i], rest[j])
    tx = -ny
    ty = nx
    coh = float(0.0)
    if glued >= 0.0:
        cure = 0.5 + 0.5 * wp.min(1.0, (clock[0] - glued) / CEMENT_CURE)
        coh = params[P_CEMENT] * cure * dt * glue_str[key]
    inv_dt = 1.0 / dt
    for k in range(n):
        cx = x0
        cy = y0
        depth = d0
        if k == 1:
            cx = x1
            cy = y1
            depth = d1
        idx = wp.atomic_add(ccount, 0, 1)
        if idx >= KMAXC:
            return
        rax = cx - px[i]
        ray = cy - py[i]
        rbx = cx - px[j]
        rby = cy - py[j]
        rna = rax * ny - ray * nx
        rnb = rbx * ny - rby * nx
        kn = inv_m[i] + inv_m[j] + inv_i[i] * rna * rna + inv_i[j] * rnb * rnb
        rta = rax * ty - ray * tx
        rtb = rbx * ty - rby * tx
        kt = inv_m[i] + inv_m[j] + inv_i[i] * rta * rta + inv_i[j] * rtb * rtb
        mn = float(0.0)
        mt = float(0.0)
        if kn > 0.0:
            mn = 1.0 / kn
        if kt > 0.0:
            mt = 1.0 / kt
        dvx = (vx[j] - w[j] * rby) - (vx[i] - w[i] * ray)
        dvy = (vy[j] + w[j] * rbx) - (vy[i] + w[i] * rax)
        vn0 = dvx * nx + dvy * ny
        bias = float(0.0)
        if depth > 0.0:
            if depth > SLOP:
                bias = wp.min(BETA * inv_dt * (depth - SLOP), 200.0)
        else:
            bias = depth * inv_dt
        if vn0 < -REST_THRESHOLD and e > 0.0:
            bounce = -e * vn0
            if bounce > bias:
                bias = bounce
        c_a[idx] = i
        c_b[idx] = j
        c_key[idx] = key * 2 + k
        c_nx[idx] = nx
        c_ny[idx] = ny
        c_rax[idx] = rax
        c_ray[idx] = ray
        c_rbx[idx] = rbx
        c_rby[idx] = rby
        c_mn[idx] = mn
        c_mt[idx] = mt
        c_bias[idx] = bias
        c_mu[idx] = friction
        c_coh[idx] = coh
        c_pn[idx] = warm_pn[key * 2 + k]
        c_pt[idx] = warm_pt[key * 2 + k]


@wp.kernel
def k_warm_start(ccount: wp.array(dtype=int), c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
                     c_nx: wp.array(dtype=float), c_ny: wp.array(dtype=float),
                     c_rax: wp.array(dtype=float), c_ray: wp.array(dtype=float),
                     c_rbx: wp.array(dtype=float), c_rby: wp.array(dtype=float),
                     c_pn: wp.array(dtype=float), c_pt: wp.array(dtype=float),
                     inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=float),
                     vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float)):
    c = wp.tid()
    if c >= ccount[0]:
        return
    a = c_a[c]
    b = c_b[c]
    nx = c_nx[c]
    ny = c_ny[c]
    tx = -ny
    ty = nx
    pnx = c_pn[c] * nx + c_pt[c] * tx
    pny = c_pn[c] * ny + c_pt[c] * ty
    wp.atomic_add(vx, a, -inv_m[a] * pnx)
    wp.atomic_add(vy, a, -inv_m[a] * pny)
    wp.atomic_add(w, a, -inv_i[a] * (c_rax[c] * pny - c_ray[c] * pnx))
    wp.atomic_add(vx, b, inv_m[b] * pnx)
    wp.atomic_add(vy, b, inv_m[b] * pny)
    wp.atomic_add(w, b, inv_i[b] * (c_rbx[c] * pny - c_rby[c] * pnx))


@wp.kernel
def k_color_init(ccount: wp.array(dtype=int), c_color: wp.array(dtype=int), c_prio: wp.array(dtype=int),
                 c_try: wp.array(dtype=int)):
    c = wp.tid()
    c_color[c] = -1
    c_try[c] = -1
    h = (c * 1103515245 + 12345) & 0x1FFFF            # pseudo-random priority, made unique by the index
    c_prio[c] = (h << 13) | c


@wp.kernel
def k_color_propose(ccount: wp.array(dtype=int), c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
                    fixed: wp.array(dtype=int), c_color: wp.array(dtype=int), c_prio: wp.array(dtype=int),
                    c_try: wp.array(dtype=int), used: wp.array(dtype=int), prop: wp.array(dtype=int)):
    """Edge colouring, round 1: every uncoloured contact proposes the lowest colour that is
    free on both of its bodies (contacts that share a dynamic body must get different colours)."""
    c = wp.tid()
    if c >= ccount[0] or c_color[c] >= 0:
        return
    a = c_a[c]
    b = c_b[c]
    m = used[a] | used[b]
    col = int(0)
    while col < NCOLORS and ((m >> col) & 1) == 1:
        col += 1
    c_try[c] = col
    if col >= NCOLORS:
        return
    if fixed[a] == 0:
        wp.atomic_max(prop, a * 32 + col, c_prio[c])
    if fixed[b] == 0:
        wp.atomic_max(prop, b * 32 + col, c_prio[c])


@wp.kernel
def k_color_commit(ccount: wp.array(dtype=int), c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
                   fixed: wp.array(dtype=int), c_color: wp.array(dtype=int), c_prio: wp.array(dtype=int),
                   c_try: wp.array(dtype=int), used: wp.array(dtype=int), prop: wp.array(dtype=int)):
    """Round 2: a contact keeps its proposal if nobody with a higher priority wanted the same colour."""
    c = wp.tid()
    if c >= ccount[0] or c_color[c] >= 0:
        return
    col = c_try[c]
    if col < 0 or col >= NCOLORS:
        return
    a = c_a[c]
    b = c_b[c]
    win = True
    if fixed[a] == 0 and prop[a * 32 + col] != c_prio[c]:
        win = False
    if fixed[b] == 0 and prop[b * 32 + col] != c_prio[c]:
        win = False
    if win:
        c_color[c] = col
        if fixed[a] == 0:
            wp.atomic_or(used, a, 1 << col)
        if fixed[b] == 0:
            wp.atomic_or(used, b, 1 << col)


@wp.kernel
def k_color_clear(prop: wp.array(dtype=int)):
    prop[wp.tid()] = 0


@wp.kernel
def k_solve(color: int, ccount: wp.array(dtype=int), c_color: wp.array(dtype=int),
            c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
            c_nx: wp.array(dtype=float), c_ny: wp.array(dtype=float),
            c_rax: wp.array(dtype=float), c_ray: wp.array(dtype=float),
            c_rbx: wp.array(dtype=float), c_rby: wp.array(dtype=float),
            c_mn: wp.array(dtype=float), c_mt: wp.array(dtype=float), c_bias: wp.array(dtype=float),
            c_pn: wp.array(dtype=float), c_pt: wp.array(dtype=float), c_mu: wp.array(dtype=float),
            c_coh: wp.array(dtype=float),
            inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=float),
            vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float)):
    """Gauss-Seidel on the GPU: all contacts of one colour touch different bodies, so they are
    solved in parallel, and the colours are swept one after another."""
    c = wp.tid()
    if c >= ccount[0] or c_color[c] != color:
        return
    a = c_a[c]
    b = c_b[c]
    nx = c_nx[c]
    ny = c_ny[c]
    tx = -ny
    ty = nx
    rax = c_rax[c]
    ray = c_ray[c]
    rbx = c_rbx[c]
    rby = c_rby[c]
    ima = inv_m[a]
    imb = inv_m[b]
    iia = inv_i[a]
    iib = inv_i[b]
    # normal row: may pull (down to -coh) when glued
    dvx = (vx[b] - w[b] * rby) - (vx[a] - w[a] * ray)
    dvy = (vy[b] + w[b] * rbx) - (vy[a] + w[a] * rax)
    dpn = c_mn[c] * (c_bias[c] - (dvx * nx + dvy * ny))
    old = c_pn[c]
    coh = c_coh[c]
    new = old + dpn
    if new < -coh:
        new = -coh
    c_pn[c] = new
    dpn = new - old
    pxn = dpn * nx
    pyn = dpn * ny
    wp.atomic_add(vx, a, -ima * pxn)
    wp.atomic_add(vy, a, -ima * pyn)
    wp.atomic_add(w, a, -iia * (rax * pyn - ray * pxn))
    wp.atomic_add(vx, b, imb * pxn)
    wp.atomic_add(vy, b, imb * pyn)
    wp.atomic_add(w, b, iib * (rbx * pyn - rby * pxn))
    # friction (cohesion adds to the limit while the cement holds)
    dvx = (vx[b] - w[b] * rby) - (vx[a] - w[a] * ray)
    dvy = (vy[b] + w[b] * rbx) - (vy[a] + w[a] * rax)
    dpt = -c_mt[c] * (dvx * tx + dvy * ty)
    lim = c_mu[c] * wp.max(new, 0.0) + coh
    oldt = c_pt[c]
    newt = wp.clamp(oldt + dpt, -lim, lim)
    c_pt[c] = newt
    dpt = newt - oldt
    px = dpt * tx
    py = dpt * ty
    wp.atomic_add(vx, a, -ima * px)
    wp.atomic_add(vy, a, -ima * py)
    wp.atomic_add(w, a, -iia * (rax * py - ray * px))
    wp.atomic_add(vx, b, imb * px)
    wp.atomic_add(vy, b, imb * py)
    wp.atomic_add(w, b, iib * (rbx * py - rby * px))


@wp.kernel
def k_store_and_break(ccount: wp.array(dtype=int), c_key: wp.array(dtype=int), c_a: wp.array(dtype=int),
                      c_b: wp.array(dtype=int), c_pn: wp.array(dtype=float), c_pt: wp.array(dtype=float),
                      c_mu: wp.array(dtype=float), c_coh: wp.array(dtype=float),
                      warm_pn: wp.array(dtype=float), warm_pt: wp.array(dtype=float),
                      glue_born: wp.array(dtype=float)):
    c = wp.tid()
    if c >= ccount[0]:
        return
    warm_pn[c_key[c]] = c_pn[c]
    warm_pt[c_key[c]] = c_pt[c]
    coh = c_coh[c]
    if coh > 0.0:
        pn = c_pn[c]
        if pn <= -coh * 0.98 or wp.abs(c_pt[c]) >= (c_mu[c] * wp.max(pn, 0.0) + coh) * 0.98:
            glue_born[c_key[c] // 2] = -1.0           # the cement cracked


@wp.kernel
def k_peak(ccount: wp.array(dtype=int), c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
           c_pn: wp.array(dtype=float), inv_m: wp.array(dtype=float), peak: wp.array(dtype=float)):
    """The hardest shock each body took: the largest velocity change a single contact gave it in one step."""
    c = wp.tid()
    if c >= ccount[0]:
        return
    pn = wp.max(c_pn[c], 0.0)
    wp.atomic_max(peak, c_a[c], pn * inv_m[c_a[c]])
    wp.atomic_max(peak, c_b[c], pn * inv_m[c_b[c]])


@wp.kernel
def k_integrate(dt: float, active: wp.array(dtype=int), fixed: wp.array(dtype=int),
                asleep: wp.array(dtype=int), nosleep: wp.array(dtype=int), grabbed: int,
                px: wp.array(dtype=float), py: wp.array(dtype=float), ang: wp.array(dtype=float),
                cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
                vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float),
                sleep_t: wp.array(dtype=float)):
    i = wp.tid()
    if active[i] == 0 or fixed[i] == 1 or asleep[i] == 1:
        return
    sp = wp.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
    if nosleep[i] == 0 and i != grabbed and sp < SLEEP_SPEED and wp.abs(w[i]) < SLEEP_SPIN:
        sleep_t[i] = sleep_t[i] + dt
    else:
        sleep_t[i] = 0.0
    if sp > MAX_SPEED:
        vx[i] = vx[i] * MAX_SPEED / sp
        vy[i] = vy[i] * MAX_SPEED / sp
    px[i] = px[i] + vx[i] * dt
    py[i] = py[i] + vy[i] * dt
    ang[i] = ang[i] + w[i] * dt
    cosa[i] = wp.cos(ang[i])
    sina[i] = wp.sin(ang[i])
    if py[i] > WORLD_H + 300.0 or px[i] < -300.0 or px[i] > WORLD_W + 300.0:
        active[i] = 0                                  # fell off the world


@wp.kernel
def k_rope(params: wp.array(dtype=float), px: wp.array(dtype=float), py: wp.array(dtype=float),
           vx: wp.array(dtype=float), vy: wp.array(dtype=float)):
    """The tethered body can't get further from its pivot than the rope is long."""
    b = int(params[P_TETHER])
    if b < 0:
        return
    dx = px[b] - params[P_PIVX]
    dy = py[b] - params[P_PIVY]
    d = wp.sqrt(dx * dx + dy * dy)
    rope = params[P_ROPE]
    if d > rope:
        nx = dx / d
        ny = dy / d
        px[b] = params[P_PIVX] + nx * rope
        py[b] = params[P_PIVY] + ny * rope
        vn = vx[b] * nx + vy[b] * ny
        if vn > 0.0:
            vx[b] = vx[b] - vn * nx
            vy[b] = vy[b] - vn * ny


# ---- sleeping islands -------------------------------------------------------
@wp.kernel
def k_sleep_init(active: wp.array(dtype=int), fixed: wp.array(dtype=int), asleep: wp.array(dtype=int),
                 nosleep: wp.array(dtype=int), sleep_t: wp.array(dtype=float), blocked: wp.array(dtype=int)):
    i = wp.tid()
    blocked[i] = 0
    if active[i] == 1 and fixed[i] == 0 and asleep[i] == 0:
        if sleep_t[i] <= SLEEP_TIME or nosleep[i] == 1:
            blocked[i] = 1


@wp.kernel
def k_sleep_prop(ccount: wp.array(dtype=int), c_a: wp.array(dtype=int), c_b: wp.array(dtype=int),
                 fixed: wp.array(dtype=int), asleep: wp.array(dtype=int), blocked: wp.array(dtype=int)):
    c = wp.tid()
    if c >= ccount[0]:
        return
    a = c_a[c]
    b = c_b[c]
    if fixed[a] == 1 or fixed[b] == 1 or asleep[a] == 1 or asleep[b] == 1:
        return
    m = wp.max(blocked[a], blocked[b])
    wp.atomic_max(blocked, a, m)
    wp.atomic_max(blocked, b, m)


@wp.kernel
def k_sleep_apply(active: wp.array(dtype=int), fixed: wp.array(dtype=int), asleep: wp.array(dtype=int),
                  blocked: wp.array(dtype=int), vx: wp.array(dtype=float), vy: wp.array(dtype=float),
                  w: wp.array(dtype=float)):
    i = wp.tid()
    if active[i] == 1 and fixed[i] == 0 and asleep[i] == 0 and blocked[i] == 0:
        asleep[i] = 1
        vx[i] = 0.0
        vy[i] = 0.0
        w[i] = 0.0


# ---- host-triggered helpers (not part of the step graph) --------------------
@wp.kernel
def k_weld_one(b: int, mult: float, only_static: int, clock: wp.array(dtype=float), active: wp.array(dtype=int),
               noglue: wp.array(dtype=int), fixed: wp.array(dtype=int), kind: wp.array(dtype=int),
               px: wp.array(dtype=float), py: wp.array(dtype=float),
               cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
               hw: wp.array(dtype=float), hh: wp.array(dtype=float), rad: wp.array(dtype=float),
               glue_born: wp.array(dtype=float), glue_str: wp.array(dtype=float)):
    """Join body b to every body it touches, with a joint `mult` times as strong as plain cement.
    only_static = 1 joins it to fixed things only (an anchor into the ground)."""
    o = wp.tid()
    if o == b or active[o] == 0 or noglue[o] == 1 or noglue[b] == 1:
        return
    if only_static == 1 and fixed[o] == 0:
        return
    i = wp.min(o, b)
    j = wp.max(o, b)
    n, nx, ny, x0, y0, d0, x1, y1, d1 = _collide(i, j, GLUE_MARGIN, kind, px, py, cosa, sina, hw, hh, rad)
    if n > 0:
        key = i * KCAP + j
        if glue_born[key] < 0.0:
            glue_born[key] = clock[0]
            glue_str[key] = mult
        elif mult > glue_str[key]:
            glue_str[key] = mult


@wp.kernel
def k_weld_all(mult: float, clock: wp.array(dtype=float), active: wp.array(dtype=int), noglue: wp.array(dtype=int),
               kind: wp.array(dtype=int), px: wp.array(dtype=float), py: wp.array(dtype=float),
               cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
               hw: wp.array(dtype=float), hh: wp.array(dtype=float), rad: wp.array(dtype=float),
               glue_born: wp.array(dtype=float), glue_str: wp.array(dtype=float)):
    tid = wp.tid()
    i = tid // KCAP
    j = tid - i * KCAP
    if j <= i or active[i] == 0 or active[j] == 0 or noglue[i] == 1 or noglue[j] == 1:
        return
    n, nx, ny, x0, y0, d0, x1, y1, d1 = _collide(i, j, GLUE_MARGIN, kind, px, py, cosa, sina, hw, hh, rad)
    if n > 0 and glue_born[tid] < 0.0:
        glue_born[tid] = clock[0]
        glue_str[tid] = mult


@wp.kernel
def k_glue_flags(clock: wp.array(dtype=float), glue_born: wp.array(dtype=float),
                 gflag: wp.array(dtype=int), gcount: wp.array(dtype=int)):
    """gflag[body] = 0 none, 1 wet cement, 2 cured cement; gcount[0] = number of glued pairs."""
    tid = wp.tid()
    born = glue_born[tid]
    if born < 0.0:
        return
    i = tid // KCAP
    j = tid - i * KCAP
    f = int(1)
    if clock[0] - born >= CEMENT_CURE:
        f = 2
    wp.atomic_max(gflag, i, f)
    wp.atomic_max(gflag, j, f)
    wp.atomic_add(gcount, 0, 1)


@wp.kernel
def k_clear_flags(gflag: wp.array(dtype=int), gcount: wp.array(dtype=int)):
    i = wp.tid()
    gflag[i] = 0
    if i == 0:
        gcount[0] = 0


@wp.kernel
def k_kick(b: int, dvx: float, dvy: float, dw: float, vx: wp.array(dtype=float), vy: wp.array(dtype=float),
           w: wp.array(dtype=float), asleep: wp.array(dtype=int), sleep_t: wp.array(dtype=float)):
    vx[b] = vx[b] + dvx
    vy[b] = vy[b] + dvy
    w[b] = w[b] + dw
    asleep[b] = 0
    sleep_t[b] = 0.0


@wp.kernel
def k_set_material(mu_new: float, e_new: float, is_tunable: wp.array(dtype=int),
                   mu: wp.array(dtype=float), rest: wp.array(dtype=float)):
    i = wp.tid()
    if is_tunable[i] == 1:
        mu[i] = mu_new
        rest[i] = e_new


@wp.kernel
def k_free_slot(s: int, glue_born: wp.array(dtype=float), warm_pn: wp.array(dtype=float),
                warm_pt: wp.array(dtype=float), glue_str: wp.array(dtype=float)):
    """Forget everything pairing slot s with another body (cement, warm start)."""
    o = wp.tid()
    key = wp.min(o, s) * KCAP + wp.max(o, s)
    glue_born[key] = -1.0
    glue_str[key] = 1.0
    warm_pn[key * 2] = 0.0
    warm_pn[key * 2 + 1] = 0.0
    warm_pt[key * 2] = 0.0
    warm_pt[key * 2 + 1] = 0.0


@wp.kernel
def k_wake_all(active: wp.array(dtype=int), fixed: wp.array(dtype=int), asleep: wp.array(dtype=int),
               sleep_t: wp.array(dtype=float)):
    i = wp.tid()
    if active[i] == 1 and fixed[i] == 0:
        asleep[i] = 0
        sleep_t[i] = 0.0


@wp.kernel
def k_clear_glue(glue_born: wp.array(dtype=float), glue_str: wp.array(dtype=float)):
    glue_born[wp.tid()] = -1.0
    glue_str[wp.tid()] = 1.0



# ----------------------------------------------------------------------------
# anchored joints
# A joint ties a point on body A to a point on body B (the anchor, stored in each body's own frame) and locks their
# relative angle.  It is solved as a 2x2 point block + one angle row, joints of one colour in parallel (they touch
# different bodies), exactly like the contact solver.  The impulse the solver has to apply to hold the joint is the
# load on it; converted to force (impulse / dt) and split along the joint axis it gives
#     T  tension (pulling the two bodies apart along the axis; compression never breaks a joint),
#     S  shear (sliding sideways),     M  moment (bending / torsion),
# and the joint breaks when the interaction  (T/Tmax)^2 + (S/Smax)^2 + (M/Mmax)^2  reaches 1.  Above FATIGUE_U0 the
# joint also wears (damage), which weakens it, so repeated loading breaks what one push would not.
# ----------------------------------------------------------------------------
@wp.kernel
def k_joint_warm(fac: float, ja: wp.array(dtype=int), jb: wp.array(dtype=int), jalive: wp.array(dtype=int),
                 jlax: wp.array(dtype=float), jlay: wp.array(dtype=float),
                 jlbx: wp.array(dtype=float), jlby: wp.array(dtype=float),
                 jpx: wp.array(dtype=float), jpy: wp.array(dtype=float), jpm: wp.array(dtype=float),
                 cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
                 inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=float),
                 vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float),
                 asleep: wp.array(dtype=int), fixed: wp.array(dtype=int), sleep_t: wp.array(dtype=float)):
    j = wp.tid()
    if jalive[j] == 0:
        return
    a = ja[j]
    b = jb[j]
    if asleep[a] == 1 and asleep[b] == 1:
        return
    if asleep[a] == 1 and fixed[a] == 0:          # the other end is moving: wake this one
        asleep[a] = 0
        sleep_t[a] = 0.0
    if asleep[b] == 1 and fixed[b] == 0:
        asleep[b] = 0
        sleep_t[b] = 0.0
    rax = jlax[j] * cosa[a] - jlay[j] * sina[a]
    ray = jlax[j] * sina[a] + jlay[j] * cosa[a]
    rbx = jlbx[j] * cosa[b] - jlby[j] * sina[b]
    rby = jlbx[j] * sina[b] + jlby[j] * cosa[b]
    pxx = jpx[j] * fac
    pyy = jpy[j] * fac
    pm = jpm[j] * fac
    jpx[j] = pxx
    jpy[j] = pyy
    jpm[j] = pm
    wp.atomic_add(vx, a, -inv_m[a] * pxx)
    wp.atomic_add(vy, a, -inv_m[a] * pyy)
    wp.atomic_add(w, a, -inv_i[a] * (rax * pyy - ray * pxx + pm))
    wp.atomic_add(vx, b, inv_m[b] * pxx)
    wp.atomic_add(vy, b, inv_m[b] * pyy)
    wp.atomic_add(w, b, inv_i[b] * (rbx * pyy - rby * pxx + pm))


@wp.kernel
def k_joint_solve(color: int, inv_dt: float, ja: wp.array(dtype=int), jb: wp.array(dtype=int),
                  jcolor: wp.array(dtype=int), jalive: wp.array(dtype=int),
                  jlax: wp.array(dtype=float), jlay: wp.array(dtype=float),
                  jlbx: wp.array(dtype=float), jlby: wp.array(dtype=float), jref: wp.array(dtype=float),
                  jpx: wp.array(dtype=float), jpy: wp.array(dtype=float), jpm: wp.array(dtype=float),
                  px: wp.array(dtype=float), py: wp.array(dtype=float), ang: wp.array(dtype=float),
                  cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
                  inv_m: wp.array(dtype=float), inv_i: wp.array(dtype=float),
                  vx: wp.array(dtype=float), vy: wp.array(dtype=float), w: wp.array(dtype=float),
                  asleep: wp.array(dtype=int)):
    j = wp.tid()
    if jalive[j] == 0 or jcolor[j] != color:
        return
    a = ja[j]
    b = jb[j]
    if asleep[a] == 1 and asleep[b] == 1:
        return
    ima = inv_m[a]
    imb = inv_m[b]
    iia = inv_i[a]
    iib = inv_i[b]
    rax = jlax[j] * cosa[a] - jlay[j] * sina[a]
    ray = jlax[j] * sina[a] + jlay[j] * cosa[a]
    rbx = jlbx[j] * cosa[b] - jlby[j] * sina[b]
    rby = jlbx[j] * sina[b] + jlby[j] * cosa[b]
    # point block: make the two anchors move together
    ex = (px[b] + rbx) - (px[a] + rax)
    ey = (py[b] + rby) - (py[a] + ray)
    bx = ex * JOINT_BETA * inv_dt
    by = ey * JOINT_BETA * inv_dt
    bl = wp.sqrt(bx * bx + by * by)
    if bl > JOINT_BIAS_MAX:
        bx = bx * JOINT_BIAS_MAX / bl
        by = by * JOINT_BIAS_MAX / bl
    dvx = (vx[b] - w[b] * rby) - (vx[a] - w[a] * ray)
    dvy = (vy[b] + w[b] * rbx) - (vy[a] + w[a] * rax)
    rx = dvx + bx
    ry = dvy + by
    k11 = ima + imb + iia * ray * ray + iib * rby * rby
    k12 = -(iia * rax * ray + iib * rbx * rby)
    k22 = ima + imb + iia * rax * rax + iib * rbx * rbx
    det = k11 * k22 - k12 * k12
    if det > 1.0e-12:
        pxx = -(k22 * rx - k12 * ry) / det
        pyy = -(k11 * ry - k12 * rx) / det
        wp.atomic_add(vx, a, -ima * pxx)
        wp.atomic_add(vy, a, -ima * pyy)
        wp.atomic_add(w, a, -iia * (rax * pyy - ray * pxx))
        wp.atomic_add(vx, b, imb * pxx)
        wp.atomic_add(vy, b, imb * pyy)
        wp.atomic_add(w, b, iib * (rbx * pyy - rby * pxx))
        jpx[j] = jpx[j] + pxx
        jpy[j] = jpy[j] + pyy
    # angle row: keep the relative angle at its rest value
    ki = iia + iib
    if ki > 1.0e-12:
        err = (ang[b] - ang[a]) - jref[j]
        wrel = w[b] - w[a]
        pm = -(wrel + err * JOINT_BETA * inv_dt) / ki
        wp.atomic_add(w, a, -iia * pm)
        wp.atomic_add(w, b, iib * pm)
        jpm[j] = jpm[j] + pm


@wp.kernel
def k_joint_check(dt: float, clock: wp.array(dtype=float), ja: wp.array(dtype=int), jb: wp.array(dtype=int),
                  jalive: wp.array(dtype=int), jnax: wp.array(dtype=float), jnay: wp.array(dtype=float),
                  jtmax: wp.array(dtype=float), jsmax: wp.array(dtype=float), jmmax: wp.array(dtype=float),
                  jrate: wp.array(dtype=float), jborn: wp.array(dtype=float), jdmg: wp.array(dtype=float),
                  jpx: wp.array(dtype=float), jpy: wp.array(dtype=float), jpm: wp.array(dtype=float),
                  jT: wp.array(dtype=float), jS: wp.array(dtype=float), jM: wp.array(dtype=float),
                  jutil: wp.array(dtype=float), cosa: wp.array(dtype=float), sina: wp.array(dtype=float),
                  asleep: wp.array(dtype=int), adj: wp.array(dtype=int)):
    j = wp.tid()
    if jalive[j] == 0:
        return
    a = ja[j]
    b = jb[j]
    if asleep[a] == 1 and asleep[b] == 1:
        return
    inv = 1.0 / dt
    fx = jpx[j] * inv                              # impulse over one step -> force
    fy = jpy[j] * inv
    nx = jnax[j] * cosa[a] - jnay[j] * sina[a]    # joint axis in the world, from A toward B
    ny = jnax[j] * sina[a] + jnay[j] * cosa[a]
    sm = JOINT_SMOOTH                              # the raw load jitters from step to step: judge a short running average
    tension = jT[j] + (-(fx * nx + fy * ny) - jT[j]) * sm   # the constraint pulls B back toward A when they are pulled apart
    shear = jS[j] + (wp.abs(-fx * ny + fy * nx) - jS[j]) * sm
    moment = jM[j] + (wp.abs(jpm[j]) * inv - jM[j]) * sm
    t = wp.max(tension, 0.0) / jtmax[j]
    sh = shear / jsmax[j]
    m = moment / jmmax[j]
    u = t * t + sh * sh + m * m
    jT[j] = tension
    jS[j] = shear
    jM[j] = moment
    jutil[j] = wp.max(jutil[j], u)
    dm = jdmg[j]
    if u > FATIGUE_U0:
        dm = dm + dt * jrate[j] * (u - FATIGUE_U0)
        jdmg[j] = dm
    eff = u / (1.0 - 0.6 * wp.min(dm, 1.0))
    if clock[0] - jborn[j] < JOINT_GRACE:
        return
    if eff >= 1.0 or dm >= 1.0:
        jalive[j] = 0                              # snapped: free the pair so they collide normally from now on
        jpx[j] = 0.0
        jpy[j] = 0.0
        jpm[j] = 0.0
        for k in range(KMAXDEG):
            if adj[a * KMAXDEG + k] == b:
                adj[a * KMAXDEG + k] = -1
            if adj[b * KMAXDEG + k] == a:
                adj[b * KMAXDEG + k] = -1


@wp.kernel
def k_sleep_prop_joints(ja: wp.array(dtype=int), jb: wp.array(dtype=int), jalive: wp.array(dtype=int),
                        fixed: wp.array(dtype=int), asleep: wp.array(dtype=int), blocked: wp.array(dtype=int)):
    """Bodies tied by a joint sleep and wake together, like bodies in contact."""
    j = wp.tid()
    if jalive[j] == 0:
        return
    a = ja[j]
    b = jb[j]
    if fixed[a] == 1 or fixed[b] == 1 or asleep[a] == 1 or asleep[b] == 1:
        return
    m = wp.max(blocked[a], blocked[b])
    wp.atomic_max(blocked, a, m)
    wp.atomic_max(blocked, b, m)


# Joint strengths: per pixel of joint width (tension, shear), and per pixel^2 for the moment, so a wider connection is
# stronger and a deeper one resists bending better.  `fatigue` is wear per second at full load.  Units are the engine's
# (mass * px / s^2); a 20 x 20 box of the default density weighs about 6700, so a nail 20 px wide holds ~4 of them.
JOINT_MATERIALS = {      # M (bending, per px^2 of face width^2) ~ T/4: the tension couple across a face of width w
    "nail": dict(T=1.5e3, S=2.5e3, M=375.0, fatigue=0.6),       # holds sideways better than it resists being pulled out
    "wood": dict(T=3.0e3, S=2.5e3, M=750.0, fatigue=0.8),       # timber joint: fatigues quickly
    "concrete": dict(T=1.0e3, S=4.0e3, M=250.0, fatigue=0.3),   # strong in shear, weak in tension
    "glass": dict(T=1.5e3, S=1.5e3, M=375.0, fatigue=0.0),      # brittle: no wear, it just goes
    "weld": dict(T=2.0e4, S=1.5e4, M=5000.0, fatigue=0.3),
    "steel": dict(T=3.0e4, S=2.5e4, M=7500.0, fatigue=0.3),
    "anchor": dict(T=1.0e5, S=8.0e4, M=25000.0, fatigue=0.1),   # into the ground
}


# ----------------------------------------------------------------------------
# materials and fracture patterns
# ----------------------------------------------------------------------------
# A body with a `material` breaks when the hardest shock it takes in one readback interval (WarpPhysics.peak_host: the
# biggest velocity change one contact gave it, in px/s) reaches `limit`.  Just over the limit a tough material only
# cracks (two pieces, held by a weak joint); well over it (`shatter` x the limit) it fragments.  `style` picks the
# pattern:  splinter = long strips along the grain, shards = glass slivers fanning out from the impact point,
# chunks = rough blocky pieces.  `joint` names the JOINT_MATERIALS preset the pieces are re-welded with (None = the
# pieces fly free); frag_scale / lat_scale / crack_scale make those welds weaker than the original connection.
MATERIALS = {
    "wood":     dict(density=0.006, mu=0.6, e=0.15, limit=420.0, style="splinter", crack=True, shatter=1.7,
                     joint="wood", frag_scale=0.5, lat_scale=0.25, crack_scale=0.35, min_piece=3.0, pop=0.04),
    "glass":    dict(density=0.0045, mu=0.3, e=0.1, limit=150.0, style="shards", crack=False, shatter=1.0,
                     joint=None, min_piece=3.0, pop=0.10),
    "concrete": dict(density=0.009, mu=0.7, e=0.05, limit=380.0, style="chunks", crack=True, shatter=1.6,
                     joint="concrete", frag_scale=0.3, crack_scale=0.35, min_piece=6.0, chunks=(4, 7), pop=0.04),
    "stone":    dict(density=0.011, mu=0.7, e=0.05, limit=650.0, style="chunks", crack=True, shatter=1.5,
                     joint="concrete", frag_scale=0.3, crack_scale=0.35, min_piece=6.0, chunks=(4, 7), pop=0.04),
    "brick":    dict(density=0.008, mu=0.7, e=0.05, limit=560.0, style="chunks", crack=False, shatter=1.0,
                     joint=None, min_piece=6.0, chunks=(2, 3), pop=0.06),
    "steel":    dict(density=0.02, mu=0.5, e=0.05, limit=None),            # ductile: never fractures
}
FRAG_GAP = 0.2            # px each fragment is shrunk by, so neighbours never start out overlapping
FRAG_IMMUNE = 0.25        # s a new fragment cannot break again (the shock that made it is still ringing)
JOINT_MATERIALS["mortar"] = dict(T=6.0e2, S=1.2e3, M=150.0, fatigue=0.4)      # the weak link of a brick wall


def _partition(lo, hi, n, min_len, rng):
    """Cut [lo, hi] into n random pieces, each at least min_len long (fewer pieces if they don't fit)."""
    n = max(1, min(n, int((hi - lo) // min_len)))
    if n == 1:
        return [lo, hi]
    spare = (hi - lo) - n * min_len
    wts = [rng.random() + 0.15 for _ in range(n)]
    tot = sum(wts)
    edges = [lo]
    for k in range(n):
        edges.append(edges[-1] + min_len + spare * wts[k] / tot)
    edges[-1] = hi
    return edges


def _cut_frac(length, min_len, rng, centre=None):
    """Where to cut a piece of this length (as an offset from its start), keeping both sides >= min_len."""
    lo = max(0.3, min_len / length)
    hi = min(0.7, 1.0 - min_len / length)
    if lo > hi:
        return 0.5 * length
    f = rng.uniform(lo, hi) if centre is None else min(max(centre / length, lo), hi)
    return f * length


def _frag_splinters(hw, hh, ix, iy, rng, spec, crack=False):
    """Wood: strips along the grain (the long axis), each snapped across near the impact at a slightly different
    place, so the break is ragged.  crack=True gives a single clean break across the grain."""
    along_x = hw >= hh
    U, V = (hw, hh) if along_x else (hh, hw)
    iu = ix if along_x else iy
    mp = spec["min_piece"]
    uv = []
    if crack:
        cut = max(-0.6 * U, min(0.6 * U, iu))
        uv = [((cut - U) / 2, 0.0, (cut + U) / 2, V, "end"), ((cut + U) / 2, 0.0, (U - cut) / 2, V, "end")]
    else:
        n = max(1, min(6, int(round(2 * V / rng.uniform(5.0, 9.0)))))
        edges = _partition(-V, V, n, max(2.0, mp * 0.6), rng)
        for k in range(len(edges) - 1):
            v0, v1 = edges[k], edges[k + 1]
            cuts = [max(-0.7 * U, min(0.7 * U, iu + rng.uniform(-0.3, 0.3) * U))]
            if 2 * U > 8 * mp and rng.random() < 0.45:
                cuts.append(rng.uniform(-0.8 * U, 0.8 * U))
            pts = [-U]
            for c in sorted(cuts):
                if c - pts[-1] >= mp and U - c >= mp:
                    pts.append(c)
            pts.append(U)
            for a, b in zip(pts, pts[1:]):
                uv.append(((a + b) / 2, (v0 + v1) / 2, (b - a) / 2, (v1 - v0) / 2, "side"))
    out = []
    for cu, cv, hu, hv, _k in uv:
        if along_x:
            out.append(dict(x=cu, y=cv, hw=hu, hh=hv, a=0.0, grain="x"))
        else:
            out.append(dict(x=cv, y=cu, hw=hv, hh=hu, a=0.0, grain="y"))
    return out


def _frag_chunks(hw, hh, ix, iy, rng, spec, crack=False):
    """Concrete / stone / brick: repeated random straight cuts (the biggest piece first); the first cut passes
    through the impact point."""
    mp = spec["min_piece"]
    rects = [(-hw, hw, -hh, hh)]
    target = 2 if crack else rng.randint(*spec.get("chunks", (4, 7)))
    first = True
    while len(rects) < target:
        done = False
        for r in sorted(rects, key=lambda q: -(q[1] - q[0]) * (q[3] - q[2])):
            w, h = r[1] - r[0], r[3] - r[2]
            if max(w, h) < 2 * mp:
                continue
            if w >= h:
                pos = r[0] + _cut_frac(w, mp, rng, (ix - r[0]) if first else None)
                new = [(r[0], pos, r[2], r[3]), (pos, r[1], r[2], r[3])]
            else:
                pos = r[2] + _cut_frac(h, mp, rng, (iy - r[2]) if first else None)
                new = [(r[0], r[1], r[2], pos), (r[0], r[1], pos, r[3])]
            rects.remove(r)
            rects += new
            first = False
            done = True
            break
        if not done:
            break
    return [dict(x=(r[0] + r[1]) / 2, y=(r[2] + r[3]) / 2, hw=(r[1] - r[0]) / 2, hh=(r[3] - r[2]) / 2, a=0.0,
                 grain=None) for r in rects]


def _frag_shards(hw, hh, ix, iy, rng, spec):
    """Glass: thin slivers fanning out from the impact point to the edges of the pane (not a tiling: mass is
    shared out afterwards, so the shards together weigh what the pane did)."""
    ix = max(-0.7 * hw, min(0.7 * hw, ix))
    iy = max(-0.7 * hh, min(0.7 * hh, iy))
    n = rng.randint(6, 8)
    step = 2 * math.pi / n
    base = rng.uniform(0, 2 * math.pi)
    angs = sorted(base + k * step + rng.uniform(-0.2, 0.2) * step for k in range(n))
    gaps = [angs[(k + 1) % n] - angs[k] + (2 * math.pi if k == n - 1 else 0.0) for k in range(n)]
    gmin = max(min(gaps), 0.3)
    rays = []
    for th in angs:
        c, s = math.cos(th), math.sin(th)
        dx = (hw - ix) / c if c > 1e-6 else ((-hw - ix) / c if c < -1e-6 else 1e9)
        dy = (hh - iy) / s if s > 1e-6 else ((-hh - iy) / s if s < -1e-6 else 1e9)
        rays.append((th, c, s, min(dx, dy)))
    t = max(2.2, min(7.0, min(hw, hh) * 0.28))               # thicker slivers for bigger panes (min: still collides sensibly)
    rin = max(3.0, t / (0.75 * 2 * math.sin(gmin / 2)))      # where neighbouring slivers are far enough apart
    out = []
    for th, c, s, d in rays:
        if d - rin < 2.5:
            continue
        mid = 0.5 * (rin + d)
        out.append(dict(x=ix + c * mid, y=iy + s * mid, hw=0.5 * (d - rin), hh=t / 2, a=th, grain=None))
    if out:                                                  # a small chip where the pane was hit
        hub = min(rin * 0.62, hw - abs(ix) - 0.3, hh - abs(iy) - 0.3)
        if hub >= 1.5:
            out.append(dict(x=ix, y=iy, hw=hub, hh=hub, a=0.0, grain=None))
    return out


def _rect_adjacency(rects, eps=1.0, min_overlap=1.5):
    """Which axis-aligned fragments share a face: (i, j, point, width, normal_i_to_j, edge_axis) in the parent's
    frame.  edge_axis is the direction the shared edge runs in ('x' or 'y')."""
    out = []
    for i in range(len(rects)):
        a = rects[i]
        for j in range(i + 1, len(rects)):
            b = rects[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            gx = abs(dx) - (a["hw"] + b["hw"])
            gy = abs(dy) - (a["hh"] + b["hh"])
            lo_y = max(a["y"] - a["hh"], b["y"] - b["hh"])
            hi_y = min(a["y"] + a["hh"], b["y"] + b["hh"])
            lo_x = max(a["x"] - a["hw"], b["x"] - b["hw"])
            hi_x = min(a["x"] + a["hw"], b["x"] + b["hw"])
            if abs(gx) <= eps and hi_y - lo_y >= min_overlap and dx != 0:
                sg = 1.0 if dx > 0 else -1.0
                xm = 0.5 * ((a["x"] + sg * a["hw"]) + (b["x"] - sg * b["hw"]))
                out.append((i, j, (xm, 0.5 * (lo_y + hi_y)), hi_y - lo_y, (sg, 0.0), "y"))
            elif abs(gy) <= eps and hi_x - lo_x >= min_overlap and dy != 0:
                sg = 1.0 if dy > 0 else -1.0
                ym = 0.5 * ((a["y"] + sg * a["hh"]) + (b["y"] - sg * b["hh"]))
                out.append((i, j, (0.5 * (lo_x + hi_x), ym), hi_x - lo_x, (0.0, sg), "x"))
    return out


def build_wall(world, x, y_bottom, cols, rows, brick_w=40.0, brick_h=20.0, material="brick", bond=True):
    """A brick (or stone, concrete ...) wall in running bond, standing on y = y_bottom and centred on x.  Alternate
    rows are shifted half a brick, with half bricks at the ends.  Returns the bodies, bottom row first; join them with
    gpu.weld_joints(bodies + [ground], material="mortar")."""
    bodies = []
    x0 = x - cols * brick_w / 2
    for r in range(rows):
        cy = y_bottom - brick_h / 2 - r * brick_h
        if bond and r % 2:
            parts = [(x0 + brick_w / 4, brick_w / 4)]
            parts += [(x0 + brick_w + i * brick_w, brick_w / 2) for i in range(cols - 1)]
            parts += [(x0 + cols * brick_w - brick_w / 4, brick_w / 4)]
        else:
            parts = [(x0 + brick_w / 2 + i * brick_w, brick_w / 2) for i in range(cols)]
        for cx, hw in parts:
            bodies.append(world.add(Body(BOX, cx, cy, hw=hw, hh=brick_h / 2, material=material, style=material)))
    return bodies


# ----------------------------------------------------------------------------
# the backend object
# ----------------------------------------------------------------------------
def cuda_available():
    try:
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


class WarpPhysics:
    """Owns the GPU arrays and runs fixed steps on the device.

    `device` is a Warp device name; the default is the first NVIDIA GPU ("cuda:0") and it is an
    error if there is none.
    """

    def __init__(self, device=None, iterations=ITERATIONS, use_graph=True):
        wp.config.quiet = True
        wp.init()
        if device is None:
            if wp.get_cuda_device_count() == 0:
                raise RuntimeError("No NVIDIA CUDA GPU found (this engine runs only on an NVIDIA GPU)")
            device = "cuda:0"
        self.device = wp.get_device(device)
        self.iterations = iterations
        d = self.device
        f = lambda: wp.zeros(CAP, dtype=float, device=d)
        i32 = lambda: wp.zeros(CAP, dtype=int, device=d)
        # --- persistent per-body state (stays on the GPU) ---
        self.px, self.py, self.ang = f(), f(), f()
        self.cosa, self.sina = wp.ones(CAP, dtype=float, device=d), f()
        self.vx, self.vy, self.w = f(), f(), f()
        self.hw, self.hh, self.rad = f(), f(), f()
        self.inv_m, self.inv_i = f(), f()
        self.kind, self.static = i32(), i32()
        self.mu, self.rest = f(), f()
        self.asleep, self.sleep_t, self.nosleep = i32(), f(), i32()
        self.active, self.noglue, self.is_tunable = i32(), i32(), i32()
        # --- per-pair persistent state: cement and warm starting ---
        self.glue_born = wp.full(CAP * CAP, -1.0, dtype=float, device=d)
        self.glue_str = wp.full(CAP * CAP, 1.0, dtype=float, device=d)      # joint strength multiplier per pair
        self.ke = wp.zeros(CAP, dtype=float, device=d)                      # kinetic energy per body (kinetic_energy())
        self.ke_total = wp.zeros(1, dtype=float, device=d)
        self.peak = wp.zeros(CAP, dtype=float, device=d)                    # hardest shock per body since the last readback
        self.peak_host = np.zeros(CAP, dtype=np.float32)
        self.h_peak = wp.zeros(CAP, dtype=float, device="cpu", pinned=d.is_cuda)
        self.warm_pn = wp.zeros(CAP * CAP * 2, dtype=float, device=d)
        self.warm_pt = wp.zeros(CAP * CAP * 2, dtype=float, device=d)
        # --- per-step scratch: contacts ---
        self.ccount = wp.zeros(1, dtype=int, device=d)
        fc = lambda: wp.zeros(MAXC, dtype=float, device=d)
        ic = lambda: wp.zeros(MAXC, dtype=int, device=d)
        self.c_a, self.c_b, self.c_key = ic(), ic(), ic()
        self.c_nx, self.c_ny = fc(), fc()
        self.c_rax, self.c_ray, self.c_rbx, self.c_rby = fc(), fc(), fc(), fc()
        self.c_mn, self.c_mt, self.c_bias = fc(), fc(), fc()
        self.c_pn, self.c_pt, self.c_mu, self.c_coh = fc(), fc(), fc(), fc()
        self.c_color, self.c_prio, self.c_try = ic(), ic(), ic()
        self.used = i32()
        self.prop = wp.zeros(CAP * 32, dtype=int, device=d)
        self.ncolors = 24
        self.blocked = i32()
        # --- misc ---
        self.clock = wp.zeros(1, dtype=float, device=d)
        self.params = wp.zeros(NPARAMS, dtype=float, device=d)
        self.gflag, self.gcount = i32(), wp.zeros(1, dtype=int, device=d)
        self.host_params = np.zeros(NPARAMS, dtype=np.float32)
        self.host_params[P_TETHER] = -1
        self.host_params[P_CEMENT] = 3.0e6
        self.params.assign(self.host_params)
        pin = d.is_cuda
        self.h_px = wp.zeros(CAP, dtype=float, device="cpu", pinned=pin)
        self.h_py = wp.zeros(CAP, dtype=float, device="cpu", pinned=pin)
        self.h_ang = wp.zeros(CAP, dtype=float, device="cpu", pinned=pin)
        self.h_asleep = wp.zeros(CAP, dtype=int, device="cpu", pinned=pin)
        self.h_active = wp.zeros(CAP, dtype=int, device="cpu", pinned=pin)
        self.h_gflag = wp.zeros(CAP, dtype=int, device="cpu", pinned=pin)
        self.h_gcount = wp.zeros(1, dtype=int, device="cpu", pinned=pin)
        self.h_ccount = wp.zeros(1, dtype=int, device="cpu", pinned=pin)
        self.slot_of = {}          # python Body -> slot
        self.body_of = {}          # slot -> python Body
        self.free = []
        self.tether_slot = -1
        self.grabbed = -1
        self.graph = None
        self.use_graph = use_graph and d.is_cuda
        self.dt = 1.0 / 240.0
        self.bundle = BUNDLE
        # CUDA event pairs around each bundle: measures the time the GPU really spent on the physics
        # kernels.  Results are read a few frames later, once the events have completed, so the
        # display never forces a synchronisation.
        self._events = ([(wp.Event(d, enable_timing=True), wp.Event(d, enable_timing=True)) for _ in range(8)]
                        if d.is_cuda else [])
        self._ev_next = 0
        self._ev_pending = []
        self._timings = []
        self.glue_count = 0
        self._joint_init()
        self.contact_count = 0

    # ---- description -----------------------------------------------------
    def describe(self):
        return f"Warp {wp.__version__} on {self.device.name} ({self.device})"

    # ---- loading bodies --------------------------------------------------
    def load(self, bodies, tether=None, noglue=(), dt=1.0 / 240.0):
        """Upload a whole scene: `bodies` is a list of python Body objects (statics first)."""
        self.dt = dt
        self.slot_of.clear()
        self.body_of.clear()
        self.free = []
        n = len(bodies)
        if n > CAP:
            raise ValueError(f"too many bodies for the GPU arrays ({n} > {CAP})")
        h = {k: np.zeros(CAP, dtype=np.float32) for k in
             ("px", "py", "ang", "cos", "sin", "vx", "vy", "w", "hw", "hh", "rad", "im", "ii", "mu", "e", "st")}
        h["cos"][:] = 1.0
        hi = {k: np.zeros(CAP, dtype=np.int32) for k in
              ("kind", "static", "asleep", "nosleep", "active", "noglue", "tunable")}
        for s, b in enumerate(bodies):
            self.slot_of[b] = s
            self.body_of[s] = b
            self._fill(h, hi, s, b)
            if b in noglue:
                hi["noglue"][s] = 1
        self.free = list(range(CAP - 1, n - 1, -1))
        self._upload(h, hi)
        self.glue_born.fill_(-1.0)
        self.glue_str.fill_(1.0)
        self.peak.zero_()
        self.warm_pn.zero_()
        self.warm_pt.zero_()
        self.clock.zero_()
        self.tether_slot = self.slot_of[tether] if tether is not None else -1
        self.host_params[P_TETHER] = self.tether_slot
        self.params.assign(self.host_params)
        self._joint_init()
        self.graph = None

    @staticmethod
    def _fill(h, hi, s, b):
        h["px"][s], h["py"][s], h["ang"][s] = b.x, b.y, b.a
        h["cos"][s], h["sin"][s] = b.c, b.s
        h["vx"][s], h["vy"][s], h["w"][s] = b.vx, b.vy, b.w
        h["hw"][s], h["hh"][s], h["rad"][s] = b.hw, b.hh, b.r
        h["im"][s], h["ii"][s] = b.im, b.iI
        h["mu"][s], h["e"][s] = b.mu, b.e
        hi["kind"][s] = b.kind
        hi["static"][s] = 1 if b.static else 0
        hi["asleep"][s] = 1 if b.asleep else 0
        hi["nosleep"][s] = 1 if b.nosleep else 0
        hi["active"][s] = 1
        hi["tunable"][s] = 1 if getattr(b, "tunable", False) else 0

    def _upload(self, h, hi):
        self.px.assign(h["px"])
        self.py.assign(h["py"])
        self.ang.assign(h["ang"])
        self.cosa.assign(h["cos"])
        self.sina.assign(h["sin"])
        self.vx.assign(h["vx"])
        self.vy.assign(h["vy"])
        self.w.assign(h["w"])
        self.hw.assign(h["hw"])
        self.hh.assign(h["hh"])
        self.rad.assign(h["rad"])
        self.inv_m.assign(h["im"])
        self.inv_i.assign(h["ii"])
        self.mu.assign(h["mu"])
        self.rest.assign(h["e"])
        self.kind.assign(hi["kind"])
        self.static.assign(hi["static"])
        self.asleep.assign(hi["asleep"])
        self.nosleep.assign(hi["nosleep"])
        self.active.assign(hi["active"])
        self.noglue.assign(hi["noglue"])
        self.is_tunable.assign(hi["tunable"])
        self.sleep_t.zero_()

    def add_body(self, b):
        """Put one new body into a free slot (add a body while the simulation is running)."""
        if not self.free:
            return None
        s = self.free.pop()
        self.slot_of[b] = s
        self.slot_version += 1
        self.body_of[s] = b
        h = {k: np.zeros(1, dtype=np.float32) for k in
             ("px", "py", "ang", "cos", "sin", "vx", "vy", "w", "hw", "hh", "rad", "im", "ii", "mu", "e", "st")}
        hi = {k: np.zeros(1, dtype=np.int32) for k in
              ("kind", "static", "asleep", "nosleep", "active", "noglue", "tunable")}
        self._fill(h, hi, 0, b)
        self.wp_set(s, h, hi)
        wp.launch(k_free_slot, dim=CAP, inputs=[s, self.glue_born, self.warm_pn, self.warm_pt, self.glue_str],
                  device=self.device)
        return s

    def wp_set(self, s, h, hi):
        """Write one slot of every body array (small host->device copies)."""
        def put(arr, val):
            wp.copy(arr, wp.array(val, dtype=arr.dtype, device=self.device), dest_offset=s, count=1)
        put(self.px, h["px"]); put(self.py, h["py"]); put(self.ang, h["ang"])
        put(self.cosa, h["cos"]); put(self.sina, h["sin"])
        put(self.vx, h["vx"]); put(self.vy, h["vy"]); put(self.w, h["w"])
        put(self.hw, h["hw"]); put(self.hh, h["hh"]); put(self.rad, h["rad"])
        put(self.inv_m, h["im"]); put(self.inv_i, h["ii"])
        put(self.mu, h["mu"]); put(self.rest, h["e"])
        put(self.kind, hi["kind"]); put(self.static, hi["static"])
        put(self.asleep, hi["asleep"]); put(self.nosleep, hi["nosleep"])
        put(self.active, hi["active"]); put(self.noglue, hi["noglue"]); put(self.is_tunable, hi["tunable"])
        z = np.zeros(1, dtype=np.float32)
        put(self.sleep_t, z)

    def remove_body(self, b):
        s = self.slot_of.pop(b, None)
        if s is None:
            return
        self.body_of.pop(s, None)
        for j, m in enumerate(self.joint_meta):
            if m["alive"] and (m["a"] == s or m["b"] == s):
                self._kill_joint(j)
        self._jcol_used.pop(s, None)
        self._frac_immune.pop(b, None)
        self.slot_version += 1
        zi = np.zeros(1, dtype=np.int32)
        for arr in (self.active,):
            wp.copy(arr, wp.array(zi, dtype=int, device=self.device), dest_offset=s, count=1)
        wp.launch(k_free_slot, dim=CAP, inputs=[s, self.glue_born, self.warm_pn, self.warm_pt, self.glue_str],
                  device=self.device)
        wp.launch(k_wake_all, dim=CAP, inputs=[self.active, self.static, self.asleep, self.sleep_t],
                  device=self.device)          # something may have been resting on it
        self.free.append(s)
        if self.grabbed == s:
            self.grabbed = -1

    # ---- inputs ----------------------------------------------------------
    def set_inputs(self, mouse, grab=None, cement=None, rope=None, accel=0.0, gravitation=None):
        """grab = (body, local_x, local_y) or None; rope = (pivot_x, pivot_y, length); accel = sideways
        acceleration of the whole world in px/s^2 (an earthquake or a gale);
        gravitation = G for Newton's attraction between bodies (None keeps the current value, 0 = off)."""
        p = self.host_params
        p[P_AX] = accel
        if gravitation is not None:
            p[P_G] = gravitation
        p[P_MX], p[P_MY] = mouse
        if grab is None:
            p[P_GRAB_ON] = 0.0
            self.grabbed = -1
        else:
            s = self.slot_of.get(grab[0])
            if s is None:
                p[P_GRAB_ON] = 0.0
                self.grabbed = -1
            else:
                p[P_GRAB_ON], p[P_GRAB_BODY], p[P_GRAB_LX], p[P_GRAB_LY] = 1.0, s, grab[1], grab[2]
                self.grabbed = s
        if cement is not None:
            p[P_CEMENT] = cement
        if rope is not None:
            p[P_PIVX], p[P_PIVY], p[P_ROPE] = rope
        self.params.assign(p)

    def kinetic_energy(self, per_body=False):
        """Total kinetic energy 1/2 m v^2 + 1/2 I w^2 of the movable bodies (mass * px^2 / s^2).
        per_body=True also returns a dict {body: energy}.  Reads the live state, so it syncs with the GPU."""
        d = self.device
        self.ke_total.zero_()
        wp.launch(k_kinetic, dim=CAP, inputs=[self.active, self.static, self.inv_m, self.inv_i,
                                              self.vx, self.vy, self.w, self.ke, self.ke_total], device=d)
        wp.synchronize_device(d)
        total = float(self.ke_total.numpy()[0])
        if not per_body:
            return total
        ke = self.ke.numpy()
        return total, {b: float(ke[s]) for b, s in self.slot_of.items()}

    def kick(self, b, dvx, dvy, dw):
        s = self.slot_of.get(b)
        if s is not None:
            wp.launch(k_kick, dim=1, inputs=[s, dvx, dvy, dw, self.vx, self.vy, self.w, self.asleep,
                                             self.sleep_t], device=self.device)

    def set_material(self, mu=None, e=None):
        # mu / e < 0 means "leave alone"; done with two tiny launches over the tunable bodies
        if mu is not None:
            wp.launch(k_set_material, dim=CAP, inputs=[float(mu), -1.0, self.is_tunable, self.mu, self._scratch_f()],
                      device=self.device)
        if e is not None:
            wp.launch(k_set_material, dim=CAP, inputs=[-1.0, float(e), self.is_tunable, self._scratch_f(), self.rest],
                      device=self.device)

    def _scratch_f(self):
        if not hasattr(self, "_scr"):
            self._scr = wp.zeros(CAP, dtype=float, device=self.device)
        return self._scr

    # ---- cement ----------------------------------------------------------
    def weld(self, b, mult=1.0, only_static=False):
        """Join body b to what it touches. mult = joint strength (1 = cement); only_static = anchor it to fixed things."""
        s = self.slot_of.get(b)
        if s is not None:
            wp.launch(k_weld_one, dim=CAP, inputs=[s, float(mult), 1 if only_static else 0, self.clock, self.active,
                                                   self.noglue, self.static, self.kind,
                                                   self.px, self.py, self.cosa, self.sina,
                                                   self.hw, self.hh, self.rad, self.glue_born, self.glue_str],
                      device=self.device)

    def weld_all(self, mult=1.0):
        wp.launch(k_weld_all, dim=CAP * CAP, inputs=[float(mult), self.clock, self.active, self.noglue, self.kind,
                                                     self.px, self.py, self.cosa, self.sina,
                                                     self.hw, self.hh, self.rad, self.glue_born, self.glue_str],
                  device=self.device)

    def clear_glue(self):
        wp.launch(k_clear_glue, dim=CAP * CAP, inputs=[self.glue_born, self.glue_str], device=self.device)

    # ---- stepping --------------------------------------------------------
    def _substep(self):
        d, dt = self.device, self.dt
        wp.launch(k_forces, dim=CAP, inputs=[dt, self.params, self.active, self.static, self.asleep, self.sleep_t,
                                             self.kind, self.vx, self.vy, self.w, self.clock], device=d)
        wp.launch(k_attract, dim=CAP, inputs=[dt, self.params, self.active, self.static, self.asleep, self.sleep_t,
                                              self.px, self.py, self.inv_m, self.vx, self.vy], device=d)
        wp.launch(k_grab, dim=1, inputs=[dt, self.params, self.px, self.py, self.cosa, self.sina,
                                         self.inv_m, self.inv_i, self.vx, self.vy, self.w,
                                         self.asleep, self.sleep_t], device=d)
        wp.launch(k_reset_counts, dim=CAP, inputs=[self.ccount, self.used], device=d)
        wp.launch(k_narrow, dim=CAP * CAP, inputs=[
            dt, self.params, self.clock, self.active, self.static, self.asleep, self.sleep_t, self.kind,
            self.px, self.py, self.cosa, self.sina, self.vx, self.vy, self.w, self.hw, self.hh, self.rad,
            self.inv_m, self.inv_i, self.mu, self.rest, self.glue_born, self.glue_str, self.warm_pn, self.warm_pt,
            self.ccount, self.c_a, self.c_b, self.c_key, self.c_nx, self.c_ny,
            self.c_rax, self.c_ray, self.c_rbx, self.c_rby, self.c_mn, self.c_mt, self.c_bias,
            self.c_pn, self.c_pt, self.c_mu, self.c_coh, self.adj], device=d)
        wp.launch(k_warm_start, dim=MAXC, inputs=[
            self.ccount, self.c_a, self.c_b, self.c_nx, self.c_ny, self.c_rax, self.c_ray, self.c_rbx,
            self.c_rby, self.c_pn, self.c_pt, self.inv_m, self.inv_i, self.vx, self.vy, self.w], device=d)
        # colour the contacts (on the GPU) so independent ones can be solved at the same time
        wp.launch(k_color_init, dim=MAXC, inputs=[self.ccount, self.c_color, self.c_prio, self.c_try], device=d)
        wp.launch(k_color_clear, dim=CAP * 32, inputs=[self.prop], device=d)
        for _ in range(COLOR_ROUNDS):
            wp.launch(k_color_propose, dim=MAXC, inputs=[self.ccount, self.c_a, self.c_b, self.static,
                                                         self.c_color, self.c_prio, self.c_try,
                                                         self.used, self.prop], device=d)
            wp.launch(k_color_commit, dim=MAXC, inputs=[self.ccount, self.c_a, self.c_b, self.static,
                                                        self.c_color, self.c_prio, self.c_try,
                                                        self.used, self.prop], device=d)
            wp.launch(k_color_clear, dim=CAP * 32, inputs=[self.prop], device=d)
        order = list(range(self.ncolors)) + [-1]       # -1: leftovers the colouring didn't finish
        nj = self.nj
        if nj:
            wp.launch(k_joint_warm, dim=nj, inputs=[
                float(JOINT_WARM), self.ja, self.jb, self.jalive, self.jlax, self.jlay, self.jlbx, self.jlby, self.jpx, self.jpy,
                self.jpm, self.cosa, self.sina, self.inv_m, self.inv_i, self.vx, self.vy, self.w,
                self.asleep, self.static, self.sleep_t], device=d)
        for it in range(self.iterations):
            for col in (order if it % 2 == 0 else order[::-1]):    # sweep up, then back down
                wp.launch(k_solve, dim=MAXC, inputs=[
                    col, self.ccount, self.c_color, self.c_a, self.c_b, self.c_nx, self.c_ny,
                    self.c_rax, self.c_ray, self.c_rbx, self.c_rby, self.c_mn, self.c_mt, self.c_bias,
                    self.c_pn, self.c_pt, self.c_mu, self.c_coh, self.inv_m, self.inv_i,
                    self.vx, self.vy, self.w], device=d)
            if nj:
                self._joint_sweep(dt)
        for _ in range(JOINT_EXTRA_SWEEPS if nj else 0):      # joints last: a chain of them needs more sweeps than contacts do
            self._joint_sweep(dt)
        if nj:
            wp.launch(k_joint_check, dim=nj, inputs=[
                dt, self.clock, self.ja, self.jb, self.jalive, self.jnax, self.jnay, self.jtmax, self.jsmax,
                self.jmmax, self.jrate, self.jborn, self.jdmg, self.jpx, self.jpy, self.jpm, self.jT, self.jS,
                self.jM, self.jutil, self.cosa, self.sina, self.asleep, self.adj], device=d)
        wp.launch(k_store_and_break, dim=MAXC, inputs=[
            self.ccount, self.c_key, self.c_a, self.c_b, self.c_pn, self.c_pt, self.c_mu, self.c_coh,
            self.warm_pn, self.warm_pt, self.glue_born], device=d)
        wp.launch(k_peak, dim=MAXC, inputs=[self.ccount, self.c_a, self.c_b, self.c_pn, self.inv_m, self.peak],
                  device=d)
        wp.launch(k_integrate, dim=CAP, inputs=[dt, self.active, self.static, self.asleep, self.nosleep,
                                                self.grabbed, self.px, self.py, self.ang, self.cosa,
                                                self.sina, self.vx, self.vy, self.w, self.sleep_t], device=d)
        wp.launch(k_rope, dim=1, inputs=[self.params, self.px, self.py, self.vx, self.vy], device=d)

    def _bundle(self):
        for _ in range(BUNDLE):
            self._substep()
        d = self.device
        wp.launch(k_sleep_init, dim=CAP, inputs=[self.active, self.static, self.asleep, self.nosleep,
                                                 self.sleep_t, self.blocked], device=d)
        for _ in range(SLEEP_PASSES):
            wp.launch(k_sleep_prop, dim=MAXC, inputs=[self.ccount, self.c_a, self.c_b, self.static,
                                                      self.asleep, self.blocked], device=d)
            if self.nj:
                wp.launch(k_sleep_prop_joints, dim=self.nj, inputs=[self.ja, self.jb, self.jalive, self.static,
                                                                    self.asleep, self.blocked], device=d)
        wp.launch(k_sleep_apply, dim=CAP, inputs=[self.active, self.static, self.asleep, self.blocked,
                                                  self.vx, self.vy, self.w], device=d)

    def step_bundle(self):
        """Advance BUNDLE fixed steps (1/60 s) entirely on the device."""
        if self.use_graph:
            if self.graph is None or self._graph_grab != self.grabbed:
                self._record()
        if self._events:
            if len(self._ev_pending) >= len(self._events):      # never re-record a pending pair
                self._ev_pending.pop(0)
            pair = self._events[self._ev_next]
            self._ev_next = (self._ev_next + 1) % len(self._events)
            wp.record_event(pair[0])
            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self._bundle()
            wp.record_event(pair[1])
            self._ev_pending.append(pair)
        else:                                                    # CPU device: calls are synchronous
            t0 = time.perf_counter()
            self._bundle()
            self._timings.append((time.perf_counter() - t0) * 1000.0)

    def take_timing(self):
        """GPU milliseconds spent per bundle (1/60 s of simulation) for every bundle that has
        finished since the last call.  Never blocks."""
        while self._ev_pending and self._ev_pending[0][1].is_complete:
            s, e = self._ev_pending.pop(0)
            self._timings.append(wp.get_event_elapsed_time(s, e, synchronize=False))
        out, self._timings = self._timings, []
        return out

    def _record(self):
        # `grabbed` is baked into the graph (it only feeds the sleep timer), so re-record when it changes
        with wp.ScopedCapture(device=self.device) as cap:
            self._bundle()
        self.graph = cap.graph
        self._graph_grab = self.grabbed

    # ---- reading back ----------------------------------------------------
    def readback(self):
        """Copy what the renderer needs to the host: positions, angles, sleeping, cement flags."""
        d = self.device
        wp.launch(k_clear_flags, dim=CAP, inputs=[self.gflag, self.gcount], device=d)
        wp.launch(k_glue_flags, dim=CAP * CAP, inputs=[self.clock, self.glue_born, self.gflag, self.gcount],
                  device=d)
        for src, dst in ((self.px, self.h_px), (self.py, self.h_py), (self.ang, self.h_ang),
                         (self.asleep, self.h_asleep), (self.active, self.h_active),
                         (self.gflag, self.h_gflag), (self.gcount, self.h_gcount), (self.ccount, self.h_ccount)):
            wp.copy(dst, src)
        wp.copy(self.h_peak, self.peak)
        self.peak.zero_()
        wp.synchronize_device(d)
        self.peak_host = self.h_peak.numpy().copy()
        self.glue_count = int(self.h_gcount.numpy()[0])
        self.contact_count = int(self.h_ccount.numpy()[0])
        self._joint_readback()
        return (self.h_px.numpy(), self.h_py.numpy(), self.h_ang.numpy(), self.h_asleep.numpy(),
                self.h_active.numpy(), self.h_gflag.numpy())


    # ---- anchored joints -------------------------------------------------------------------------------
    def _joint_init(self):
        d = self.device
        f = lambda: wp.zeros(MAXJ, dtype=float, device=d)
        i = lambda: wp.zeros(MAXJ, dtype=int, device=d)
        self.ja, self.jb, self.jcolor, self.jalive = i(), i(), i(), i()
        self.jlax, self.jlay, self.jlbx, self.jlby, self.jref = f(), f(), f(), f(), f()
        self.jnax, self.jnay = f(), f()
        self.jtmax, self.jsmax, self.jmmax, self.jrate, self.jborn = f(), f(), f(), f(), f()
        self.jdmg, self.jpx, self.jpy, self.jpm = f(), f(), f(), f()
        self.jT, self.jS, self.jM, self.jutil = f(), f(), f(), f()
        self.adj = wp.full(CAP * MAXDEG, -1, dtype=int, device=d)      # per body: the bodies it is jointed to
        self.nj = 0
        self.joint_meta = []                 # host record of every joint (slot a/b, limits, alive...)
        self._jcol_used = {}                 # body slot -> set of joint colours in use on it (fixed bodies excluded)
        self.ncolors_j = 0
        self.h_jalive = wp.zeros(MAXJ, dtype=int, device="cpu", pinned=d.is_cuda)
        self.h_jutil = wp.zeros(MAXJ, dtype=float, device="cpu", pinned=d.is_cuda)
        self.h_jload = wp.zeros(MAXJ * 3, dtype=float, device="cpu", pinned=d.is_cuda)
        self.joint_util_host = np.zeros(MAXJ, dtype=np.float32)
        self.broken_joints = []              # events found by the last readback (see take_broken_joints)
        self._jfree = []                     # slots of joints that are gone: add_joint reuses them
        self._jpair = {}                     # frozenset(body slots) -> joint id, for the live joints
        self._bulk, self._bulk_t0, self.progress_cb = None, 0.0, None   # bulk joint upload (weld_joints)
        self._pose_cache = None              # while fracture_step runs: poses of the bodies (saves GPU reads)
        self._frac_immune = {}               # fragment Body -> time until it cannot break again
        self._frac_rng = random.Random(7)
        self.slot_version = getattr(self, "slot_version", 0) + 1     # changes whenever a body is added or removed
        self.joint_scale = 1.0               # every joint's strength is multiplied by this (set_joint_scale)
        self.fracture_scale = 1.0            # every material's fracture limit is multiplied by this
        self.fractures = []

    def _joint_sweep(self, dt):
        for col in range(self.ncolors_j):
            wp.launch(k_joint_solve, dim=self.nj, inputs=[
                col, 1.0 / dt, self.ja, self.jb, self.jcolor, self.jalive, self.jlax, self.jlay, self.jlbx,
                self.jlby, self.jref, self.jpx, self.jpy, self.jpm, self.px, self.py, self.ang, self.cosa,
                self.sina, self.inv_m, self.inv_i, self.vx, self.vy, self.w, self.asleep], device=self.device)

    def _pose(self, s):
        if self._pose_cache is not None and s in self._pose_cache:
            return self._pose_cache[s]
        return float(self.px.numpy()[s]), float(self.py.numpy()[s]), float(self.ang.numpy()[s])

    def _bulk_flush(self):
        """Send the joints staged during a bulk weld to the device: one copy per array instead of one per joint."""
        bulk, self._bulk = self._bulk, None
        for arr, js, vals in (bulk or {}).values():
            h = arr.numpy()
            h[np.array(js)] = np.array(vals, dtype=h.dtype)
            wp.copy(arr, wp.array(h, dtype=arr.dtype, device=self.device))

    def _put(self, arr, j, val):
        wp.copy(arr, wp.array(np.array([val], dtype=np.float32 if arr.dtype == wp.float32 else np.int32),
                              dtype=arr.dtype, device=self.device), dest_offset=j, count=1)

    def add_joint(self, a, b, point, tmax, smax, mmax, fatigue=0.0, axis=None, label="", _raw=False):
        """Tie body a to body b at the world point `point` (anchor), locking their relative angle.
        tmax / smax / mmax: tension, shear and moment limits (force, force, force*px).  fatigue: wear per second at full
        load (0 = brittle: no wear).  axis: the joint direction from a toward b (default: centre to centre).
        Returns the joint id.  Both bodies stop colliding with each other while the joint holds."""
        if self.nj >= MAXJ and not self._jfree:
            raise RuntimeError(f"too many joints ({MAXJ})")
        if not _raw:                                    # the 'joints x' setting (set_joint_scale)
            tmax, smax, mmax = tmax * self.joint_scale, smax * self.joint_scale, mmax * self.joint_scale
        sa, sb = self.slot_of[a], self.slot_of[b]
        xa, ya, aa = self._pose(sa)
        xb, yb, ab = self._pose(sb)
        px_, py_ = float(point[0]), float(point[1])

        def local(x, y, ang):
            dx, dy = px_ - x, py_ - y
            c, s = math.cos(ang), math.sin(ang)
            return dx * c + dy * s, -dx * s + dy * c
        lax, lay = local(xa, ya, aa)
        lbx, lby = local(xb, yb, ab)
        if axis is None:
            axis = (xb - xa, yb - ya)
        n = math.hypot(axis[0], axis[1]) or 1.0
        wx, wy = axis[0] / n, axis[1] / n
        ca, sa_ = math.cos(aa), math.sin(aa)
        nax, nay = wx * ca + wy * sa_, -wx * sa_ + wy * ca              # the axis in A's own frame
        # adjacency list: find a free place in each body's row (breaks on the GPU free places there)
        adj = self.adj.numpy().copy()
        for s_, o in ((sa, sb), (sb, sa)):
            row = adj[s_ * MAXDEG:(s_ + 1) * MAXDEG]
            if o in row:
                continue
            free = np.nonzero(row == -1)[0]
            if not len(free):
                raise RuntimeError("a body has too many joints (MAXDEG)")
            adj[s_ * MAXDEG + free[0]] = o
        self.adj.assign(adj)
        # colour: the lowest colour free on both (fixed bodies never conflict, as in the contact solver)
        used = set()
        for s_, bd in ((sa, a), (sb, b)):
            if not bd.static:
                used |= self._jcol_used.setdefault(s_, set())
        col = 0
        while col in used:
            col += 1
        if col >= JOINT_COLORS:
            raise RuntimeError("joint colouring ran out of colours")
        for s_, bd in ((sa, a), (sb, b)):
            if not bd.static:
                self._jcol_used.setdefault(s_, set()).add(col)
        self.ncolors_j = max(self.ncolors_j, col + 1)
        if self._jfree:
            j = self._jfree.pop()
        else:
            j = self.nj
            self.nj += 1
        t0 = self._bulk_t0 if self._bulk is not None else float(self.clock.numpy()[0])
        for arr, val in ((self.ja, sa), (self.jb, sb), (self.jcolor, col), (self.jalive, 1),
                         (self.jlax, lax), (self.jlay, lay), (self.jlbx, lbx), (self.jlby, lby),
                         (self.jref, ab - aa), (self.jnax, nax), (self.jnay, nay), (self.jtmax, tmax),
                         (self.jsmax, smax), (self.jmmax, mmax), (self.jrate, fatigue), (self.jborn, t0),
                         (self.jdmg, 0.0), (self.jpx, 0.0), (self.jpy, 0.0), (self.jpm, 0.0), (self.jutil, 0.0),
                         (self.jT, 0.0), (self.jS, 0.0), (self.jM, 0.0)):
            if self._bulk is not None:
                ent = self._bulk.setdefault(id(arr), (arr, [], []))
                ent[1].append(j)
                ent[2].append(val)
            else:
                self._put(arr, j, val)
        meta = dict(a=sa, b=sb, body_a=a, body_b=b, alive=True, tmax=tmax, smax=smax, mmax=mmax, label=label,
                    point=(px_, py_), col=col, rate=fatigue, axis=(wx, wy))
        if j < len(self.joint_meta):
            self.joint_meta[j] = meta
        else:
            self.joint_meta.append(meta)
        self._jpair[frozenset((sa, sb))] = j
        self.graph = None                       # the solve loop has a new shape: record it again
        return j

    @staticmethod
    def _extent(shape, ux, uy):
        """Half-width of a body (x, y, angle, Body) along the unit vector (ux, uy)."""
        x, y, ang, bd = shape
        if bd.kind == CIRCLE:
            return bd.r
        c, s = math.cos(ang), math.sin(ang)
        return abs(ux * c + uy * s) * bd.hw + abs(-ux * s + uy * c) * bd.hh

    def weld_joints(self, bodies=None, material="weld", margin=3.0, scale=1.0, fatigue=None):
        """Joint every touching pair among `bodies` (default: all) at the middle of the face where they meet.
        Strength comes from JOINT_MATERIALS and from the width of the contact (a wider face holds more, and resists
        bending as width^2).  A body that is fixed (the ground) gets joints too.  Returns the new joint ids.
        (The face is found along the line between the centres, which is exact for boxes stacked square to each other.)"""
        if self._bulk is not None:
            return self._weld_joints(bodies, material, margin, scale, fatigue)
        self._bulk, self._bulk_t0 = {}, float(self.clock.numpy()[0])
        own_cache = self._pose_cache is None
        if own_cache:
            px_, py_, an_ = self.px.numpy(), self.py.numpy(), self.ang.numpy()
            self._pose_cache = {s_: (float(px_[s_]), float(py_[s_]), float(an_[s_])) for s_ in self.body_of}
        try:
            return self._weld_joints(bodies, material, margin, scale, fatigue)
        finally:
            if own_cache:
                self._pose_cache = None
            self._bulk_flush()

    def _weld_joints(self, bodies, material, margin, scale, fatigue):
        m = JOINT_MATERIALS[material]
        bodies = list(self.slot_of) if bodies is None else list(bodies)
        pose = {bd: (*self._pose(self.slot_of[bd]), bd) for bd in bodies}
        reach = {bd: (bd.r if bd.kind == CIRCLE else math.hypot(bd.hw, bd.hh)) for bd in bodies}
        out = []
        cb = self.progress_cb
        for i, a in enumerate(bodies):
            if cb is not None and i % 40 == 0:
                cb()
            for b in bodies[i + 1:]:
                if a.static and b.static:
                    continue
                sa, sb = self.slot_of[a], self.slot_of[b]
                if frozenset((sa, sb)) in self._jpair:
                    continue
                A, B = pose[a], pose[b]
                if abs(A[0] - B[0]) > reach[a] + reach[b] + margin or abs(A[1] - B[1]) > reach[a] + reach[b] + margin:
                    continue
                face = self._parallel_face(A, B, margin)
                if face is False:
                    continue
                if face is not None:
                    pt, width, nrm = face
                    width *= scale
                    out.append(self.add_joint(a, b, pt, m["T"] * width, m["S"] * width, m["M"] * width * width,
                                              m["fatigue"] if fatigue is None else fatigue, axis=nrm, label=material))
                    continue
                dx, dy = B[0] - A[0], B[1] - A[1]
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    continue
                ux, uy = dx / dist, dy / dist
                tx, ty = -uy, ux
                gap = dist - self._extent(A, ux, uy) - self._extent(B, ux, uy)
                if gap > margin:
                    continue
                ea, eb = self._extent(A, tx, ty), self._extent(B, tx, ty)
                ca, cb = A[0] * tx + A[1] * ty, B[0] * tx + B[1] * ty
                lo, hi = max(ca - ea, cb - eb), min(ca + ea, cb + eb)
                if hi - lo < 1.0:
                    continue
                d_face = ((A[0] * ux + A[1] * uy + self._extent(A, ux, uy)) +
                          (B[0] * ux + B[1] * uy - self._extent(B, ux, uy))) * 0.5
                t_mid = 0.5 * (lo + hi)
                pt = (ux * d_face + tx * t_mid, uy * d_face + ty * t_mid)
                width = (hi - lo) * scale
                out.append(self.add_joint(a, b, pt, m["T"] * width, m["S"] * width, m["M"] * width * width,
                                          m["fatigue"] if fatigue is None else fatigue, axis=(ux, uy),
                                          label=material))
        return out

    @staticmethod
    def _parallel_face(A, B, margin):
        """Exact meeting face of two boxes with parallel sides (any offset, e.g. bricks in running bond):
        (point, width, normal from A to B), False if they do not touch, None if the pair is not parallel boxes."""
        if A[3].kind != BOX or B[3].kind != BOX:
            return None
        d = (B[2] - A[2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
        if abs(d) > 0.02:
            return None
        ca, sa = math.cos(A[2]), math.sin(A[2])
        rx, ry = B[0] - A[0], B[1] - A[1]
        dx, dy = rx * ca + ry * sa, -rx * sa + ry * ca
        hwa, hha = A[3].hw, A[3].hh
        quarter = abs(math.sin(B[2] - A[2])) > 0.7                    # B turned a quarter: its sides swap
        hwb, hhb = (B[3].hh, B[3].hw) if quarter else (B[3].hw, B[3].hh)
        gx, gy = abs(dx) - (hwa + hwb), abs(dy) - (hha + hhb)
        best = None
        if abs(gx) <= margin and dx != 0:
            lo, hi = max(-hha, dy - hhb), min(hha, dy + hhb)
            if hi - lo >= 1.0:
                sg = 1.0 if dx > 0 else -1.0
                best = (abs(gx), ((sg * hwa + dx - sg * hwb) / 2, (lo + hi) / 2), hi - lo, (sg, 0.0))
        if abs(gy) <= margin and dy != 0 and (best is None or abs(gy) < best[0]):
            lo, hi = max(-hwa, dx - hwb), min(hwa, dx + hwb)
            if hi - lo >= 1.0:
                sg = 1.0 if dy > 0 else -1.0
                best = (abs(gy), ((lo + hi) / 2, (sg * hha + dy - sg * hhb) / 2), hi - lo, (0.0, sg))
        if best is None:
            return False
        _, (lx, ly), width, (nx, ny) = best
        return ((A[0] + lx * ca - ly * sa, A[1] + lx * sa + ly * ca), width, (nx * ca - ny * sa, nx * sa + ny * ca))

    def _kill_joint(self, j):
        """Break joint j from the host (used when a body is removed)."""
        mt = self.joint_meta[j]
        if not mt["alive"]:
            return
        mt["alive"] = False
        self._put(self.jalive, j, 0)
        self._release_joint(j)
        adj = self.adj.numpy().copy()
        for s_, o in ((mt["a"], mt["b"]), (mt["b"], mt["a"])):
            row = adj[s_ * MAXDEG:(s_ + 1) * MAXDEG]
            row[row == o] = -1
        self.adj.assign(adj)

    def _joint_readback(self):
        """Fetch joint state; any joint that has snapped since the last call becomes an entry of broken_joints."""
        self.broken_joints = []
        if not self.nj:
            return
        n = self.nj
        wp.copy(self.h_jalive, self.jalive, count=n)
        wp.copy(self.h_jutil, self.jutil, count=n)
        wp.synchronize_device(self.device)
        alive = self.h_jalive.numpy()[:n]
        self.joint_util_host = self.h_jutil.numpy().copy()
        T, S, M = self.jT.numpy()[:n], self.jS.numpy()[:n], self.jM.numpy()[:n]
        for j in np.nonzero(alive == 0)[0]:
            j = int(j)
            mt = self.joint_meta[j]
            if mt["alive"]:
                mt["alive"] = False
                self._release_joint(j)
                self.broken_joints.append(dict(
                    id=j, a=mt["body_a"], b=mt["body_b"], label=mt["label"], point=mt["point"],
                    tension=float(T[j]), shear=float(S[j]), moment=float(M[j]),
                    tension_ratio=max(float(T[j]), 0.0) / mt["tmax"], shear_ratio=float(S[j]) / mt["smax"],
                    moment_ratio=float(M[j]) / mt["mmax"], time=self.sim_time()))
        self.jutil.zero_()

    def take_broken_joints(self):
        """Joints that snapped since the last readback(): each says which bodies, where, and which load did it
        (tension / shear / moment, as a fraction of its limit).  This is what the state manager should perceive."""
        out, self.broken_joints = self.broken_joints, []
        return out

    def joint_loads(self):
        """(alive, load) per joint as of the last readback: load is the worst (T/Tmax)^2+(S/Smax)^2+(M/Mmax)^2 since
        then, so 1.0 means it was at its limit.  For colouring joints by how stressed they are."""
        return np.array([m["alive"] for m in self.joint_meta], bool), self.joint_util_host[:self.nj].copy()

    # ---- fracture: bodies that break --------------------------------------------------------------------
    def _release_joint(self, j):
        """Host bookkeeping for a joint that is gone (snapped on the GPU or killed here): free its colour and slot."""
        mt = self.joint_meta[j]
        if mt.get("released"):
            return
        mt["released"] = True
        for s_ in (mt["a"], mt["b"]):
            st = self._jcol_used.get(s_)
            if st:
                st.discard(mt["col"])
        key = frozenset((mt["a"], mt["b"]))
        if self._jpair.get(key) == j:
            del self._jpair[key]
        self._jfree.append(j)

    def _joints_of(self, body):
        """The live joints touching `body`, as plain records (so they can be handed on to its fragments)."""
        s = self.slot_of[body]
        out = []
        for j, mt in enumerate(self.joint_meta):
            if mt["alive"] and (mt["a"] == s or mt["b"] == s):
                out.append(dict(j=j, parent_is_a=(mt["a"] == s), other=mt["body_b"] if mt["a"] == s else mt["body_a"],
                                point=mt["point"], tmax=mt["tmax"], smax=mt["smax"], mmax=mt["mmax"],
                                rate=mt["rate"], axis=mt["axis"], label=mt["label"]))
        return out

    def fracture_step(self, world, rng=None):
        """Call right after readback().  Every body that has a `material` and took a shock at or over that material's
        limit since the last readback is replaced by fragments (see MATERIALS):
            wood      long splinters along the grain, still tied to each other by weak joints
            glass     slivers fanning out from the impact point, flying free
            concrete / stone   blocky pieces, loosely tied
            brick     two or three chunks, free  (a brick wall fails at its mortar joints first)
        A tough material first only cracks (two halves on a weak joint) and fragments on a harder hit; the pieces
        keep the parent's mass and motion, and the joints that held the parent now hold the nearest fragment.  A piece
        too small to break up becomes dust (it just disappears: that is the event for a particle effect).
        Returns the list of events: dict(kind='crack' | 'shatter' | 'dust', material, parent, fragments, point,
        impact, ratio, time).  Fragments cannot break again for FRAG_IMMUNE seconds."""
        rng = rng or self._frac_rng
        t = self.sim_time()
        for b in [b for b, until in self._frac_immune.items() if until <= t]:
            del self._frac_immune[b]
        peak = self.peak_host
        cand = []
        for slot, b in self.body_of.items():
            mat = getattr(b, "material", None)
            spec = MATERIALS.get(mat) if mat else None
            if not spec or b.static or not spec.get("limit") or b in self._frac_immune:
                continue
            size = b.r * 1.77 if b.kind == CIRCLE else math.sqrt(4 * b.hw * b.hh)
            lim = spec["limit"] * min(4.0, max(1.0, b.ref / max(size, 1.0))) * self.fracture_scale   # smaller pieces are stronger (fewer flaws)
            if peak[slot] >= lim:
                cand.append((float(peak[slot]) / lim, slot, b))
        events = []
        if not cand:
            self.fractures = events
            return events
        cand.sort(key=lambda c: -c[0])
        px, py, ang = self.px.numpy(), self.py.numpy(), self.ang.numpy()
        vx, vy, w = self.vx.numpy(), self.vy.numpy(), self.w.numpy()
        cc = int(self.ccount.numpy()[0])
        con = None
        if cc:
            con = (self.c_a.numpy()[:cc], self.c_b.numpy()[:cc], self.c_pn.numpy()[:cc], self.c_rax.numpy()[:cc],
                   self.c_ray.numpy()[:cc], self.c_rbx.numpy()[:cc], self.c_rby.numpy()[:cc])
        self._pose_cache = {s: (float(px[s]), float(py[s]), float(ang[s])) for s in self.body_of}
        try:
            for ratio, slot, b in cand:
                ev = self._fracture_one(world, b, slot, float(peak[slot]), ratio, (vx, vy, w), con, rng)
                if ev:
                    events.append(ev)
        finally:
            self._pose_cache = None
        self.fractures = events
        return events

    def _fracture_one(self, world, b, slot, impact, ratio, vel, con, rng):
        spec = MATERIALS[b.material]
        X, Y, A = self._pose(slot)
        ca, sa = math.cos(A), math.sin(A)
        t = self.sim_time()
        if b.kind == CIRCLE:
            hw = hh = b.r * 0.886                                # the square with the same area
        else:
            hw, hh = b.hw, b.hh
        # where it was hit: the hardest contact of the last step, in the body's own frame
        ip = None
        if con is not None:
            ca_, cb_, pn, rax, ray, rbx, rby = con
            hit = np.nonzero((ca_ == slot) | (cb_ == slot))[0]
            if len(hit):
                k = hit[int(np.argmax(pn[hit]))]
                ax_, ay_ = (rax[k], ray[k]) if ca_[k] == slot else (rbx[k], rby[k])
                ip = (float(ax_ * ca + ay_ * sa), float(-ax_ * sa + ay_ * ca))
        if ip is None:
            ip = (rng.uniform(-0.3, 0.3) * hw, rng.uniform(-0.3, 0.3) * hh)
        ix, iy = ip
        ix, iy = max(-hw, min(hw, ix)), max(-hh, min(hh, iy))
        wpt = (X + ix * ca - iy * sa, Y + ix * sa + iy * ca)      # impact point in the world
        mp = spec.get("min_piece", 4.0)
        event = dict(material=b.material, parent=b, fragments=[], point=wpt, impact=impact, ratio=ratio, time=t)
        crack = bool(spec.get("crack")) and ratio < spec.get("shatter", 1.0) and 2 * max(hw, hh) >= 4 * mp
        style = spec["style"]
        if 2 * max(hw, hh) < 2.2 * mp or (2 * min(hw, hh) < mp and style != "splinter"):
            rects = []
        elif style == "splinter":
            rects = _frag_splinters(hw, hh, ix, iy, rng, spec, crack)
        elif style == "shards":
            rects = _frag_shards(hw, hh, ix, iy, rng, spec)
            if len(rects) < 3:
                rects = _frag_chunks(hw, hh, ix, iy, rng, dict(spec, chunks=(3, 5)))
        else:
            rects = _frag_chunks(hw, hh, ix, iy, rng, spec, crack)
        if len(rects) < 2 or len(rects) > len(self.free) + 1:
            if len(rects) >= 2:
                return None                                       # no free body slots: leave it whole
            event["kind"] = "dust"
            world.remove(b)
            self.remove_body(b)
            return event
        for r in rects:                                           # leave a hair of room between neighbours
            r["hw"] = max(r["hw"] - FRAG_GAP, 0.8)
            r["hh"] = max(r["hh"] - FRAG_GAP, 0.8)
        joints = self._joints_of(b)
        mass = b.m
        dens = mass / sum(4 * r["hw"] * r["hh"] for r in rects)    # the pieces weigh what the parent did
        vx, vy, w = vel
        v0x, v0y, w0 = float(vx[slot]), float(vy[slot]), float(w[slot])
        world.remove(b)
        self.remove_body(b)
        frags, fpose = [], []
        pop = spec.get("pop", 0.05) * min(impact, 2 * spec["limit"])
        for r in rects:
            wx, wy = X + r["x"] * ca - r["y"] * sa, Y + r["x"] * sa + r["y"] * ca
            rx, ry = wx - X, wy - Y
            fb = Body(BOX, wx, wy, hw=r["hw"], hh=r["hh"], density=dens, mu=b.mu, e=b.e, style=b.style,
                      angle=A + r["a"], material=b.material)
            fb.gen = b.gen + 1
            fb.ref = b.ref
            fb.nosleep, fb.tunable = b.nosleep, b.tunable
            ox, oy = wx - wpt[0], wy - wpt[1]
            od = math.hypot(ox, oy) or 1.0
            kick = pop * rng.uniform(0.5, 1.0)
            fb.vx = v0x - w0 * ry + ox / od * kick
            fb.vy = v0y + w0 * rx + oy / od * kick
            fb.w = w0
            world.add(fb)
            s = self.add_body(fb)
            self._pose_cache[s] = (wx, wy, A + r["a"])
            self._frac_immune[fb] = t + FRAG_IMMUNE
            frags.append(fb)
            fpose.append((wx, wy, A + r["a"], r))
        event["fragments"] = frags
        event["kind"] = "crack" if crack else "shatter"
        # the joints that held the parent now hold whichever fragment is nearest to where they were anchored
        for rec in joints:
            px_, py_ = rec["point"]
            best, bd = 0, 1e18
            for k, (wx, wy, fa, r) in enumerate(fpose):
                dx, dy = px_ - wx, py_ - wy
                c_, s_ = math.cos(fa), math.sin(fa)
                lx, ly = dx * c_ + dy * s_, -dx * s_ + dy * c_
                d = math.hypot(max(abs(lx) - r["hw"], 0.0), max(abs(ly) - r["hh"], 0.0))
                if d < bd:
                    best, bd = k, d
            fr = frags[best]
            a_, b_ = (fr, rec["other"]) if rec["parent_is_a"] else (rec["other"], fr)
            try:
                self.add_joint(a_, b_, rec["point"], rec["tmax"], rec["smax"], rec["mmax"], rec["rate"],
                               axis=rec["axis"], label=rec["label"], _raw=True)
            except RuntimeError:
                pass
        # the fragments hold each other with weaker joints
        jname = spec.get("joint")
        if jname:
            jm = JOINT_MATERIALS[jname]
            aligned = [dict(x=r["x"], y=r["y"], hw=r["hw"], hh=r["hh"]) for r in rects]
            for i, j, pt, width, nrm, edge in _rect_adjacency(aligned):
                if crack:
                    sc, tag = spec["crack_scale"], "crack"
                elif rects[i].get("grain") and edge == rects[i]["grain"]:
                    sc, tag = spec.get("lat_scale", spec["frag_scale"]), "split"
                else:
                    sc, tag = spec["frag_scale"], "frag"
                wp_ = (X + pt[0] * ca - pt[1] * sa, Y + pt[0] * sa + pt[1] * ca)
                nw = (nrm[0] * ca - nrm[1] * sa, nrm[0] * sa + nrm[1] * ca)
                try:
                    self.add_joint(frags[i], frags[j], wp_, jm["T"] * width * sc, jm["S"] * width * sc,
                                   jm["M"] * width * width * sc, jm["fatigue"], axis=nw, label=f"{b.material}-{tag}")
                except RuntimeError:
                    break
        return event

    def set_joint_scale(self, f):
        """Make every joint (and every joint made from now on) f times as strong as it was built: 1 = as built."""
        f = max(float(f), 0.01)
        r = f / self.joint_scale
        self.joint_scale = f
        if abs(r - 1.0) < 1e-6 or not self.nj:
            return
        for arr in (self.jtmax, self.jsmax, self.jmmax):
            a = arr.numpy()
            a[:self.nj] *= r
            arr.assign(a)
        for m in self.joint_meta:
            m["tmax"] *= r
            m["smax"] *= r
            m["mmax"] *= r

    def take_fractures(self):
        """Fracture events of the last fracture_step() (and forget them)."""
        out, self.fractures = self.fractures, []
        return out

    def sim_time(self):
        return float(self.clock.numpy()[0])


# ----------------------------------------------------------------------------
# particles: an endless spawner for benchmarking the GPU
# ----------------------------------------------------------------------------
# Same units and constants as the rigid engine (700 px/s^2 gravity, 1/240 s steps, 4 steps
# per 1/60 s bundle).  Particles are discs; collisions are position based (a few Jacobi
# passes per step, with Coulomb-style friction), neighbours are found with a spatial hash grid
# that is rebuilt on the GPU every step.  The picture is also drawn on the GPU (one thread per
# particle splats pixels into an RGB buffer), so rendering cost is part of what is measured.
PARTICLE_CAP = 2 * 1024 * 1024
MAX_WORLD_W, MAX_WORLD_H = 2560, 1440     # biggest picture (a 1440p screen)
WORLD_CAP_W, WORLD_CAP_H = 12000, 4000    # biggest open area (zoomed far out)
MAX_PEGS = 32
P_ITERS = 3
P_RELAX = 0.7


@wp.kernel
def k_p_spawn(base: int, limit: int, ex: float, ey: float, width: float, r: float, cols: int,
              speed: float, seed: int, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    slot = (base + t) % limit                      # ring buffer: when full, the oldest is replaced
    st = wp.rand_init(seed, t)
    col = t % cols
    row = t // cols
    spacing = width / float(cols)
    px = ex - width * 0.5 + (float(col) + 0.5) * spacing + (wp.randf(st) - 0.5) * 0.2 * r
    py = ey - float(row) * 2.05 * r
    x[slot] = wp.vec3(px, py, 0.0)
    v[slot] = wp.vec3((wp.randf(st) - 0.5) * 30.0, speed + wp.randf(st) * 40.0, 0.0)


@wp.kernel
def k_p_predict(dt: float, n: int, x: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3),
                v: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if i >= n:
        return
    vi = v[i] + wp.vec3(0.0, GRAVITY * dt, 0.0)
    v[i] = vi
    xp[i] = x[i] + vi * dt


@wp.kernel
def k_p_project(grid: wp.uint64, n: int, r: float, mu: float, x: wp.array(dtype=wp.vec3),
                xp: wp.array(dtype=wp.vec3), corr: wp.array(dtype=wp.vec3),
                pegs: wp.array(dtype=wp.vec3), npegs: int):
    """Push overlapping particles apart (each particle applies its own half: no atomics needed)."""
    i = wp.tid()
    if i >= n:
        return
    pi = xp[i]
    rel_i = pi - x[i]
    c = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, pi, 2.0 * r)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j == i:
            continue
        d = pi - xp[j]
        dist = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        pen = 2.0 * r - dist
        if pen > 0.0:
            nrm = wp.vec3(1.0, 0.0, 0.0)
            if dist > 1.0e-6:
                nrm = d / dist
            elif i < j:
                nrm = wp.vec3(-1.0, 0.0, 0.0)
            c = c + nrm * (pen * 0.5)
            rel = rel_i - (xp[j] - x[j])
            tang = rel - nrm * wp.dot(rel, nrm)
            tl = wp.sqrt(tang[0] * tang[0] + tang[1] * tang[1])
            if tl > 1.0e-9:
                c = c - tang * (wp.min(1.0, mu * pen / tl) * 0.5)
    for k in range(npegs):
        pg = pegs[k]
        d = pi - wp.vec3(pg[0], pg[1], 0.0)
        dist = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        pen = pg[2] + r - dist
        if pen > 0.0 and dist > 1.0e-6:
            nrm = d / dist
            c = c + nrm * pen
            tang = rel_i - nrm * wp.dot(rel_i, nrm)
            tl = wp.sqrt(tang[0] * tang[0] + tang[1] * tang[1])
            if tl > 1.0e-9:
                c = c - tang * wp.min(1.0, mu * pen / tl)
    corr[i] = c


@wp.kernel
def k_p_apply(n: int, r: float, mu: float, relax: float, floor_y: float, wall_x: float, x: wp.array(dtype=wp.vec3),
              xp: wp.array(dtype=wp.vec3), corr: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if i >= n:
        return
    c = corr[i] * relax
    m = wp.sqrt(c[0] * c[0] + c[1] * c[1])
    if m > r:
        c = c * (r / m)
    p = xp[i] + c
    if p[0] < r:
        p = wp.vec3(r, p[1], 0.0)
    if p[0] > wall_x - r:
        p = wp.vec3(wall_x - r, p[1], 0.0)
    if p[1] > floor_y - r:
        pen = p[1] - (floor_y - r)
        dxr = p[0] - x[i][0]
        f = wp.min(1.0, mu * pen / (wp.abs(dxr) + 1.0e-6))
        p = wp.vec3(p[0] - dxr * f, floor_y - r, 0.0)
    xp[i] = p


@wp.kernel
def k_p_update(dt: float, n: int, x: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3),
               v: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if i >= n:
        return
    vi = (xp[i] - x[i]) / dt * 0.9998
    sp = wp.sqrt(vi[0] * vi[0] + vi[1] * vi[1])
    if sp < 15.0:
        vi = vi * 0.9                                   # let piles settle instead of creeping
    if sp > MAX_SPEED:
        vi = vi * (MAX_SPEED / sp)
    v[i] = vi
    x[i] = xp[i]


@wp.kernel
def k_p_clear(img: wp.array(dtype=wp.uint8)):
    i = wp.tid()
    q = i % 3
    v = int(0)
    if q == 0:
        v = 14
    elif q == 1:
        v = 17
    else:
        v = 26
    img[i] = wp.uint8(v)


@wp.kernel
def k_p_render(n: int, r: float, sc: float, iw: int, ih: int, cam_x: float, cam_y: float, zoom: float,
               x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), img: wp.array(dtype=wp.uint8)):
    i = wp.tid()
    if i >= n:
        return
    p = x[i]
    vi = v[i]
    s = wp.min(wp.sqrt(vi[0] * vi[0] + vi[1] * vi[1]) / 300.0, 1.0)
    cr = int(70.0 + 185.0 * s)
    cg = int(140.0 + 80.0 * s)
    cb = int(255.0 - 135.0 * s)
    rs = r * zoom * sc
    sx = (p[0] - cam_x) * zoom * sc
    sy = (p[1] - cam_y) * zoom * sc
    if sx < -rs - 1.0 or sy < -rs - 1.0 or sx > float(iw) + rs or sy > float(ih) + rs:
        return                                           # outside the camera view
    ri = int(rs + 0.5)
    if ri < 1:
        ri = 1
    cx = int(sx)
    cy = int(sy)
    for dy in range(-ri + 1, ri):
        for dx in range(-ri + 1, ri):
            if float(dx * dx + dy * dy) <= rs * rs + 0.5:
                px = cx + dx
                py = cy + dy
                if px >= 0 and px < iw and py >= 0 and py < ih:
                    o = (py * iw + px) * 3
                    img[o] = wp.uint8(cr)
                    img[o + 1] = wp.uint8(cg)
                    img[o + 2] = wp.uint8(cb)


class GpuTimer:
    """CUDA-event stopwatch whose results are read later without blocking.
    (On the CPU device it just uses perf_counter.)"""

    def __init__(self, device, slots=8):
        self.device = device
        self.pairs = [(wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True))
                      for _ in range(slots)] if device.is_cuda else []
        self.i = 0
        self.pending = []
        self.done = []
        self.t0 = 0.0
        self.cur = None

    def begin(self):
        if self.pairs:
            if len(self.pending) >= len(self.pairs):
                self.pending.pop(0)
            self.cur = self.pairs[self.i]
            self.i = (self.i + 1) % len(self.pairs)
            wp.record_event(self.cur[0])
        else:
            self.t0 = time.perf_counter()

    def end(self):
        if self.pairs:
            wp.record_event(self.cur[1])
            self.pending.append(self.cur)
        else:
            self.done.append((time.perf_counter() - self.t0) * 1000.0)

    def take(self):
        while self.pending and self.pending[0][1].is_complete:
            s, e = self.pending.pop(0)
            self.done.append(wp.get_event_elapsed_time(s, e, synchronize=False))
        out, self.done = self.done, []
        return out


class GpuSensors:
    """GPU usage, heat, power, clock and memory from NVIDIA's NVML library (the one nvidia-smi uses), read by a small
    background thread twice a second so the frame loop never waits on the driver.  No NVML (or the CPU backend): every
    figure stays None and the graphs say so."""

    def __init__(self, device=None, start=True):
        self.util = self.temp = self.power = self.clock = self.mem_used = self.mem_total = None
        self.name = ""
        self.note = "no NVIDIA sensor"
        self._lib = self._h = None
        self._stop = False
        try:
            if device is not None and not device.is_cuda:
                raise RuntimeError("CPU backend")
            self._open(device)
        except Exception as e:
            self.note = f"no sensor ({type(e).__name__}: {e})"[:60]
            return
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    def _open(self, device):
        import ctypes
        names = ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so"]
        lib = None
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if lib is None:
            raise OSError("NVML not found")
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        h = ctypes.c_void_p()
        bus = getattr(device, "pci_bus_id", None) if device is not None else None
        rc = 1
        if bus:
            rc = lib.nvmlDeviceGetHandleByPciBusId_v2(str(bus).encode(), ctypes.byref(h))
        if rc != 0:
            rc = lib.nvmlDeviceGetHandleByIndex_v2(int(getattr(device, "ordinal", 0) or 0), ctypes.byref(h))
        if rc != 0:
            raise OSError("no NVML device handle")
        self._lib, self._h, self._ct = lib, h, ctypes
        self.note = "NVML"
        self.poll()

    def poll(self):
        """Read every figure once (safe to call from any thread; a failing query just leaves that figure as it was)."""
        ct, lib, h = self._ct, self._lib, self._h

        class Util(ct.Structure):
            _fields_ = [("gpu", ct.c_uint), ("memory", ct.c_uint)]

        class Mem(ct.Structure):
            _fields_ = [("total", ct.c_ulonglong), ("free", ct.c_ulonglong), ("used", ct.c_ulonglong)]

        u, t, p, c, m = Util(), ct.c_uint(), ct.c_uint(), ct.c_uint(), Mem()
        if lib.nvmlDeviceGetUtilizationRates(h, ct.byref(u)) == 0:
            self.util = float(u.gpu)
        if lib.nvmlDeviceGetTemperature(h, 0, ct.byref(t)) == 0:
            self.temp = float(t.value)
        if lib.nvmlDeviceGetPowerUsage(h, ct.byref(p)) == 0:
            self.power = p.value / 1000.0
        if lib.nvmlDeviceGetClockInfo(h, 0, ct.byref(c)) == 0:
            self.clock = float(c.value)
        if lib.nvmlDeviceGetMemoryInfo(h, ct.byref(m)) == 0:
            self.mem_used, self.mem_total = m.used / 1.0e9, m.total / 1.0e9

    def _loop(self):
        while not self._stop:
            try:
                self.poll()
            except Exception:
                pass
            time.sleep(0.5)

    def close(self):
        self._stop = True


LAB_PERF_W = 404                      # width of the performance panel (pixels; a multiple of 4)
LAB_PERF_SAMPLE = 0.1                 # seconds per point on the graphs
LAB_PERF_POINTS = LAB_PERF_W // 2 - 4  # points drawn (2 pixels each): about 20 s of history
LAB_PERF_BG = (13, 16, 21)
LAB_SEG_NAMES = ("step", "pic", "stat", "tk", "flip", "wait")     # where one frame's wall-clock time goes, bottom to top of the stack
LAB_SEG_COLOURS = ((255, 140, 60), (90, 150, 255), (240, 90, 150), (180, 120, 255), (90, 200, 120), (110, 120, 140))


class PerfMonitor:
    """Frame-by-frame performance record and the picture of it.

    frame(dt) is called once per displayed frame with the real time since the previous frame.  sample() folds those
    into one point every 0.1 s.  The panel (draw()) is rasterised with numpy into one small RGB picture, so it costs a
    fraction of a millisecond whatever the history length; the words are drawn by the app as ordinary text items.

      response time   the real time between frames (ms); the line is the average, the dim line the worst frame
      1% lows         the frame rate of the slowest 1% of the last ~1200 frames (and the slowest 0.1%)
      compute         GPU milliseconds spent per displayed frame: physics (orange) + picture (blue)
      usage / heat    the GPU's own utilisation and temperature (NVML)"""

    def __init__(self):
        self.ft = deque(maxlen=1200)
        self.hist = deque(maxlen=LAB_PERF_POINTS)
        self._acc = []
        self.t_sample = time.perf_counter()
        self.sim_s = 0.0               # simulated seconds since the last point
        self.draws = 0                 # pictures drawn since the last point
        self.skip = 30                 # frames to ignore at start-up
        self.seg = {}                  # wall-clock seconds per named part of the frame since the last point
        self.cur = None                # the latest point (a dict) for the text beside the graphs

    def frame(self, dt):
        if self.skip > 0:                  # the first frames compile kernels and capture graphs: not what the graphs are about
            self.skip -= 1
            return
        if 0.0 < dt < 5.0:
            self.ft.append(dt)
            self._acc.append(dt)

    def add(self, name, dt):
        """Credit dt seconds of this frame to a named part (see LAB_SEG_NAMES)."""
        self.seg[name] = self.seg.get(name, 0.0) + dt

    def lows(self):
        """(average fps, 1% low fps, 0.1% low fps) over the recorded frames, or None with too few frames."""
        n = len(self.ft)
        if n < 20:
            return None
        a = np.sort(np.fromiter(self.ft, dtype=np.float64, count=n))[::-1]          # slowest first
        k1, k01 = max(1, int(math.ceil(n * 0.01))), max(1, int(math.ceil(n * 0.001)))
        return n / float(a.sum()), 1.0 / float(a[:k1].mean()), 1.0 / float(a[:k01].mean())

    def sample(self, now, phys_ms, render_ms, sens, speed_ticks=None):
        """Add a point if 0.1 s have passed (returns True when it did).  phys_ms is the cost of a whole 1/60 s of
        physics; render_ms the cost of one picture."""
        if now - self.t_sample < LAB_PERF_SAMPLE:
            return False
        span = now - self.t_sample
        self.t_sample = now
        acc, self._acc = self._acc, []
        n = max(len(acc), 1)
        mean = (sum(acc) / len(acc)) if acc else span * 1000.0 / 1000.0
        worst = max(acc) if acc else span
        lw = self.lows()
        ticks = self.sim_s * 60.0 / n                 # 1/60 s ticks of physics per displayed frame
        phys = (phys_ms or 0.0) * ticks
        rend = (render_ms or 0.0) * self.draws / n
        self.sim_s, self.draws = 0.0, 0
        seg = [1000.0 * self.seg.get(k, 0.0) / n for k in LAB_SEG_NAMES]
        self.seg = {}
        pt = dict(seg=seg, ms=mean * 1000.0, worst=worst * 1000.0, fps=1.0 / max(mean, 1e-6),
                  avg=lw[0] if lw else None, low1=lw[1] if lw else None, low01=lw[2] if lw else None,
                  phys=phys, rend=rend, util=sens.util, temp=sens.temp, power=sens.power, clock=sens.clock,
                  mem_used=sens.mem_used, mem_total=sens.mem_total, frames=len(acc))
        self.hist.append(pt)
        self.cur = pt
        return True

    # ---- drawing ---------------------------------------------------------------------------------------------
    @staticmethod
    def _heat_colours(temps):
        out = np.zeros((len(temps), 3), np.uint8)
        for i, t in enumerate(temps):
            if t is None:
                out[i] = (90, 90, 100)
            elif t < 65:
                out[i] = (90, 200, 120)
            elif t < 78:
                out[i] = (240, 190, 70)
            else:
                out[i] = (240, 90, 70)
        return out

    @staticmethod
    def _plot(img, x0, y0, w, h, vals, vmax, vmin, colour, fill=0.35, line=True):
        """One series: bars under a 2-pixel line, right-aligned in the box (x0, y0, w, h); colour may be a tuple or one
        colour per point."""
        pts = [v for v in vals]
        if not pts:
            return
        v = np.array([vmin if p is None else p for p in pts], np.float64)
        top = np.round(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0) * (h - 1)).astype(np.int64)
        cols = np.repeat(top, 2)
        n = len(cols)
        n = min(n, w)
        cols = cols[-n:]
        prev = np.concatenate(([cols[0]], cols[:-1]))
        lo, hi = np.minimum(cols, prev), np.maximum(cols, prev)
        rows = np.arange(h)[:, None]
        region = img[y0:y0 + h, x0 + w - n:x0 + w]
        if np.ndim(colour) == 2:
            cc = np.repeat(np.asarray(colour, np.float64), 2, axis=0)[-n:]
        else:
            cc = np.tile(np.asarray(colour, np.float64), (n, 1))
        if fill > 0.0:
            m = (rows < cols[None, :])[::-1]
            idx = np.nonzero(m)
            region[idx] = (region[idx] * (1.0 - fill) + cc[idx[1]] * fill).astype(np.uint8)
        if line:
            m = ((rows >= lo[None, :]) & (rows <= hi[None, :]) | (rows == cols[None, :] - 1))[::-1]
            idx = np.nonzero(m)
            region[idx] = cc[idx[1]].astype(np.uint8)

    @staticmethod
    def _dashes(img, x0, y0, w, h, value, vmax, vmin, colour=(110, 120, 140)):
        if not (vmin <= value <= vmax):
            return
        r = int(round((1.0 - (value - vmin) / (vmax - vmin)) * (h - 1)))
        for x in range(x0, x0 + w, 8):
            img[y0 + r, x:min(x + 4, x0 + w)] = colour

    def layout(self, ph):
        """(block height, header height, panel height) for a plot height ph."""
        blk = 18 + ph + 6
        return blk, 20, 20 + 6 * blk + 18

    def draw(self, ph=54):
        """The panel as an (H, W, 3) uint8 picture, and the texts as a list of (key, string): header_l, header_r,
        then t0..t4 (titles) and v0..v4 (values) in the order of the five graphs."""
        blk, head, H = self.layout(ph)
        W = LAB_PERF_W
        img = np.empty((H, W, 3), np.uint8)
        img[:] = LAB_PERF_BG
        hist = list(self.hist)
        cur = self.cur
        x0, w = 8, W - 16
        gy = [head + k * blk + 18 for k in range(6)]
        for y in gy:
            img[y:y + ph, x0:x0 + w] = (20, 25, 33)
            img[y + ph - 1, x0:x0 + w] = (36, 44, 56)             # baseline
        texts = {}
        if not hist:
            return img, {"header_l": "PERFORMANCE   measuring ...", "header_r": ""}
        # 1  response time
        mx = max(max(p["worst"] for p in hist), 20.0)
        vmax = float(min(100, int(math.ceil(mx * 1.1 / 10.0)) * 10))
        self._dashes(img, x0, gy[0], w, ph, 1000.0 / 60.0, vmax, 0.0)
        self._plot(img, x0, gy[0], w, ph, [p["worst"] for p in hist], vmax, 0.0, (170, 110, 60), fill=0.0)
        self._plot(img, x0, gy[0], w, ph, [p["ms"] for p in hist], vmax, 0.0, (80, 200, 255), fill=0.25)
        texts["t0"], texts["v0"] = "Response time ms", f"{cur['ms']:.1f} now   worst {cur['worst']:.0f}"
        # 2  frame rate and its 1% lows
        fmax = float(max(10, int(math.ceil(max(max(p["fps"] for p in hist), 1.0) * 1.15 / 10.0)) * 10))   # scales to what was reached
        fmax = min(fmax, 300.0)
        self._dashes(img, x0, gy[1], w, ph, 60.0, fmax, 0.0)
        self._plot(img, x0, gy[1], w, ph, [p["fps"] for p in hist], fmax, 0.0, (90, 200, 120), fill=0.28)
        lo = [p["low1"] for p in hist]
        if any(v is not None for v in lo):
            self._plot(img, x0, gy[1], w, ph, [v if v is not None else 0.0 for v in lo], fmax, 0.0, (255, 170, 60), fill=0.0)
        a, l1, l01 = cur["avg"], cur["low1"], cur["low01"]
        texts["t1"] = f"Frame rate {cur['fps']:.0f} fps"
        texts["v1"] = (f"avg {a:.0f}  1% low {l1:.0f}  0.1% low {l01:.0f}" if a else "collecting frames ...")
        # 3  computation: GPU ms per displayed frame, physics + picture
        cm = max(max(p["phys"] + p["rend"] for p in hist), 12.0)
        cmax = float(int(math.ceil(cm * 1.1 / 5.0)) * 5)
        self._dashes(img, x0, gy[2], w, ph, 1000.0 / 60.0, cmax, 0.0)
        self._plot(img, x0, gy[2], w, ph, [p["phys"] + p["rend"] for p in hist], cmax, 0.0, (90, 150, 255), fill=0.45)
        self._plot(img, x0, gy[2], w, ph, [p["phys"] for p in hist], cmax, 0.0, (255, 140, 60), fill=0.55)
        tot = cur["phys"] + cur["rend"]
        texts["t2"] = "GPU compute ms"
        texts["v2"] = f"phys {cur['phys']:.1f} + draw {cur['rend']:.1f} = {tot:.1f}"
        # 4  usage
        us = [p["util"] for p in hist]
        for q in (25, 50, 75):
            img[gy[3] + int(round((1 - q / 100.0) * (ph - 1))), x0:x0 + w:3] = (40, 48, 60)
        texts["t3"] = "GPU usage %"
        if cur["util"] is None:
            texts["v3"] = "no sensor (NVIDIA only)"
        else:
            self._plot(img, x0, gy[3], w, ph, us, 100.0, 0.0, (180, 120, 255), fill=0.3)
            mem = f"  {cur['mem_used']:.1f}/{cur['mem_total']:.0f} GB" if cur["mem_total"] else ""
            clk = f"  {cur['clock']:.0f} MHz" if cur["clock"] else ""
            texts["v3"] = f"{cur['util']:.0f}%{mem}{clk}"
        # 5  heat
        tm = [p["temp"] for p in hist]
        texts["t4"] = "GPU heat"
        if cur["temp"] is None:
            texts["v4"] = "no sensor (NVIDIA only)"
        else:
            for q in (50, 70, 85):
                img[gy[4] + int(round((1 - (q - 30) / 70.0) * (ph - 1))), x0:x0 + w:3] = (40, 48, 60)
            self._plot(img, x0, gy[4], w, ph, tm, 100.0, 30.0, self._heat_colours(tm), fill=0.3)
            pw = f"  {cur['power']:.0f} W" if cur["power"] else ""
            texts["v4"] = f"{cur['temp']:.0f}\u00b0C{pw}"
        # 6  where the frame goes: stacked wall-clock time of each part of the loop
        segs = [p["seg"] for p in hist]
        cum = [np.cumsum(sg) for sg in segs]
        tot = np.array([c[-1] for c in cum])
        smax = float(max(20, int(math.ceil(float(tot.max()) * 1.1 / 10.0)) * 10))
        self._dashes(img, x0, gy[5], w, ph, 1000.0 / 60.0, smax, 0.0)
        for j in range(5, -1, -1):
            self._plot(img, x0, gy[5], w, ph, [c[j] for c in cum], smax, 0.0, LAB_SEG_COLOURS[j], fill=0.85, line=False)
        texts["t5"] = "Where the frame goes ms"
        texts["v5"] = f"{sum(cur['seg']):.0f} of {cur['ms']:.0f}"
        for j, nm in enumerate(LAB_SEG_NAMES):
            v = float(np.mean([sg[j] for sg in segs[-5:]]))                 # a short average: steadier to read
            texts[f"s{j}"] = f"{nm} {v:.0f}" if v >= 10 else f"{nm} {v:.1f}"
        texts["header_l"] = "PERFORMANCE  %d s  (-- 60 fps)" % round(len(hist) * LAB_PERF_SAMPLE)
        texts["header_r"] = ""
        return img, texts


class WarpParticles:
    """Endless particle spawner running entirely on the GPU, for benchmarking."""

    def __init__(self, device=None, capacity=PARTICLE_CAP):
        wp.config.quiet = True
        wp.init()
        if device is None:
            if wp.get_cuda_device_count() == 0:
                raise RuntimeError("No NVIDIA CUDA GPU found (this engine runs only on an NVIDIA GPU)")
            device = "cuda:0"
        self.device = d = wp.get_device(device)
        self.cap = capacity
        self.limit = capacity                  # live "max particles" (ring buffer size)
        self.x = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.xp = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.v = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.corr = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.grid = wp.HashGrid(512, 512, 1, device=d)
        self.pegs = wp.zeros(MAX_PEGS, dtype=wp.vec3, device=d)
        self.npegs = 0
        self.img = wp.zeros(MAX_WORLD_W * MAX_WORLD_H * 3, dtype=wp.uint8, device=d)
        self.h_img = wp.zeros(self.img.shape[0], dtype=wp.uint8, device="cpu", pinned=d.is_cuda)
        self.total = 0                         # particles spawned so far
        self.radius = 1.5
        self.mu = 0.5
        self.world_w = 900.0                   # the simulated area = the visible sim area; its
        self.floor_y = 600.0                   # bottom edge is an invisible floor, the sides are walls
        self.seed = 1
        self.t_sim = GpuTimer(d)
        self.t_render = GpuTimer(d)

    @property
    def count(self):
        return min(self.total, self.limit)

    def describe(self):
        return f"Warp {wp.__version__} on {self.device.name}"

    def reset(self):
        self.total = 0
        self.v.zero_()

    def set_world(self, w, h):
        """Size of the open area particles live in (pixels)."""
        self.world_w = float(min(w, WORLD_CAP_W))
        self.floor_y = float(min(h, WORLD_CAP_H))

    def set_pegs(self, pegs):
        arr = np.zeros((MAX_PEGS, 3), dtype=np.float32)
        pegs = pegs[:MAX_PEGS]
        for k, p in enumerate(pegs):
            arr[k] = p
        self.pegs.assign(arr)
        self.npegs = len(pegs)

    def set_limit(self, limit):
        self.limit = int(max(1, min(limit, self.cap)))
        if self.total > self.limit:
            self.total = self.limit

    def spawn(self, n, ex, ey, width, speed=150.0):
        n = int(n)
        if n <= 0:
            return
        n = min(n, self.limit)
        cols = max(1, min(n, int(width / (2.05 * self.radius))))     # few particles: spread them along the nozzle
        self.seed += 1
        base = self.total % self.limit
        wp.launch(k_p_spawn, dim=n, inputs=[base, self.limit, float(ex), float(ey), float(width),
                                            float(self.radius), cols, float(speed), self.seed, self.x, self.v],
                  device=self.device)
        self.total += n

    def step_bundle(self):
        """4 fixed steps (1/60 s)."""
        n = self.count
        if n == 0:
            return
        d, dt, r = self.device, 1.0 / 240.0, self.radius
        self.t_sim.begin()
        for _ in range(BUNDLE):
            wp.launch(k_p_predict, dim=n, inputs=[dt, n, self.x, self.xp, self.v], device=d)
            self.grid.build(self.xp[:n], 2.0 * r)
            for _ in range(P_ITERS):
                wp.launch(k_p_project, dim=n, inputs=[self.grid.id, n, r, self.mu, self.x, self.xp, self.corr,
                                                      self.pegs, self.npegs], device=d)
                wp.launch(k_p_apply, dim=n, inputs=[n, r, self.mu, P_RELAX, self.floor_y, self.world_w, self.x, self.xp,
                                                    self.corr], device=d)
            wp.launch(k_p_update, dim=n, inputs=[dt, n, self.x, self.xp, self.v], device=d)
        self.t_sim.end()

    def render(self, scale=1, view=(900, 600), cam=(0.0, 0.0), zoom=1.0):
        """Draw the particles seen by the camera (top-left `cam` in world pixels, magnified by `zoom`)
        into the GPU image (1/scale of the view size) and copy it to the host.
        Returns (width, height, RGB uint8 array)."""
        n = self.count
        d = self.device
        iw, ih = (int(view[0]) + scale - 1) // scale, (int(view[1]) + scale - 1) // scale
        nbytes = iw * ih * 3
        self.t_render.begin()
        wp.launch(k_p_clear, dim=nbytes, inputs=[self.img], device=d)
        if n:
            wp.launch(k_p_render, dim=n, inputs=[n, self.radius, 1.0 / scale, iw, ih, float(cam[0]), float(cam[1]), float(zoom),
                                         self.x, self.v, self.img],
                      device=d)
        self.t_render.end()
        wp.copy(self.h_img, self.img, count=nbytes)
        wp.synchronize_device(d)
        return iw, ih, self.h_img.numpy()[:nbytes]

    def stats(self):
        """(mean sim ms per bundle, mean render ms) of whatever has finished since the last call."""
        a, b = self.t_sim.take(), self.t_render.take()
        return (sum(a) / len(a) if a else None), (sum(b) / len(b) if b else None)


# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# bodies and the world (plain Python descriptions; the GPU holds the live state)
# ----------------------------------------------------------------------------
BOX, CIRCLE = 0, 1


# ----------------------------------------------------------------------------
# rigid bodies
# ----------------------------------------------------------------------------
class Body:
    __slots__ = ("id", "kind", "style", "x", "y", "a", "c", "s", "vx", "vy", "w",
                 "m", "im", "I", "iI", "hw", "hh", "r", "mu", "e", "static",
                 "aabb", "items", "asleep", "sleep_t", "nosleep", "tunable", "material", "gen", "ref")

    def __init__(self, kind, x, y, hw=0.0, hh=0.0, r=0.0, density=None,
                 static=False, mu=None, e=None, style="default", angle=0.0, material=None):
        spec = MATERIALS.get(material) if material else None       # a material sets density / friction / bounce
        density = density if density is not None else (spec["density"] if spec else 0.006)
        mu = mu if mu is not None else (spec["mu"] if spec else 0.5)
        e = e if e is not None else (spec["e"] if spec else 0.1)
        self.material = material            # a key of MATERIALS: what the body breaks like (None = unbreakable)
        self.gen = 0                        # how many times its ancestors have been broken up
        self.ref = (r * 1.77 if kind == CIRCLE else math.sqrt(4 * hw * hh))   # size of the unbroken original (see fracture_step)
        self.id = 0
        self.kind = kind
        self.style = style
        self.x, self.y, self.a = x, y, angle
        self.c, self.s = math.cos(angle), math.sin(angle)
        self.vx = self.vy = self.w = 0.0
        self.hw, self.hh, self.r = hw, hh, r
        self.mu, self.e = mu, e
        self.static = static
        if kind == BOX:
            m = density * 4 * hw * hh
            inertia = m * ((2 * hw) ** 2 + (2 * hh) ** 2) / 12
        else:
            m = density * math.pi * r * r
            inertia = 0.5 * m * r * r
        if static:
            self.m = self.im = self.I = self.iI = 0.0
        else:
            self.m, self.im = m, 1 / m
            self.I, self.iI = inertia, 1 / inertia
        self.aabb = None
        self.items = []
        self.asleep = static          # static bodies count as permanently asleep
        self.sleep_t = 0.0
        self.nosleep = False
        self.tunable = False          # True: WarpPhysics.set_material() changes this body's friction / bounce

    def compute_aabb(self):
        if self.kind == CIRCLE:
            ex = ey = self.r
        else:
            ac, as_ = abs(self.c), abs(self.s)
            ex = ac * self.hw + as_ * self.hh
            ey = as_ * self.hw + ac * self.hh
        self.aabb = (self.x - ex, self.y - ey, self.x + ex, self.y + ey)

    def contains(self, px, py):
        dx, dy = px - self.x, py - self.y
        if self.kind == CIRCLE:
            return dx * dx + dy * dy <= self.r * self.r
        lx = dx * self.c + dy * self.s
        ly = -dx * self.s + dy * self.c
        return abs(lx) <= self.hw and abs(ly) <= self.hh


# ----------------------------------------------------------------------------
# the rigid world
# ----------------------------------------------------------------------------
class World:
    """Just a list of the bodies. All the physics happens on the GPU (WarpPhysics)."""

    def __init__(self):
        self.dyn = []
        self.static = []
        self.next_id = 1
        self.grab = None           # ("body", body, lx, ly): the body being dragged with the mouse
        self.mouse = (0.0, 0.0)
        self.time = 0.0

    def add(self, body):
        body.id = self.next_id
        self.next_id += 1
        if body.static:
            body.compute_aabb()
            self.static.append(body)
        else:
            self.dyn.append(body)
        return body

    def remove(self, body):
        if body in self.dyn:
            self.dyn.remove(body)
        if self.grab and self.grab[0] == "body" and self.grab[1] is body:
            self.grab = None

    def try_grab_body(self, mx, my):
        for b in reversed(self.dyn):               # newest on top
            if b.contains(mx, my):
                dx, dy = mx - b.x, my - b.y
                lx = dx * b.c + dy * b.s
                ly = -dx * b.s + dy * b.c
                self.grab = ("body", b, lx, ly)
                return True
        return False

    def bodies_near(self, box, margin):
        """Dynamic bodies whose bounding box is close to box = (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
        out = []
        for b in self.dyn:
            bx0, by0, bx1, by1 = b.aabb
            if not (bx1 < x0 or x1 < bx0 or by1 < y0 or y1 < by0):
                out.append(b)
        return out




# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _corners(cx, cy, hw, hh, a):
    c, s = math.cos(a), math.sin(a)
    return [(cx + sx * hw * c - sy * hh * s, cy + sx * hw * s + sy * hh * c)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def obb_overlap(p, q, tol=1.0):
    """True if two oriented boxes (cx, cy, hw, hh, angle) overlap by more than `tol` pixels."""
    pa, pb = _corners(*p), _corners(*q)
    for box in (p, q):
        c, s = math.cos(box[4]), math.sin(box[4])
        for ax, ay in ((c, s), (-s, c)):
            a = [x * ax + y * ay for x, y in pa]
            b = [x * ax + y * ay for x, y in pb]
            if min(max(a), max(b)) - max(min(a), min(b)) <= tol:
                return False
    return True


# ####################################################################################################
##### PART: 2  STATE MANAGER + STATE VIEWER
# ####################################################################################################

# ####################################################################################################
##### SECTION: clock
# ####################################################################################################

"""Simulated clock.  Belongs to the ENVIRONMENT, not to the character.

Time is stored as a minute-resolution ISO string so it is trivially JSON
serialisable and human readable in save files and debug views.
"""

from datetime import datetime, timedelta

FMT = "%Y-%m-%dT%H:%M"
DEFAULT_START = "2040-01-01T08:00"


def parse(ts: str) -> datetime:
    return datetime.strptime(ts, FMT)


def minutes_between(earlier: str, later: str) -> int:
    return int((parse(later) - parse(earlier)).total_seconds() // 60)


def short(ts: str) -> str:
    """HH:MM view used in compact displays."""
    return parse(ts).strftime("%H:%M")


class SimClock:
    def __init__(self, now: str = DEFAULT_START, ticks: int = 0):
        self.now = now
        self.ticks = ticks  # total simulated minutes elapsed since creation

    def advance(self, minutes: int) -> str:
        minutes = max(0, int(minutes))
        self.now = (parse(self.now) + timedelta(minutes=minutes)).strftime(FMT)
        self.ticks += minutes
        return self.now

    def stamp(self) -> str:
        return self.now

    def to_dict(self) -> dict:
        return {"now": self.now, "ticks": self.ticks}

    @classmethod
    def from_dict(cls, d: dict) -> "SimClock":
        return cls(d["now"], d.get("ticks", 0))


# ####################################################################################################
##### SECTION: events
# ####################################################################################################

"""Event log.

EVENT   = something that happened (recorded here, never edited, never lossy).
It is NOT the character's experience and NOT their memory.  Events carry a `layer`:

    world  - the environment changed / an action physically occurred
    actor  - something changed in the character's own state (e.g. they learned a fact)
    system - session bookkeeping (start, resume, save)

Event -> Observation -> Experience -> Memory is the pipeline; this file is the
first stage only.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


class EV:
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_RESUMED = "SESSION_RESUMED"
    LOOKED_AROUND = "LOOKED_AROUND"
    ROOM_ENTERED = "ROOM_ENTERED"
    ACTOR_MOVED = "ACTOR_MOVED"
    OBJECT_OBSERVED = "OBJECT_OBSERVED"
    OBJECT_EXAMINED = "OBJECT_EXAMINED"
    OBJECT_PICKED_UP = "OBJECT_PICKED_UP"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    OBJECT_USED = "OBJECT_USED"
    OBJECT_REVEALED = "OBJECT_REVEALED"
    WORLD_OBJECT_CHANGED = "WORLD_OBJECT_CHANGED"
    TIME_PASSED = "TIME_PASSED"
    ACTION_FAILED = "ACTION_FAILED"
    INFORMATION_LEARNED = "INFORMATION_LEARNED"
    UTTERANCE = "UTTERANCE"


@dataclass
class Event:
    id: int
    time: str
    layer: str
    type: str
    actor: str
    target: Optional[str] = None
    place: Optional[str] = None
    data: dict = field(default_factory=dict)

    def describe(self) -> str:
        tgt = f" -> {self.target}" if self.target else ""
        extra = f" {self.data}" if self.data else ""
        return f"#{self.id:03d} [{self.time}] ({self.layer}) {self.type} by {self.actor}{tgt}{extra}"


class EventLog:
    MAX_EVENTS = 6000          # long unattended runs must not grow the log without bound
    KEEP_EVENTS = 5000

    def __init__(self, events: Optional[list] = None):
        self.events: list[Event] = events or []
        self.last_id = self.events[-1].id if self.events else 0

    def add(self, time: str, layer: str, type: str, actor: str = "me",
            target: Optional[str] = None, place: Optional[str] = None, **data) -> Event:
        self.last_id += 1
        ev = Event(self.last_id, time, layer, type, actor, target, place, data)
        self.events.append(ev)
        if len(self.events) > self.MAX_EVENTS:
            del self.events[:len(self.events) - self.KEEP_EVENTS]
        return ev

    def recent(self, n: int = 10) -> list[Event]:
        return self.events[-n:]

    def to_dict(self) -> dict:
        return {"events": [asdict(e) for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict) -> "EventLog":
        return cls([Event(**e) for e in d.get("events", [])])


# ####################################################################################################
##### SECTION: state
# ####################################################################################################

"""The character's INTERNAL STATE.

Everything here is the character's own: what they have perceived, remember, believe, doubt
and want.  It is deliberately plain dicts/lists so it serialises to
JSON untouched and can be extended later (see `extra`).

Layers, from most to least transient:
    EVENT        world log (events.py) - not in here at all
    OBSERVATION  filtered percepts of one moment (perception.py)
    EXPERIENCE   the character's processed record of an observation -> `recent_observations`
    MEMORY       consolidated, lossy, confidence-weighted -> `memories`
    STATE        the standing picture: beliefs, known_*, uncertainties, goals
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

SCHEMA_VERSION = 2


@dataclass
class CharState:
    identity: dict = field(default_factory=dict)
    current_location: dict = field(default_factory=dict)
    current_activity: str = "idle"
    known_people: dict = field(default_factory=dict)
    known_places: dict = field(default_factory=dict)
    known_objects: dict = field(default_factory=dict)
    memories: list = field(default_factory=list)
    beliefs: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    recent_observations: list = field(default_factory=list)
    # --- extensions beyond the required minimum --------------------------
    inventory: list = field(default_factory=list)          # object ids the character knows they are holding
    perceived_keys: dict = field(default_factory=dict)     # percept key -> {completeness,label,time,subject}
    topic_index: dict = field(default_factory=dict)        # topic -> [{key,subject,gist,time}]
    impressions: dict = field(default_factory=dict)        # latent, unplaced familiarity (tag -> strength)
    state_changes: list = field(default_factory=list)      # [{time,field,change}]
    needs: dict = field(default_factory=dict)             # need -> {"band": 0-3, "since": time}  (what the character FEELS)
    flags: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)              # free-form bag for future features

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CharState":
        d = dict(d)
        d.pop("schema_version", None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def new_character(start_time: str, c: Optional[dict] = None) -> CharState:
    """The character as they are when first 'switched on', built from the profile's `character` block:
    they know who they are (vaguely, if you like), and only what you told them about the place."""
    c = c or {}
    name = (c.get("name") or "Alex").strip()
    s = CharState()
    s.identity = {"name": name, "role": (c.get("role") or "").strip(),
                  "self_description": (c.get("description") or f"I'm {name}.").strip(),
                  "voice": (c.get("voice") or "").strip(), "pronouns": (c.get("pronouns") or "they/them").strip()}
    s.current_location = {"room": None, "room_label": "somewhere I haven't taken in yet", "spot": None, "spot_label": ""}
    s.current_activity = "just getting my bearings"
    disoriented = bool(c.get("disoriented", True))
    knows_arrival = bool(c.get("knows_arrival", False))
    goals, unc, n = [], [], 0
    if disoriented:
        goals.append({"id": "g_orient", "text": "Work out where I am", "status": "active", "priority": 1, "created": start_time})
        unc.append({"id": "u_0001", "key": "where_am_i", "question": "Where am I?", "about": "location", "status": "open",
                    "raised": start_time, "resolved": None, "notes": "I haven't really looked around yet."})
    if not knows_arrival:
        goals.append({"id": "g_how", "text": "Understand how I got here", "status": "active", "priority": 2, "created": start_time})
        unc.append({"id": f"u_{len(unc) + 1:04d}", "key": "how_here", "question": "How did I get here?", "about": "arrival",
                    "status": "open", "raised": start_time, "resolved": None, "notes": "I don't remember arriving."})
    goals.append({"id": "g_explore", "text": "See what is around me and what it means", "status": "active", "priority": 3,
                  "created": start_time})
    for i, text in enumerate(c.get("goals") or []):
        if str(text).strip():
            goals.append({"id": f"g_own_{i + 1}", "text": str(text).strip(), "status": "active", "priority": 4,
                          "created": start_time})
    s.goals, s.uncertainties = goals, unc
    back = (c.get("backstory") or "").strip()
    s.memories = [{
        "id": "m_0001", "time": start_time, "text": back if back else "I remember who I am, but beyond that things are hazy.",
        "confidence": float(c.get("backstory_confidence", 0.5 if back else 0.35)), "source": "background", "subject": None,
        "place": None, "percept_keys": [], "event_ids": [], "incomplete": True, "recalled": 0, "last_recalled": None,
        "tags": ["identity"],
    }]
    for i, fact in enumerate(c.get("knowledge") or []):       # things they already believe about the world
        if str(fact).strip():
            s.beliefs.append({"id": f"b_{i + 1:04d}", "about": "background", "attr": f"fact{i}", "value": str(fact).strip(),
                              "text": str(fact).strip().rstrip(".") + ".", "confidence": 0.8, "source": "background",
                              "evidence": [], "formed": start_time, "updated": start_time, "revisions": 0})
    s.impressions = {}
    s.needs = {n_: {"band": 0, "since": start_time} for n_ in ("thirst", "hunger", "fatigue", "boredom")}
    s.counters = {"memory": 1, "observation": 0, "belief": len(s.beliefs), "uncertainty": len(unc)}
    s.flags = {"has_looked": False}
    s.extra = {"rng_seed": int(c.get("seed", 1337))}
    return s


# ####################################################################################################
##### SECTION: persistence
# ####################################################################################################

"""JSON persistence: ONE file, state_save.json, holding four sections - "profile", "world", "character" and "events".

The sections are still independent on purpose: delete the "character" key and a fresh character wakes up in the same,
changed world; delete "world" and the character keeps their memories but finds the world rebuilt from the profile.

Writes are atomic (temp file + rename) so a crash mid-save cannot corrupt a save.  A file that cannot be read is kept
beside the original as *.corrupt-<time> instead of being destroyed.
"""

import json
import os
import time
from pathlib import Path

SECTIONS = ("profile", "world", "character", "events")
SAVE_NAME = "state_save.json"
SAVE_HOME = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))        # saves go beside this script unless --save says otherwise


class SaveStore:
    def __init__(self, path: str):
        p = Path(path)
        if not p.is_absolute() and p.parent == Path("."):        # a bare name like state_save.json: put it beside the script, not in whatever folder it was started from
            p = Path(SAVE_HOME) / p
        self.file = p if p.suffix.lower() == ".json" else p / SAVE_NAME      # a folder path puts state_save.json inside it
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.dir = self.file.parent
        self.migrated = False
        self._data: Optional[dict] = None

    # ---- reading
    def _read(self) -> dict:
        if self._data is None:
            self._data = {}
            if self.file.exists():
                try:
                    d = json.loads(self.file.read_text(encoding="utf-8"))
                    if not isinstance(d, dict):
                        raise ValueError("not an object")
                    self._data = d
                except (json.JSONDecodeError, OSError, ValueError):
                    try:
                        self.file.rename(self.file.with_name(self.file.name + f".corrupt-{int(time.time())}"))
                    except OSError:
                        pass
        return self._data

    # ---- writing
    def flush(self):
        d = self._read()
        out = {"format": "state-save", "version": 1, **{k: d[k] for k in SECTIONS if k in d}}
        text = json.dumps(out, indent=1, ensure_ascii=False)
        tmp = self.file.with_suffix(".tmp")
        err = None
        for attempt in range(6):                         # OneDrive / antivirus can hold the file for a moment: retry, then write directly
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.file)
                return
            except OSError as e:
                err = e
                time.sleep(0.15)
        try:
            self.file.write_text(text, encoding="utf-8")
            return
        except OSError as e:
            err = e
        if not getattr(self, "_warned", False):          # still can't: say so once, keep the simulation running
            self._warned = True
            print(f"(Could not save to {self.file}: {err}. The simulation keeps running; use --save PATH to pick a writable place.)")

    def exists(self, name: str) -> bool:
        return name in self._read()

    def load(self, name: str) -> Optional[dict]:
        return self._read().get(name)

    def save(self, name: str, data: dict) -> None:
        self._read()[name] = data
        self.flush()

    def save_all(self, **sections) -> None:
        self._read().update(sections)
        self.flush()

    def delete(self, name: str) -> None:
        if name in self._read():
            del self._read()[name]
            self.flush()


# ####################################################################################################
##### SECTION: perception
# ####################################################################################################

"""PERCEPTION: the one-way filter between WORLD and the CHARACTER.

Every function here reads the authoritative world and returns an Observation
containing only Percepts - the things the character could legitimately perceive right
now.  World `truth`, hidden objects, un-reached detail depth, text that is too
dark to read, and the identities of things seen from afar never pass through.

The Character module never imports `world`; it only ever sees what this file
emits.  That is the architectural guarantee behind "the character does not have the
whole world state".
"""

from dataclasses import dataclass, field, asdict
from typing import Optional



@dataclass
class Percept:
    key: str                          # stable id of this piece of perceivable information
    text: str                         # as perceived (first-person friendly)
    subject: Optional[str] = None     # object the character can attribute it to (None = room/ambient)
    subject_label: str = ""           # what the character calls the subject
    gist: str = ""                    # short form used when forming a memory
    salience: float = 0.6             # how memorable (0-1)
    completeness: float = 1.0         # 1.0 full, <1 partial (dark, far, faded)
    label: str = ""                   # content-free pointer for "I saw something about X"
    kind: str = "fact"                # fact | sighting | room | sensation | ambient | held | released
    topics: list = field(default_factory=list)
    recognizable: Optional[dict] = None
    unresolved: Optional[str] = None
    belief: Optional[dict] = None
    place: Optional[str] = None
    place_label: str = ""
    attach_to: Optional[str] = None   # narrate as ", and ..." after another percept
    narrate: bool = True
    holds: Optional[str] = None       # object id now held
    releases: Optional[str] = None
    resolves: list = field(default_factory=list)   # uncertainty keys this percept answers
    mobile: bool = False              # a portable thing (worth tracking the location of)
    hint: str = ""                    # what the thing looks like it is for: a need (thirst/hunger/fatigue/boredom) or "light"
    loose: bool = False               # a loose item on a surface (resolves "what is lying there?")
    body: Optional[dict] = None       # {"need","band"[, "relief": True]} - a bodily sensation
    act: str = ""                     # what the character would say they did ("drank water") - feeds the activity report
    stale_minutes: int = 0            # time passed unobserved: changeable beliefs go stale


@dataclass
class Observation:
    kind: str                         # look | arrive | inspect | take | drop | use | ambient | inventory
    time: str
    room: Optional[str] = None
    room_label: str = ""
    place: Optional[str] = None
    place_label: str = ""
    subject: Optional[str] = None
    subject_label: str = ""
    light: float = 1.0
    percepts: list = field(default_factory=list)
    activity: str = ""
    id: str = ""
    event_ids: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ####################################################################################################
##### SECTION: character
# ####################################################################################################

"""CHARACTER: how the character's internal state changes when they perceive something.

This module never imports `world`.  Its only inputs are Observations (already
filtered by perception.py) and the passage of time.  It owns the transitions

    OBSERVATION -> EXPERIENCE -> MEMORY
                -> BELIEFS / UNCERTAINTIES / KNOWN_* / GOALS

and the retrieval used when the character is asked about themselves or the world.

Memory is deliberately NOT a copy of world truth: it stores the character's lossy,
confidence-weighted account of what he experienced.  Low-salience details may
be dropped when a memory forms (deterministically, from a seeded RNG), and
confidence decays with simulated time.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional


MIN_MEMORY_SALIENCE = 0.3
HALF_LIFE_MINUTES = 24 * 60          # memory confidence halves every simulated day
SYNONYMS = {"photo": "picture", "pic": "picture", "lady": "woman", "guy": "man", "movie": "film", "films": "film",
            "exit": "door", "inscription": "writing", "year": "date"}
# Question words widen the search, but only against *labels* (what a thing is about),
# never against free text - otherwise "where?" would match "I can't place him".
EXPAND = {"where": {"place", "location"}, "when": {"date", "time"}, "who": {"person"}, "someone": {"person"},
          "whose": {"person"}}
MUTABLE_ATTRS = ("time_of_day", "state")       # beliefs about things that change while unobserved
NEED_ORDER = ("thirst", "hunger", "fatigue", "boredom")
NEED_GOAL = {"thirst": "Get something to drink", "hunger": "Find something to eat", "fatigue": "Get some sleep",
             "boredom": "Find something to occupy my mind"}
MAX_MEMORIES = 300
MAX_DIARIES = 60
KIND_PRIORITY = {"person": 0, "memory": 1, "belief": 2, "forgotten": 3, "uncertainty": 4}
QSTOP = {"what", "who", "where", "when", "why", "how", "do", "does", "did", "you", "your", "know", "about", "is", "are",
         "was", "were", "there", "remember", "tell", "me", "see", "saw", "can", "could", "it", "its", "i", "of",
         "in", "on", "that", "this", "the", "a", "an", "to", "and", "or", "any", "anything", "have", "has", "recall",
         "think", "say", "look", "looked", "looks", "like"}


def _stem_word(t: str) -> str:
    """watched / watching / watches -> watch, so a question and a memory worded differently still meet."""
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def _tokens(text: str) -> set:
    toks = {SYNONYMS.get(_stem_word(t), _stem_word(t)) for t in re.findall(r"[a-z0-9']+", text.lower())}
    return {t for t in toks if t not in QSTOP and len(t) > 1}


@dataclass
class Delta:
    """Everything that changed in the character because of one observation (for debugging)."""
    new: list = field(default_factory=list)
    upgraded: list = field(default_factory=list)
    repeats: list = field(default_factory=list)
    voice: list = field(default_factory=list)      # things the character would say aloud (vs. debug `notes`)
    recognitions: list = field(default_factory=list)
    memories: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def learned_something(self) -> bool:
        return bool(self.new or self.upgraded)

    def lines(self) -> list:
        out = []
        for p in self.new:
            out.append(f"+ learned [{p.key}] {p.gist or p.text}")
        for p in self.upgraded:
            out.append(f"^ upgraded [{p.key}] (now {p.completeness:.1f}) {p.gist or p.text}")
        if self.repeats and not (self.new or self.upgraded):
            out.append(f"= nothing new ({len(self.repeats)} already-known percepts)")
        out += [f"~ {n}" for n in self.notes]
        for m in self.memories:
            out.append(f"* memory {m['id']} ({m['confidence']}, {m['source']}"
                       f"{', incomplete' if m['incomplete'] else ''}): {m['text']}")
        return out


class Character:
    def __init__(self, state: CharState):
        self.s = state
        self.now = ""

    # ------------------------------------------------------------------ small helpers
    def _next(self, name: str) -> int:
        self.s.counters[name] = self.s.counters.get(name, 0) + 1
        return self.s.counters[name]

    def _changed(self, field_: str, change: str):
        self.s.state_changes.append({"time": self.now, "field": field_, "change": change})
        del self.s.state_changes[:-300]

    def _name(self, oid: str) -> str:
        return self.s.known_objects.get(oid, {}).get("name", oid)

    # ------------------------------------------------------------------ uncertainty & goals
    def open_uncertainty(self, key: str, question: str, about: Optional[str] = None, notes: str = "",
                         goal: bool = True) -> bool:
        if any(u["key"] == key and u["status"] == "open" for u in self.s.uncertainties):
            return False
        self.s.uncertainties.append({"id": f"u_{self._next('uncertainty'):04d}", "key": key, "question": question,
                                     "about": about, "status": "open", "raised": self.now, "resolved": None,
                                     "notes": notes})
        self._changed("uncertainties", f"opened: {question}")
        if goal and key not in ("where_am_i", "how_here"):
            self._add_goal(f"Find out: {question}", key)
        return True

    def resolve_uncertainty(self, key: str, note: str = "") -> bool:
        hit = False
        for u in self.s.uncertainties:
            if u["key"] == key and u["status"] == "open":
                u.update(status="resolved", resolved=self.now, notes=(note or u["notes"]))
                self._changed("uncertainties", f"resolved: {u['question']} -> {note}")
                hit = True
        if hit:
            for g in self.s.goals:
                if g.get("link") == key and g["status"] == "active":
                    g["status"] = "done"
                    self._changed("goals", f"done: {g['text']}")
            if key == "where_am_i":
                for g in self.s.goals:
                    if g["id"] == "g_orient":
                        g["status"] = "done"
        return hit

    def _add_goal(self, text: str, link: str):
        for g in self.s.goals:
            if g.get("link") == link:
                if g["status"] != "active" and link.startswith("need:"):
                    g["status"] = "active"
                    self._changed("goals", f"again: {text}")
                return
        self.s.goals.append({"id": f"g_{len(self.s.goals) + 1:03d}", "text": text, "status": "active",
                             "priority": 5, "created": self.now, "link": link})
        self._changed("goals", f"new: {text}")

    # ------------------------------------------------------------------ beliefs
    def update_belief(self, about: str, attr: str, value, text: str, confidence: float, source: str,
                      evidence: Optional[str] = None) -> Optional[str]:
        """Add / confirm / revise a belief.  Returns 'new' | 'confirmed' | 'revised'."""
        for b in self.s.beliefs:
            if b["about"] == about and b["attr"] == attr:
                if b["value"] == value:
                    b["confidence"] = round(max(b["confidence"], confidence), 2)
                    b["updated"] = self.now
                    return "confirmed"
                b.setdefault("history", []).append({"value": b["value"], "text": b["text"], "until": self.now})
                b.update(value=value, text=text, confidence=round(confidence, 2), updated=self.now,
                         source=source, revisions=b.get("revisions", 0) + 1)
                self._changed("beliefs", f"revised: {text}")
                return "revised"
        self.s.beliefs.append({"id": f"b_{self._next('belief'):04d}", "about": about, "attr": attr, "value": value,
                               "text": text, "confidence": round(confidence, 2), "source": source,
                               "evidence": [evidence] if evidence else [], "formed": self.now, "updated": self.now,
                               "revisions": 0})
        self._changed("beliefs", f"new: {text}")
        return "new"

    # ------------------------------------------------------------------ main entry point
    def perceive(self, obs: Observation, now: str) -> Delta:
        s, d = self.s, Delta()
        self.now = now
        obs.id = f"o_{self._next('observation'):04d}"
        held_before = list(s.inventory)

        if obs.place and obs.kind not in ("inventory", "body"):   # touch / bodily feelings alone doesn't tell him where he is
            loc = {"room": obs.room, "room_label": obs.room_label, "spot": obs.place, "spot_label": obs.place_label}
            if loc != s.current_location:
                self._changed("current_location", f"{obs.room_label}, {obs.place_label}")
            s.current_location = loc
            s.extra.setdefault("room_last", {})[obs.room] = now
            s.known_places.setdefault(obs.room, {"label": obs.room_label, "kind": "room", "first_seen": now})
            s.known_places.setdefault(obs.place, {"label": obs.place_label, "kind": "spot", "room": obs.room,
                                                  "first_seen": now})
        activity = obs.activity or {"look": "looking around", "arrive": "getting my bearings",
                                    "inspect": f"examining the {obs.subject_label}", "take": f"holding the {obs.subject_label}",
                                    "drop": "standing here", "use": f"using the {obs.subject_label}",
                                    "ambient": s.current_activity, "inventory": s.current_activity,
                                    "body": s.current_activity}.get(obs.kind, "idle")
        if activity != s.current_activity:
            self._changed("current_activity", activity)
            s.current_activity = activity

        statuses = []
        for p in obs.percepts:
            status = self._status(p)
            statuses.append((p, status))
            {"new": d.new, "upgraded": d.upgraded, "repeat": d.repeats}[status].append(p)
            self._integrate(p, obs, status, d)

        if obs.kind == "inspect" and obs.subject:
            ko = s.known_objects.setdefault(obs.subject, {"name": obs.subject_label or obs.subject})
            ko.setdefault("name", obs.subject_label or obs.subject)
            ko["inspections"] = ko.get("inspections", 0) + 1
            if not any(p.key.endswith(".too_dark") for p in obs.percepts):
                self.resolve_uncertainty(f"partial:{obs.subject}.too_dark", "I could see it properly this time.")
        if obs.kind == "use" and obs.subject:
            ko = s.known_objects.setdefault(obs.subject, {"name": obs.subject_label or obs.subject})
            ko.setdefault("name", obs.subject_label or obs.subject)
            ko["uses"] = ko.get("uses", 0) + 1
        if obs.kind == "look" or any(p.kind == "room" for p in obs.percepts):
            s.flags["has_looked"] = True
            self.resolve_uncertainty("where_am_i", f"I'm in {obs.room_label}.")
        if obs.kind == "inventory":
            self._reconcile_inventory(obs, held_before, d)

        self._consolidate(obs, statuses, d)
        self._record_experience(obs, d)
        self._log_acts(obs)
        pk = s.perceived_keys
        if len(pk) > 700:                       # one-off keys ("...@time") are never needed again
            for k in [k for k in pk if "@" in k][:-250]:
                del pk[k]
        return d

    # ------------------------------------------------------------------ the body
    def _apply_body(self, p: Percept, d: Delta):
        """A bodily sensation: update what the character feels, set/clear need goals, learn what relieves what."""
        s, need, band = self.s, p.body["need"], p.body["band"]
        old = s.needs.get(need, {"band": 0, "since": self.now})
        if band != old["band"]:
            s.needs[need] = {"band": band, "since": self.now}
            self._changed("needs", f"{need}: {old['band']} -> {band}")
        else:
            s.needs[need] = old
        link = f"need:{need}"
        if band >= 2:
            self._add_goal(NEED_GOAL[need], link)
        elif band <= 1:
            for g in s.goals:
                if g.get("link") == link and g["status"] == "active":
                    g["status"] = "done"
                    self._changed("goals", f"done: {g['text']}")
        if p.body.get("relief") and p.subject:
            known = s.extra.setdefault("relief", {}).setdefault(need, [])
            name = p.subject_label
            if name not in known:
                known.append(name)
                d.notes.append(f"learned: the {name} relieves {need}")

    def _age_beliefs(self, minutes: int, d: Delta):
        """Time passed unobserved (sleep): beliefs about changeable things become less certain."""
        hit = False
        for b in self.s.beliefs:
            if b["attr"] in MUTABLE_ATTRS and b["source"] == "direct_observation" and not b["about"].startswith("link:"):
                new = round(max(0.15, b["confidence"] * 0.5 ** (minutes / 480)), 2)
                if new < b["confidence"]:
                    b["confidence"], hit = new, True
                    b["stale"] = True
        if hit:
            d.notes.append("some of what I believed may be out of date after the time that passed")
            self.open_uncertainty("stale:time", "What is it like outside now? I lost track of the time.", "time_of_day",
                                  "I slept, so what I last saw may have changed.", goal=False)

    # ------------------------------------------------------------------ activity log (for `report`)
    def log_act(self, text: str):
        if not text:
            return
        tot = self.s.extra.setdefault("activity_totals", {})
        tot[text] = tot.get(text, 0) + 1
        days = self.s.extra.setdefault("day_acts", {})
        today = days.setdefault((self.now or "")[:10], {})
        today[text] = today.get(text, 0) + 1
        for old in sorted(days)[:-3]:                       # only the last few days are needed for the diary
            del days[old]
        self._check_goals()

    def _log_acts(self, obs: Observation):
        acts = [p.act for p in obs.percepts if p.act]
        if not acts:
            sl = obs.subject_label
            derived = {"inspect": f"looked closely at the {sl}", "take": f"picked up the {sl}", "drop": f"put down the {sl}",
                       "look": "looked around", "use": f"used the {sl}" if sl else ""}.get(obs.kind, "")
            acts = [derived] if derived else []
        for a in acts[:1]:
            self.log_act(a)

    def _lasting_keys(self) -> int:
        return sum(1 for k in self.s.perceived_keys if "@" not in k)

    def mark_session(self):
        s = self.s
        s.extra["session"] = {"start": self.now, "totals": dict(s.extra.get("activity_totals", {})),
                              "keys": self._lasting_keys(), "memories": s.counters.get("memory", 0),
                              "objects": len(s.known_objects)}

    def session_report(self) -> dict:
        s = self.s
        sess = s.extra.get("session") or {"start": (s.memories[0]["time"] if s.memories else self.now), "totals": {},
                                          "keys": 0, "memories": 0}
        tot = s.extra.get("activity_totals", {})
        acts = [(a, c - sess["totals"].get(a, 0)) for a, c in tot.items() if c - sess["totals"].get(a, 0) > 0]
        acts = self._collapse_acts(acts)
        basics = [a for a in acts if re.match(r"(slept|drank|ate)\b", a[0])]
        acts = (basics + [a for a in acts if a not in basics])[:8]        # staying alive always makes the report
        notable = self.notable_memories(after_id=sess["memories"], limit=3)           # what happened, not the daily routine
        return {"since": sess["start"], "until": self.now, "acts": acts,
                "learned": max(0, len(s.known_objects) - sess.get("objects", 0)),
                "needs": {n: v["band"] for n, v in s.needs.items()},
                "highlights": [self.as_fact(m["text"]) for m in notable],
                "minutes": minutes_between(sess["start"], self.now) if self.now else 0}

    @staticmethod
    def _collapse_acts(acts: list, keep_total: bool = False) -> list:
        """'looked closely at the desk twice, ... the door twice ...' -> 'looked closely at 14 different things'."""
        out, groups, weight = [], {}, {}
        for a, c in acts:
            for pre, plural in (("looked closely at the ", "looked closely at {n} different things"),
                                ("picked up the ", "picked up {n} different things"),
                                ("used the ", "used {n} different things")):
                if a.startswith(pre):
                    g = groups.setdefault(pre, [plural, 0, []])
                    g[1] += c
                    g[2].append((a, c))
                    break
            else:
                out.append((a, c))
        for pre, (plural, total, items) in groups.items():
            if len(items) >= 3:
                t = plural.format(n=len(items))
                out.append((t, total if keep_total else 1))
                weight[t] = total
            else:
                out += items
        out.sort(key=lambda x: -weight.get(x[0], x[1]))
        return out

    def _prune_memories(self):
        diaries = [m for m in self.s.memories if m["source"] == "diary"]
        if len(diaries) > MAX_DIARIES:                       # the oldest pages eventually go
            gone = {m["id"] for m in diaries[:len(diaries) - MAX_DIARIES]}
            self.s.memories = [m for m in self.s.memories if m["id"] not in gone]
        live = [m for m in self.s.memories if m["source"] not in ("background", "diary")]
        if len(live) <= MAX_MEMORIES:
            return
        cull = sorted(live, key=lambda m: (self.effective_confidence(m), m["time"]))[:len(live) - MAX_MEMORIES]
        ids = {m["id"] for m in cull}
        self.s.memories = [m for m in self.s.memories if m["id"] not in ids]

    # ------------------------------------------------------------------ percept handling
    def _status(self, p: Percept) -> str:
        old = self.s.perceived_keys.get(p.key)
        if not old:
            return "new"
        return "upgraded" if p.completeness > old["completeness"] + 0.05 else "repeat"

    def _integrate(self, p: Percept, obs: Observation, status: str, d: Delta):
        s = self.s
        if p.place and p.place not in s.known_places and p.place_label:
            s.known_places[p.place] = {"label": p.place_label, "kind": "spot", "room": obs.room, "first_seen": self.now}
            d.notes.append(f"new place known: {p.place_label}")
        if p.subject:
            is_new_object = p.subject not in s.known_objects
            o = s.known_objects.setdefault(p.subject, {"name": p.subject_label, "first_seen": self.now, "inspections": 0,
                                                      "uses": 0, "held": p.subject in s.inventory,
                                                      "last_known_place": None})
            o.setdefault("name", p.subject_label or p.subject)
            if is_new_object or o.get("first_seen") is None:
                o.setdefault("first_seen", self.now)
            if is_new_object:
                d.notes.append(f"new object known: {p.subject_label}")
                self._changed("known_objects", f"+{p.subject_label}")
            o["last_seen"] = self.now
            if obs.room and p.subject not in s.inventory and p.kind != "held":
                o["room"] = obs.room                       # where I last met it (so I can walk back for it)
            if p.kind == "sighting" and p.place and p.place_label:
                o["place_label"] = p.place_label
            if getattr(p, "hint", ""):
                o["hint"] = p.hint                        # what it LOOKS like it is for (food, water, a bed, a door...)
            if p.kind == "sighting" and p.mobile:
                o["looks_portable"] = True        # small and loose: visibly something one could pick up
            if p.kind == "sighting" and p.place and p.mobile:
                o["last_known_place"] = p.place
                self.update_belief(p.subject, "location", p.place, f"The {p.subject_label} is at {p.place_label}.",
                                   0.8 if p.completeness >= 1 else 0.6, "direct_observation", p.key)
            if p.kind == "sighting" and p.place and (p.mobile or p.loose):
                if p.completeness >= 1:
                    self.resolve_uncertainty(f"q:{p.place}.unidentified",
                                             f"It was {('an' if p.subject_label[:1] in 'aeiou' else 'a')} {p.subject_label}.")
        if p.holds:
            if p.holds not in s.inventory:
                s.inventory.append(p.holds)
                self._changed("inventory", f"+{self._name(p.holds)}")
            s.known_objects.setdefault(p.holds, {"name": p.subject_label})["held"] = True
            s.known_objects[p.holds]["last_known_place"] = "me"
            self.update_belief(p.holds, "location", "me", f"I have the {p.subject_label}.", 0.95,
                               "direct_observation", p.key)
        if p.releases:
            if p.releases in s.inventory:
                s.inventory.remove(p.releases)
                self._changed("inventory", f"-{self._name(p.releases)}")
            s.known_objects.setdefault(p.releases, {"name": p.subject_label})["held"] = False
            s.known_objects[p.releases]["last_known_place"] = obs.place
            self.update_belief(p.releases, "location", obs.place,
                               f"The {p.subject_label} is where I left it, at {obs.place_label}.", 0.9,
                               "direct_observation", p.key)

        if p.body:
            self._apply_body(p, d)
        if p.stale_minutes:
            self._age_beliefs(p.stale_minutes, d)

        if status != "repeat":
            s.perceived_keys[p.key] = {"completeness": p.completeness, "label": p.label, "time": self.now,
                                       "subject": p.subject}
        for k in p.resolves:
            self.resolve_uncertainty(k, p.text)

        if status != "repeat":
            if p.recognizable:
                self._recognize(p, d)
            if p.unresolved:
                self.open_uncertainty(f"q:{p.key}", p.unresolved, p.subject)
            if p.completeness < 0.99 and p.kind == "fact":
                self.open_uncertainty(f"partial:{p.key}", f"What does {p.label or 'it'} say, exactly?", p.subject,
                                      "I could only make out part of it.", goal=False)
            if p.kind == "sensation" and p.key.endswith(".too_dark"):
                self.open_uncertainty(f"partial:{p.key}", f"What am I missing about {p.label}?",
                                      p.subject, "It was too dark to make them out.", goal=False)
            if status == "upgraded" and p.completeness >= 0.99:
                self.resolve_uncertainty(f"partial:{p.key}", "I can read it clearly now.")
            self._link_topics(p, d)
        if p.belief:
            res = self.update_belief(p.subject or obs.room, p.belief["attr"], p.belief["value"], p.belief["text"],
                                     p.belief.get("confidence", 0.7), "direct_observation", p.key)
            if res == "revised":
                d.notes.append(f"belief revised: {p.belief['text']}")
            if p.belief["attr"] == "time_of_day":
                self.resolve_uncertainty("stale:time", "I've seen the sky again.")

    def _recognize(self, p: Percept, d: Delta):
        s, r = self.s, p.recognizable
        imp = s.impressions.get(r["tag"], {})
        strength, identity = imp.get("strength", 0.0), imp.get("identity")
        status = "known" if identity else ("familiar" if strength >= 0.3 else "unknown")
        pid = f"person:{r['tag']}"
        if pid not in s.known_people:
            s.known_people[pid] = {"label": r["label"], "description": p.gist, "first_seen": self.now,
                                   "identity": identity, "familiarity": strength, "pronoun": r.get("pronoun", "them"),
                                   "seen_in": p.subject}
            self._changed("known_people", f"{r['label']} ({status})")
        d.recognitions.append({"key": p.key, "label": r["label"], "pronoun": r.get("pronoun", "them"), "status": status,
                               "familiarity": strength, "identity": identity, "description": p.text})
        if status in ("unknown", "familiar"):
            self.open_uncertainty(f"who:{r['tag']}", f"Who is {r['label']} in the {self._name(p.subject)}?", p.subject,
                                  "Looks familiar, but I can't place it." if status == "familiar" else "")
        if status == "familiar":
            self.update_belief(pid, "familiarity", "familiar", f"I feel I've seen {r['label']} before, but I can't place "
                               f"{r.get('pronoun', 'them')}.", strength * 0.85, "impression", p.key)

    def _link_topics(self, p: Percept, d: Delta):
        if not p.topics:
            return
        for topic in p.topics:
            entries = self.s.topic_index.setdefault(topic, [])
            if p.subject:
                for e in [e for e in entries if e["subject"] and e["subject"] != p.subject]:
                    self._link_belief(p, e, topic, d)
            if not any(e["key"] == p.key for e in entries):
                entries.append({"key": p.key, "subject": p.subject, "gist": p.gist, "time": self.now})

    def _link_belief(self, p: Percept, other: dict, topic: str, d: Delta):
        a, b = sorted([p.subject, other["subject"]])
        about = f"link:{a}|{b}"
        existing = next((x for x in self.s.beliefs if x["about"] == about), None)
        topics = sorted(set((existing["value"] if existing else []) + [topic]))
        if existing and topic in existing["value"]:
            return
        conf = min(0.75, 0.3 + 0.15 * len(topics))
        text = (f"The {self._name(a)} and the {self._name(b)} seem to be connected "
                f"({self._name(p.subject)}: {p.gist}; {self._name(other['subject'])}: {other['gist']}).")
        self.update_belief(about, "connection", topics, text, conf, "inference", p.key)
        d.notes.append(f"inferred a connection: {self._name(a)} <-> {self._name(b)} ({topic})")

    def _reconcile_inventory(self, obs: Observation, held_before: list, d: Delta):
        """On resume: does what I *feel* in my hands match what I believed?"""
        actual = set(obs.extra.get("held", []))
        for oid in held_before:
            if oid not in actual:
                n = self._name(oid)
                self.s.inventory.remove(oid)
                self.s.known_objects[oid]["held"] = False
                self.update_belief(oid, "location", "unknown", f"I thought I had the {n}, but I don't - I'm not sure where it is.",
                                   0.4, "direct_observation")
                self.open_uncertainty(f"lost:{oid}", f"Where is the {n}? I thought I was holding it.", oid)
                d.notes.append(f"I thought I was holding the {n}, but I'm not.")
                d.voice.append(f"I thought I was holding the {n}, but I'm not. I don't know where it is.")
        for oid in actual:
            if oid not in held_before:
                n = self._name(oid)
                self.s.known_objects.setdefault(oid, {"name": n})["held"] = True
                self.open_uncertainty(f"unexplained:{oid}", f"Why am I holding the {n}? I don't remember picking it up.", oid)
                d.notes.append(f"I'm holding the {n}, but I don't remember picking it up.")
                d.voice.append(f"I'm holding the {n}, but I don't remember picking it up.")

    # ------------------------------------------------------------------ memory
    def _consolidate(self, obs: Observation, statuses: list, d: Delta):
        s = self.s
        episode = [p for p, st in statuses if st != "repeat" and p.narrate and p.salience >= MIN_MEMORY_SALIENCE and p.gist]
        if not episode:
            for m in s.memories:
                if obs.subject and m.get("subject") == obs.subject:
                    floor = 0.55 if obs.kind == "inspect" else 0.0       # a fresh, deliberate look re-fixes a faded memory
                    m["confidence"] = round(min(0.95, max(m["confidence"] + 0.03, floor)), 2)
                    m["refreshed"] = self.now
            return
        mid = f"m_{self._next('memory'):04d}"
        seed = s.extra.get("rng_seed", 0)
        kept, dropped = [], 0
        for p in episode:
            rnd = random.Random(f"{seed}:{mid}:{p.key}").random()
            if p.salience >= 0.6 or rnd < p.salience + 0.25:
                kept.append(p)
            else:
                dropped += 1
        if not kept:
            kept = [max(episode, key=lambda x: x.salience)]
            dropped = len(episode) - 1
        label = obs.subject_label or obs.room_label
        if obs.kind in ("take", "drop"):
            text = f"I remember {kept[0].gist}."
        else:
            verb = {"look": f"looking around {obs.room_label}", "arrive": f"walking over to {obs.place_label}",
                    "inspect": f"examining the {label}", "use": f"using the {label}",
                    "ambient": "noticing a change", "inventory": "checking my hands",
                    "body": "noticing how my body felt"}.get(obs.kind, "being here")
            text = f"I remember {verb}: " + "; ".join(p.gist.rstrip(".") for p in kept) + ("..." if dropped else ".")
        conf = sum(p.salience * p.completeness for p in kept) / len(kept) + 0.12
        mem = {"id": mid, "time": self.now, "text": text, "confidence": round(max(0.2, min(0.95, conf)), 2),
               "source": "self_action" if obs.kind in ("take", "drop", "arrive") else "direct_observation",
               "subject": obs.subject, "place": obs.place, "percept_keys": [p.key for p in kept],
               "event_ids": list(obs.event_ids), "incomplete": bool(dropped) or any(p.completeness < 0.99 for p in kept),
               "recalled": 0, "last_recalled": None, "tags": sorted({t for p in kept for t in p.topics})}
        upgraded_keys = {p.key for p in d.upgraded}
        for old in s.memories:   # a clearer view supersedes the older, hazier one
            if old.get("subject") == obs.subject and set(old["percept_keys"]) & upgraded_keys and not old.get("superseded_by"):
                old["superseded_by"] = mid
        s.memories.append(mem)
        d.memories.append(mem)
        self._prune_memories()
        self._changed("memories", f"{mid}: {text}")

    def _record_experience(self, obs: Observation, d: Delta):
        s = self.s
        s.recent_observations.append({
            "id": obs.id, "time": obs.time, "kind": obs.kind, "subject": obs.subject, "place": obs.place,
            "light": obs.light, "event_ids": list(obs.event_ids),
            "perceived": [p.text for p in obs.percepts if p.narrate][:6],
            "new": [p.key for p in d.new], "upgraded": [p.key for p in d.upgraded],
            "memory_formed": [m["id"] for m in d.memories]})
        del s.recent_observations[:-25]

    # ------------------------------------------------------------------ recall & retrieval
    def effective_confidence(self, m: dict) -> float:
        if m["source"] == "background":
            return m["confidence"]
        age = max(0, minutes_between(m.get("refreshed") or m["time"], self.now)) if self.now else 0
        life = HALF_LIFE_MINUTES * (7 if m["source"] == "diary" else 1)     # a diary entry outlasts the day it describes
        return round(m["confidence"] * 0.5 ** (age / life), 2)

    def _live_memories(self) -> list:
        return [m for m in self.s.memories if not m.get("superseded_by")]

    def recall(self, topic: Optional[str] = None, limit: int = 6) -> list:
        want = _tokens(topic) if topic else set()
        out = []
        for m in reversed(self._live_memories()):
            hay = _tokens(m["text"]) | set(m.get("tags", [])) | _tokens(self._name(m["subject"]) if m.get("subject") else "")
            if want and not (want & hay):
                continue
            eff = self.effective_confidence(m)
            m["recalled"] += 1
            m["last_recalled"] = self.now
            out.append({"kind": "memory", "id": m["id"], "text": m["text"], "confidence": eff,
                        "incomplete": m["incomplete"], "time": m["time"], "faded": eff < 0.15})
            if len(out) >= limit:
                break
        return out

    def retrieve(self, question: str, limit: int = 4) -> list:
        """Evidence relevant to a question - drawn only from the character's own state."""
        words = set(re.findall(r"[a-z0-9']+", question.lower()))
        q = _tokens(question)
        q_exp = set().union(*[EXPAND[w] for w in words if w in EXPAND]) if words & set(EXPAND) else set()
        if not q and not q_exp:
            return []
        s, found = self.s, []

        def add(kind, text, conf, hay_text, label_text="", **kw):
            score = len(q & _tokens(hay_text)) + len(q_exp & _tokens(label_text))
            if score:
                found.append({"kind": kind, "text": text, "confidence": conf, "score": score, **kw})

        def labels(keys):
            return " ".join(s.perceived_keys.get(k, {}).get("label", "") for k in keys)

        for m in self._live_memories():
            if m["source"] in ("conversation", "hearsay"):      # reachable via "what did I ask you?" - never mixed into topical answers
                continue
            eff = self.effective_confidence(m)
            add("memory", m["text"], eff, m["text"] + " " + " ".join(m.get("tags", [])) + " " + self._name(m.get("subject") or ""),
                labels(m["percept_keys"]), incomplete=m["incomplete"], faded=eff < 0.15)
        for b in s.beliefs:
            add("belief", b["text"], b["confidence"], b["text"])
        for u in s.uncertainties:
            if u["status"] == "open":
                add("uncertainty", u["question"], 0.0, u["question"] + " " + u["notes"], u["question"], notes=u["notes"])
        for pid, p in s.known_people.items():
            add("person", p["label"], p["familiarity"], p["label"] + " " + (p["description"] or ""), "person",
                identity=p["identity"], familiarity=p["familiarity"], pronoun=p["pronoun"])
        remembered = {k for m in self._live_memories() for k in m["percept_keys"]}
        for key, info in s.perceived_keys.items():
            if key not in remembered and info["label"]:
                add("forgotten", info["label"], 0.0, info["label"], info["label"])
        found.sort(key=lambda e: (-e["score"], KIND_PRIORITY[e["kind"]], -e["confidence"]))
        return found[:limit]

    # ------------------------------------------------------------------ conversations become memories
    QUESTION_RE = re.compile(r"(?i)\?\s*$|^\s*(what|who|where|when|why|how|which|do|does|did|is|are|was|were|can|could|have|has|"
                             r"tell|describe|explain)\b")

    def remember_talk(self, said: str, replied: str, told: bool = False):
        """What the person said (and what I answered) is remembered like anything else: timestamped, lossy, hearsay.
        It never becomes a BELIEF about the world - being told something is not seeing it."""
        said = re.sub(r"\s+", " ", said).strip()
        if not said:
            return
        asked = bool(self.QUESTION_RE.search(said)) and not told
        first = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", replied or "").strip())[0]
        if len(first) > 90:
            first = first[:90].rsplit(" ", 1)[0] + "..."
        if "The person" in first or '"' in first:               # an answer made of other conversations is not worth quoting
            first = ""
        if asked:
            first = first.rstrip(".")
            text = f'The person asked me: "{said[:140]}"' + (f' and I answered: "{first}"' if first else "")
            conf, src, tags = 0.8, "conversation", ["conversation", "person", "asked"]
        else:
            text = f'The person told me: "{said[:160]}"'
            conf, src, tags = 0.55, "hearsay", ["conversation", "person", "told"]
        mid = f"m_{self._next('memory'):04d}"
        self.s.memories.append({"id": mid, "time": self.now, "text": text, "confidence": conf, "source": src, "subject": None,
                                "place": None, "percept_keys": [], "event_ids": [], "incomplete": False, "recalled": 0,
                                "last_recalled": None, "tags": tags})
        self._prune_memories()
        self._changed("memories", f"{mid}: {text}")

    # ------------------------------------------------------------------ goals the person gives me
    GOAL_VERBS = {"read": "use", "use": "use", "try": "use", "play": "use", "open": "use", "inspect": "inspect", "examine": "inspect",
                  "study": "inspect", "check": "inspect", "look at": "inspect", "take": "take", "grab": "take", "pick up": "take",
                  "go to": "move", "visit": "move", "explore": "move", "walk to": "move", "head to": "move"}
    NUMWORDS = {"a": 1, "one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6}

    @classmethod
    def parse_goal(cls, text: str) -> Optional[dict]:
        t = text.lower().strip().rstrip(".!")
        t = re.sub(r"^(please |you should |you could |try to |i want you to |i'd like you to |go and |could you |can you |"
                   r"would you |start to |start by |make sure you |i want you to )+", "", t).strip()
        m = re.match(r"^(?:watch|see)\s+(?:(\d+|a|one|two|three|four|five|six)\s+)?(?:more\s+)?(?:films?|movies?|dvds?)\b", t)
        if m:
            n = m.group(1)
            return {"cmd": "watch", "arg": "", "count": int(n) if n and n.isdigit() else cls.NUMWORDS.get(n or "a", 1)}
        verbs = sorted(cls.GOAL_VERBS, key=len, reverse=True)
        m = re.match(r"^(" + "|".join(re.escape(v) for v in verbs) + r")\s+(?:the\s+|a\s+|an\s+|my\s+)?(.+?)"
                     r"(?:\s+(\d+|once|twice|two|three|four|five)(?:\s+times)?)?$", t)
        if m and m.group(2):
            n = m.group(3)
            return {"cmd": cls.GOAL_VERBS[m.group(1)], "arg": m.group(2).strip(), "count": int(n) if n and n.isdigit()
                    else cls.NUMWORDS.get(n or "once", 1)}
        return None

    @staticmethod
    def _words(text: str) -> set:
        return {re.sub(r"s$", "", w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ("the", "a", "an", "my")}

    def _goal_count(self, act: dict) -> int:
        s = self.s
        if act["cmd"] == "watch":
            return sum(v for k, v in s.extra.get("activity_totals", {}).items() if k.startswith("watched "))
        want = self._words(act["arg"])
        return sum(v for k, v in s.extra.get("attempts", {}).items()
                   if k.startswith(act["cmd"] + ":") and want <= self._words(k.split(":", 1)[1]))

    def goal_progress(self, g: dict) -> int:
        return max(0, self._goal_count(g["act"]) - g.get("baseline", 0)) if g.get("act") else 0

    def add_user_goal(self, text: str) -> dict:
        act = self.parse_goal(text)
        gid = f"ug_{self._next('goal'):03d}"
        g = {"id": gid, "text": text.strip().rstrip("."), "status": "active", "priority": 2, "created": self.now,
             "link": f"person:{gid}", "source": "person", "act": act, "baseline": self._goal_count(act) if act else 0}
        self.s.goals.append(g)
        self._changed("goals", f"new (from the person): {g['text']}")
        return g

    def user_goals(self) -> list:
        return [g for g in self.s.goals if g.get("source") == "person"]

    def _check_goals(self):
        for g in self.s.goals:
            if g.get("source") == "person" and g["status"] == "active" and g.get("act"):
                if self.goal_progress(g) >= g["act"].get("count", 1):
                    g["status"] = "done"
                    g["finished"] = self.now
                    self._changed("goals", f"done: {g['text']}")
                    self.s.extra.setdefault("goal_news", []).append(g["text"])

    def conversation_summary(self, limit: int = 6) -> str:
        talk = [m for m in self._live_memories() if m["source"] in ("conversation", "hearsay")][-limit:]
        if not talk:
            return "We haven't really talked yet."
        asked = [re.match(r'The person asked me: "(.*?)"(?: and I answered|$)', m["text"]) for m in talk if m["source"] == "conversation"]
        told = [re.match(r'The person told me: "(.*)"$', m["text"]) for m in talk if m["source"] == "hearsay"]
        parts = []
        if any(asked):
            parts.append("You asked me " + "; then ".join(f'"{a.group(1)}"' for a in asked if a) + ".")
        if any(told):
            parts.append("You also told me " + "; and ".join(f'"{t.group(1)}"' for t in told if t) + ".")
        return " ".join(parts)

    def pop_goal_news(self) -> list:
        news, self.s.extra["goal_news"] = self.s.extra.get("goal_news", []), []
        return news

    # ------------------------------------------------------------------ the diary
    _TENSE = [(r"^examining the ", "I examined the "), (r"^using the ", "I used the "), (r"^looking around ", "I looked around "),
              (r"^walking over to ", "I walked over to "), (r"^noticing a change", "I noticed a change"),
              (r"^noticing how my body felt", "I noticed how my body felt"), (r"^checking my hands", "I checked my hands"),
              (r"^being here", "I was here")]

    def as_fact(self, text: str) -> str:
        """'I remember examining the notebook: a star chart.' -> 'I examined the notebook - a star chart'."""
        t = re.sub(r"^I remember\s+", "", text).rstrip(".…").strip()
        for pat, rep in self._TENSE:
            if re.match(pat, t):
                t = re.sub(pat, rep, t, count=1)
                break
        else:
            t = "I remember " + t
        return t.replace(": ", " - ", 1)

    def notable_memories(self, day: Optional[str] = None, after_id: int = 0, limit: int = 6) -> list:
        """What actually HAPPENED, as opposed to routine upkeep: his clearest memory per thing, minus eating/drinking/sleeping."""
        upkeep = {n.lower() for need in ("thirst", "hunger", "fatigue") for n in self.s.extra.get("relief", {}).get(need, [])}
        best: dict = {}
        for m in self._live_memories():
            if m["source"] not in ("direct_observation", "self_action"):
                continue
            if day and m["time"][:10] != day:
                continue
            if int(m["id"].split("_")[1]) <= after_id:
                continue
            low = m["text"].lower()
            if any(k in low for k in ("noticing how my body felt", "checking my hands")) or any(f"using the {n}:" in low for n in upkeep):
                continue
            key = m.get("subject") or m["id"]
            if key not in best or m["confidence"] > best[key]["confidence"]:
                best[key] = m
        before = {m.get("subject") for m in self._live_memories() if m.get("subject") and day and m["time"][:10] < day}
        top = sorted(best.values(), key=lambda m: -(m["confidence"] - (0.3 if m.get("subject") in before else 0.0)))[:limit]
        return sorted(top, key=lambda m: m["time"])

    def day_facts(self, day: str) -> list:
        """Short first-person facts about one simulated day, taken ONLY from my own record of it: what happened, not tallies."""
        notable = self.notable_memories(day=day, limit=8)
        changes = [m for m in notable if m["text"].startswith("I remember noticing a change")]
        out = [self.as_fact(m["text"]) for m in notable if m not in changes][:6]
        if changes:                                                     # sounds and weather: one line, not one per sound
            out.append("I noticed the world change around me - " + "; ".join(
                re.sub(r"^I remember noticing a change:\s*", "", m["text"]).rstrip(".…") for m in changes[:3]))
        for g in self.s.goals:
            if g.get("source") == "person" and str(g.get("finished", ""))[:10] == day:
                out.append(f"I finished what the person asked of me: {g['text']}")
        mems = [m for m in self._live_memories() if m["time"][:10] == day]
        qs = [re.match(r'The person asked me: "(.*?)"(?: and I answered|$)', m["text"]) for m in mems if m["source"] == "conversation"]
        qs = [q.group(1) for q in qs if q]
        told = [re.match(r'The person told me: "(.*)"$', m["text"]) for m in mems if m["source"] == "hearsay"]
        told = [t.group(1) for t in told if t]
        if qs:
            out.append(f"The person asked me {len(qs)} question{'s' if len(qs) > 1 else ''}, including \"{qs[-1][:70]}\"")
        if told:
            out.append(f"The person told me: \"{told[-1][:80]}\"")
        acts = self._collapse_acts(list(self.s.extra.get("day_acts", {}).get(day, {}).items()), keep_total=True)
        if acts and acts[0][1] >= 4:                                   # one line about the rhythm of the day, never a tally
            out.append(f"For much of the day I {acts[0][0]}")
        from datetime import date, timedelta
        recent = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        wondering = [u["question"] for u in self.s.uncertainties if u["status"] == "open" and str(u.get("raised", ""))[:10] >= recent]
        if wondering:
            q = wondering[-1].rstrip("?. ")
            out.append(f"I'm still wondering: {q[:1].lower() + q[1:]}?")
        dedup, keys = [], set()
        for f in out:
            if f.lower() not in keys:
                keys.add(f.lower())
                dedup.append(f)
        return dedup

    def add_diary(self, day: str, text: str):
        mid = f"m_{self._next('memory'):04d}"
        self.s.memories.append({"id": mid, "time": self.now, "text": f"Diary, {day}: {text}", "confidence": 0.85, "source": "diary",
                                "subject": None, "place": None, "percept_keys": [], "event_ids": [], "incomplete": False,
                                "recalled": 0, "last_recalled": None, "tags": ["diary", "day", "yesterday", "night", "morning"]})
        self._changed("memories", f"{mid}: diary for {day}")

    # ------------------------------------------------------------------ attempts (for autonomy)
    def note_activity(self, kind: str, when: str):
        log = self.s.extra.setdefault("activity", [])
        log.append([when, kind])
        del log[:-150]

    def note_attempt(self, cmd: str, name: str, ok: bool):
        if not ok:
            self.s.extra.setdefault("last_fail", {})[f"{cmd}:{name}"] = self.now
        bucket = self.s.extra.setdefault("attempts" if ok else "failures", {})
        k = f"{cmd}:{name}"
        bucket[k] = bucket.get(k, 0) + 1
        if ok and cmd == "use":                                  # it works now: old failures (empty, locked...) no longer count
            self.s.extra.get("failures", {}).pop(k, None)
        if ok:
            self.s.extra.setdefault("last_att", {})[k] = self.now
        self._check_goals()

    # ------------------------------------------------------------------ plain-data snapshot
    def snapshot(self) -> dict:
        s = self.s
        recent = [{"text": m["text"], "confidence": self.effective_confidence(m)}
                  for m in self._live_memories()[-5:]]
        return {
            "identity": dict(s.identity), "location": dict(s.current_location), "activity": s.current_activity,
            "inventory": [self._name(i) for i in s.inventory],
            "goals": [g["text"] for g in s.goals if g["status"] == "active"],
            "goal_acts": [{"id": g["id"], "text": g["text"], **g["act"], "progress": self.goal_progress(g)}
                          for g in s.goals if g.get("source") == "person" and g["status"] == "active" and g.get("act")],
            "open_uncertainties": [{"question": u["question"], "notes": u["notes"]} for u in s.uncertainties
                                   if u["status"] == "open"],
            "beliefs": [{"text": b["text"], "confidence": b["confidence"]} for b in s.beliefs
                        if b["attr"] != "location" and not b["about"].startswith("link:")][-8:]
                       + [{"text": b["text"], "confidence": b["confidence"]} for b in s.beliefs
                          if b["about"].startswith("link:")][-3:],
            "recent_memories": recent,
            "known_objects": [{"name": o.get("name") or oid, "held": o.get("held", False), "inspections": o.get("inspections", 0),
                               "uses": o.get("uses", 0), "portable": o.get("looks_portable", False),
                               "room": o.get("room"), "place_label": o.get("place_label"), "hint": o.get("hint", "")}
                              for oid, o in s.known_objects.items()],
            "flags": dict(s.flags),
            "needs": {n: v["band"] for n, v in s.needs.items()},
            "relief": {n: list(v) for n, v in s.extra.get("relief", {}).items()},
            "states": {self._name(b["about"]): b["value"] for b in s.beliefs
                       if b["attr"] == "state" and b["about"] in s.known_objects},
            "last_light": (s.recent_observations[-1]["light"] if s.recent_observations else None),
            "tally": self._collapse_acts(list(s.extra.get("activity_totals", {}).items()), keep_total=True)[:10],
            "since": (s.memories[0]["time"] if s.memories else self.now),
            "now": self.now, "last_fail": dict(s.extra.get("last_fail", {})),
            "attempts": dict(s.extra.get("attempts", {})), "failures": dict(s.extra.get("failures", {})),
            "last_att": dict(s.extra.get("last_att", {})), "room_last": dict(s.extra.get("room_last", {})),
            "tags": dict(s.extra.get("act_tags", {})),
            "rooms": {rid: p.get("label", rid) for rid, p in s.known_places.items() if p.get("kind") == "room"},
            "spots": [{"label": p.get("label", ""), "room": p.get("room")} for p in s.known_places.values() if p.get("kind") == "spot"],
            "stale": self._stale_names(),
        }

    def _stale_names(self) -> list:
        """Things I once looked at whose memory has faded below recall: the honest reason to go and look again."""
        best: dict = {}
        for m in self._live_memories():
            if m.get("subject") and m["source"] in ("direct_observation", "self_action"):
                best[m["subject"]] = max(best.get(m["subject"], 0.0), self.effective_confidence(m))
        faded = sorted((c, oid) for oid, c in best.items() if c < 0.2 and oid in self.s.known_objects)
        return [self._name(oid) for _, oid in faded]


# ####################################################################################################
##### SECTION: viewer   (the State Viewer window; needs tkinter, and only when it is started)
# ####################################################################################################

try:
    from tkinter import filedialog, messagebox, ttk      # (tk itself is imported at the top of this file)
except ImportError:
    filedialog = messagebox = ttk = None

APP = "State Viewer"
DEFAULT_FILE = os.path.join(SAVE_HOME, "state_test.json")
POLL_MS = 1000

FIELD_COLOURS = {   # state_changes "field" -> colour in the Thoughts list
    "beliefs": "#7fc8ff", "memories": "#5dff8a", "uncertainties": "#ffd866", "goals": "#c9a4ff",
    "needs": "#ff7b72", "current_location": "#4fd1c5", "current_activity": "#a9b1bb", "known_objects": "#ffb347",
    "known_people": "#ffb347", "inventory": "#ffb347",
}
FIELD_WORDS = {"beliefs": "belief", "memories": "memory", "uncertainties": "question", "goals": "goal", "needs": "feeling",
               "current_location": "location", "current_activity": "activity", "known_objects": "noticed",
               "known_people": "person", "inventory": "holding"}
BAND = {0: "fine", 1: "a little", 2: "quite", 3: "very"}


def _short_time(ts) -> str:
    s = str(ts or "")
    return s[5:16].replace("T", " ") if len(s) >= 16 else s


def _slug(text: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", text.lower()))[:30] or "thing"


# ---------------------------------------------------------------------------------------------- the data side
class StateFile:
    """Loads a state file into a CharState + Character, works out 'now', and (for test files) feeds and saves."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.raw: dict = {}
        self.mtime = 0.0
        self.writable = True
        self.fmt = ""
        self.cs = None
        self.char = None
        self.clock = None
        self.events = None

    # ---- reading
    def load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict):
                raise ValueError("this file is not a state file (it isn't a JSON object)")
            self.mtime = os.path.getmtime(self.path)
        else:
            d = {}
        if "character" in d:
            cdict = d["character"]
        elif "identity" in d:                         # a bare CharState dump
            cdict = d
        elif not d:
            cdict = None
        else:
            raise ValueError("no 'character' section in this file - it doesn't look like a state save")
        world = d.get("world") or {}
        self.fmt = str(d.get("format", ""))
        # a save written by a running app: look, don't touch
        self.writable = not bool(world.get("objects"))
        self.clock = SimClock.from_dict(world["clock"]) if isinstance(world.get("clock"), dict) else None
        self.cs = CharState.from_dict(cdict) if cdict else new_character(DEFAULT_START, {"name": "Character"})
        if self.clock is None:
            self.clock = SimClock(self._latest_time() or DEFAULT_START)
        self.events = EventLog.from_dict(d["events"]) if isinstance(d.get("events"), dict) else EventLog()
        self.char = Character(self.cs)
        self.char.now = self.clock.now
        self.raw = d

    def _latest_time(self) -> str:
        times = [c.get("time") for c in self.cs.state_changes] + [m.get("time") for m in self.cs.memories]
        times = [t for t in times if isinstance(t, str) and len(t) == 16]
        return max(times) if times else ""

    def changed_on_disk(self) -> bool:
        try:
            return os.path.getmtime(self.path) != self.mtime
        except OSError:
            return False

    # ---- writing (test files only)
    def save(self):
        if not self.writable:
            raise PermissionError("this file belongs to a running app - the viewer only looks at it")
        d = dict(self.raw)
        d.update({"format": d.get("format", "statemanager-test"), "character": self.cs.to_dict(),
                  "events": self.events.to_dict(), "world": {"clock": self.clock.to_dict()}})
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)
        self.raw = d
        self.mtime = os.path.getmtime(self.path)

    def feed(self, subject: str, text: str, believe: bool, question: str, salience: float) -> list:
        """Turn one line typed by the person into an Observation and let the state engine take it in."""
        now = self.clock.now
        sid = _slug(subject) if subject else None
        p = Percept(key=f"{sid or 'ambient'}.{int(time.time() * 1000)}", text=text, subject=sid, subject_label=subject,
                       gist=text.rstrip(".")[:90], salience=salience, kind="fact" if subject else "ambient",
                       label=f"something about the {subject}" if subject else "something I noticed",
                       unresolved=question or None,
                       belief={"attr": "note", "value": text, "text": text if text.endswith(".") else text + ".",
                               "confidence": 0.8} if (believe and subject) else None)
        obs = Observation(kind="inspect" if subject else "ambient", time=now, subject=sid, subject_label=subject,
                             percepts=[p])
        ev = self.events.add(now, "world", "OBSERVED", "me", sid, None, text=text)
        obs.event_ids = [ev.id]
        delta = self.char.perceive(obs, now)
        if delta.new:
            self.events.add(now, "actor", EV.INFORMATION_LEARNED, "me", sid, None, new=[x.key for x in delta.new])
        self.save()
        return delta.lines()

    def advance(self, minutes: int):
        self.clock.advance(minutes)
        self.char.now = self.clock.now
        self.events.add(self.clock.now, "world", EV.TIME_PASSED, minutes=minutes)
        self.save()


# ---------------------------------------------------------------------------------------------- the window
# ---------------------------------------------------------------------------------------------- the window
# One window, one fixed layout:   header (who / when / file)   |   sidebar (pages)  +  page   |   feed bar   |   status
SV_BG, SV_SIDE, SV_PANEL, SV_CARD, SV_LINE = "#0d1015", "#11151b", "#151a21", "#1b212a", "#272e39"
SV_FG, SV_DIM, SV_ACCENT, SV_SEL = "#e8ebf0", "#8b95a3", "#5b9cff", "#223552"
SV_GOOD, SV_WARN = "#4ade80", "#fbbf24"


# Every formula the script uses, for the "Engine formulas" page of the viewer.
# (section, [(name, equation, where it lives), ...])
FORMULAS = (
    ("Motion and integration", (
        ("Velocity update (gravity, sideways push)", "vy <- (vy + g*dt) * 0.9997      vx <- (vx + ax*dt) * 0.9997\n"
         "g = 700 px/s^2,  dt = 1/240 s,  ax = earthquake / wind acceleration", "k_forces"),
        ("Spin damping", "w <- w * 0.998 (boxes)     w <- w * 0.994 (circles)", "k_forces"),
        ("Semi-implicit Euler", "v <- v + a*dt           (velocity first)\nx <- x + v*dt           (then position, with the NEW velocity)\n"
         "angle <- angle + w*dt", "k_forces, k_integrate"),
        ("Speed cap", "if |v| > 1200:   v <- v * 1200 / |v|", "k_integrate"),
        ("Mass and inertia", "box:     m = density * (2hw)(2hh)     I = m * ((2hw)^2 + (2hh)^2) / 12\n"
         "circle:  m = density * pi * r^2        I = 1/2 * m * r^2\ninverse mass = 1/m,  inverse inertia = 1/I  (0 for fixed bodies)", "Body"),
        ("Newton's law of universal gravitation", "F = G * m1 * m2 / r^2\n"
         "acceleration of body i:  a_i = SUM over j of  G * m_j / (r^2 + 8^2)  *  (direction to j)\n"
         "(8 px softening keeps it finite when bodies overlap; fixed bodies do not attract)", "k_attract"),
        ("Kinetic energy", "KE = 1/2 * m * v^2  +  1/2 * I * w^2", "k_kinetic, kinetic_energy()"),
        ("Momentum", "p = m * v      (stored as inverse mass; contact impulses are equal and opposite)", "k_solve (implicit)"),
    )),
    ("Collisions and contacts", (
        ("Velocity of a point on a spinning body", "v_point = v + w x r = ( vx - w*ry ,  vy + w*rx )", "k_narrow, k_solve"),
        ("2D cross and dot product", "a x b = ax*by - ay*bx          a . b = ax*bx + ay*by\n"
         "torque arm  r x n = rx*ny - ry*nx", "k_narrow, k_solve"),
        ("Effective mass at a contact", "kn = 1/mA + 1/mB + (rA x n)^2 / IA + (rB x n)^2 / IB      mn = 1/kn\n"
         "same again along the tangent for friction (kt, mt)", "k_narrow"),
        ("Normal impulse", "dP = mn * (bias - v_rel . n)       P <- max(P + dP, -cohesion)\n"
         "vA -= dP*n / mA     vB += dP*n / mB     wA -= (rA x dP*n) / IA     wB += (rB x dP*n) / IB", "k_solve"),
        ("Penetration correction", "bias = min( beta/dt * (depth - slop), 200 )      beta = 0.2, slop = 0.5 px\n"
         "if the bodies are still apart:  bias = depth / dt   (speculative contact)", "k_narrow"),
        ("Restitution (bounciness)", "e = min(eA, eB)\nif v_rel . n < -60:   bias = max( bias, -e * v_rel . n )", "k_narrow"),
        ("Coulomb friction", "mu = sqrt(muA * muB)\ndPt = -mt * (v_rel . t)      |Pt| <= mu * max(Pn, 0) + cohesion", "k_narrow, k_solve"),
        ("Cement / joint strength", "cohesion = F * cure * dt * strength      cure = 0.5 + 0.5 * min(1, (t - t_born) / 3 s)\n"
         "cracks when   Pn <= -0.98*cohesion   or   |Pt| >= 0.98 * (mu*Pn + cohesion)\n"
         "strength multiplier: nail 0.3, weld 1.0, anchor 3.0", "k_narrow, k_store_and_break"),
        ("Hardest hit", "peak = max over contacts of  Pn * inverse mass     (largest velocity change in one step)", "k_peak"),
        ("Rotation of a body's axes", "cos(angle), sin(angle) rebuilt after every step; world = R * local\n"
         "rx = lx*cos - ly*sin     ry = lx*sin + ly*cos", "k_integrate, k_grab"),
        ("Box overlap test (separating axes)", "project both boxes' corners on each box's two axes;\nthey overlap only if every projection overlaps by more than the tolerance", "obb_overlap"),
    )),
    ("Constraints and interaction", (
        ("Rope", "d = |x - pivot|\nif d > L:   x <- pivot + n*L      v <- v - (v . n) * n   (only the outward part)", "k_rope"),
        ("Mouse grab (spring and damper, Hooke's law)", "F = K*e - C*v_point        K = 90, C = 16,   e = mouse - grab point (capped at 60 px)\n"
         "|F| capped at 6000", "k_grab"),
        ("Sleeping", "a body sleeps when |v| < 10 and |w| < 0.12 have held for 0.6 s\n"
         "whole touching islands sleep together", "k_integrate, k_sleep_*"),
    )),
    ("Breakable joints and fracture (2D)", (
        ("Joint: keep two anchors together", "each joint holds a point on body A and a point on body B (rA, rB rotated by each body's angle)\n"
         "gap e = (xB + rB) - (xA + rA)       bias = e * beta / dt      beta = 0.25, |bias| <= 600 px/s\n"
         "relative anchor velocity  dv = (vB + wB x rB) - (vA + wA x rA);   solve the 2x2 system  K * P = -(dv + bias)\n"
         "K11 = 1/mA + 1/mB + ray^2/IA + rby^2/IB    K22 = 1/mA + 1/mB + rax^2/IA + rbx^2/IB    K12 = -(rax*ray/IA + rbx*rby/IB)\n"
         "P is applied equal and opposite (linear and spinning parts); the impulse is summed over the substep as (Px, Py)", "k_joint_solve"),
        ("Joint: keep the relative angle", "err = (angleB - angleA) - rest angle        dw = wB - wA\n"
         "Pm = -(dw + err * beta / dt) / (1/IA + 1/IB)        wA -= Pm / IA      wB += Pm / IB", "k_joint_solve"),
        ("Joint solve order", "per substep (dt = 1/240 s): contacts first, then the joints in colour groups (no two joints of a colour share a body, so they run in parallel),\n"
         "then 6 extra joint-only sweeps so a load travels further along a chain.  Joints applied last win over contacts.\n"
         "warm start: each substep begins by re-applying 0.7 of the last substep's joint impulse (all of it overshoots)", "k_joint_warm, _joint_sweep"),
        ("Joint strength from material", "width w = length of the face where the two pieces meet\n"
         "Tmax = T * w      Smax = S * w      Mmax = M * w^2       (T, S per px of width; M per px^2, about T / 4)\n"
         "nail 1500/2500/375    wood 3000/2500/750    concrete 1000/4000/250    glass 1500/1500/375\n"
         "weld 2e4/1.5e4/5000    steel 3e4/2.5e4/7500    anchor 1e5/8e4/25000    mortar 600/1200/150\n"
         "the Joints x slider multiplies Tmax, Smax and Mmax of every joint by one factor", "JOINT_MATERIALS, weld_joints, set_joint_scale"),
        ("Joint load", "force = impulse / dt          tension T = -(F . n)  (n = joint axis, A toward B)        shear S = |F x n|        moment M = |Pm| / dt\n"
         "each is smoothed:  x <- x + (new - x) * 0.35     (the raw load jitters from step to step)", "k_joint_check"),
        ("Joint breaking criterion", "u = (T / Tmax)^2 + (S / Smax)^2 + (M / Mmax)^2        (tension only counts when pulling apart)\n"
         "wear: if u > 0.3:   damage += dt * fatigue * (u - 0.3)\n"
         "effective load  eff = u / (1 - 0.6 * min(damage, 1))\n"
         "the joint snaps when  eff >= 1  or  damage >= 1   (not in the first 0.3 s after it was made); the pair then collides normally", "k_joint_check"),
        ("Joint pairs do not collide", "while a joint holds, the two bodies skip each other in the contact test (an adjacency list of at most 12 joints per body); it is cleared when the joint snaps", "k_narrow, add_joint"),
        ("Fracture trigger", "peak = largest single-contact velocity change a body got in one readback interval (px/s)\n"
         "size = sqrt(4 hw hh)  (circle: 1.77 r)       limit = material limit * min(4, max(1, ref size / size)) * Materials x\n"
         "the body breaks when  peak >= limit      ratio = peak / limit\n"
         "wood 420   glass 150   concrete 380   stone 650   brick 560   steel never (ductile)", "k_peak, fracture_step"),
        ("Crack or shatter", "if the material can crack and ratio < shatter (wood 1.7, concrete 1.6, stone 1.5):  2 pieces on a weak joint (crack_scale 0.35)\n"
         "otherwise it fragments:  wood = strips along the grain,  glass = slivers fanning out from the impact point,  concrete/stone = 4-7 blocks,  brick = 2-3 chunks\n"
         "a piece too small to break (below min_piece) just becomes dust and disappears", "_fracture_one"),
        ("Fragments keep the mass", "density of each fragment = parent mass / sum( 4 * hw * hh ) over the fragments\n"
         "fragment velocity = v + w x r + pop      pop = pop_k * min(peak, 2 * limit) * U(0.5, 1), pointing away from the impact point\n"
         "fragments cannot break again for 0.25 s;  each is a size ratio smaller, so it is harder to break", "_fracture_one"),
        ("Fragments are re-welded weaker", "neighbouring fragments are joined at the middle of the face they share with the material's joint x scale\n"
         "frag_scale 0.5 (wood) / 0.3 (concrete, stone), lat_scale 0.25 along the grain of wood, crack_scale 0.35;  glass and brick pieces fly free\n"
         "joints that held the parent now hold the nearest fragment:  d = distance from the anchor to each fragment's rectangle", "_fracture_one"),
        ("Where joints are made (weld)", "for every touching pair: boxes with parallel sides meet along a face of width w = overlap of the two sides\n"
         "anchor = middle of that face;  pieces closer than 3 px count as touching;  the ground gets 'anchor' joints,\n"
         "brick walls use mortar between bricks, so the mortar fails before the brick does", "weld_joints, _parallel_face"),
        ("Blast", "every free body within R of the blast:   dv = P * (1 - d / R)  directly away from the centre  (d = distance)", "kick"),
    )),
    ("Particles (position-based fluid / sand)", (
        ("Predict", "v <- v + g*dt        x* = x + v*dt", "k_p_predict"),
        ("Overlap push", "pen = 2r - |xi - xj|        correction = n * pen / 2   (each particle takes its half)", "k_p_project"),
        ("Particle friction", "t = relative motion minus its part along n\ncorrection -= t * min(1, mu*pen / |t|) * 0.5", "k_p_project"),
        ("Relaxation", "x* <- x* + 0.7 * correction     (correction capped at one radius)", "k_p_apply"),
        ("Velocity from positions", "v = (x* - x) / dt * 0.9998      if |v| < 15:  v *= 0.9      then cap at 1200", "k_p_update"),
    )),
    ("Destruction lab (chunks and bonds)", (
        ("Bond spring and damper", "along each live bond:  F = k * (1 - 0.6*damage) * (L - L0)  +  c * (v_rel . n)\n"
         "edge (diagonal) bonds use half of k;   c = 0.2 * sqrt(k * m)", "k_lab_bonds"),
        ("Strain", "strain = (L - L0) / L0", "k_lab_bonds"),
        ("Plastic yielding (steel)", "elastic strain  ee = strain - ep        yield limit  y = ey*st + hard*|ep|\n"
         "if |ee| > y:   ep += (|ee| - y) / (1 + hard)      (the excess becomes permanent bend)\n"
         "force uses the elastic part:  F = k (L - L0 (1 + ep)) ;  the bond snaps only when |ep| > ep_max * st", "k_lab_bonds"),
        ("Stable time step", "dt < ~ 2 / w_max,   w_max ~ sqrt(k_bond * neighbours / m)   so a stiffer or lighter material needs more steps\n"
         "steps per frame needed = 24 * sqrt( (k/rho) / (k_stone/rho_stone) / 1.3 )  (at least 24)", "DestructionLab.build"),
        ("Damage and breaking", "above the yield strain:   damage += rate * dt * (strain - yield) / (break - yield)\n"
         "a bond snaps when strain > break strain, or strain < -3 * break strain, or damage >= 1\n"
         "every bond has its own random strength that scales these limits\n"
         "checked in the same pass as the spring force; each end of a bond updates its own copy (same strain at both ends)", "k_lab_bonds"),
        ("Chunk-to-chunk contact and ground", "pen = size - distance        F = max( Kc*pen - Cc*(v_rel . n), 0 )  along n\n"
         "friction opposes sliding with  min( mu*F, Cc*|v_slide| )        ground: the same with pen = size/2 - y", "k_lab_contacts"),
        ("Hit impulse", "dv = speed * (1 - d/R)^2 * (m_ref / m)     along the hit direction (hammer, pick)\n"
         "or away from the point (blast);  d = distance from the hit, R = tool radius", "k_lab_impulse"),
        ("Gravity ease-in", "g = 9.81 * min(1, t / 0.8 s)      so a newly placed object settles instead of dropping", "DestructionLab.step"),
        ("Kinetic energy", "KE = sum over chunks of  1/2 * m * v^2", "k_lab_kinetic"),
        ("Pieces", "pieces = groups of chunks still joined by live bonds (repeated min-label spreading)", "DestructionLab._components"),
        ("Shading", "normal = direction away from the centre of the live neighbours\nbrightness = 0.3 + 0.7 * max(0, normal . light)\n"
         "colour darkens and reddens with damage = average over the chunk's bonds (broken = 1)", "k_lab_surface, k_lab_colour"),
        ("Camera", "screen x = cx + f * (p . right) / (p . forward)      screen y = cy - f * (p . up) / (p . forward)\n"
         "orbit: eye = target + dist * (sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch))", "k_lab_depth, LabCamera"),
        ("Laser: heat", "the beam adds   h += power * dt * (1 - d/R)   to chunks inside it, near the first surface it meets\n"
         "heat conducts:   h += dt * cond * (mean neighbour h - h)        and cools:   h -= dt * cool * h\n"
         "wood that is alight (h > 0.12) makes its own heat:   h += dt * burn   -- so fire spreads", "k_lab_laser, k_lab_heat_a"),
        ("Laser: damage and burning", "above h = 0.4 every bond of the chunk wears:   damage += rate * dt * (h - 0.4)\n"
         "at h >= 1 the chunk burns away: its bonds are cut and it is removed  (wood, which does not melt)", "k_lab_heat_b"),
        ("Laser: melting", "stone melts at h = 0.70, glass 0.55, clay 0.80 (wood burns instead):  at h >= melt every bond of the chunk is cut\n"
         "and the chunk becomes liquid.  It has no bonds, so it no longer conducts, and it cools slower:  cool * 0.35\n"
         "it sets again, as a loose solid chunk, when h < 0.7 * melt", "k_lab_heat_b, k_lab_heat_a"),
        ("Liquid motion (the particle solver, in 3D)", "predict:     v += (F/m + a) dt        x* = x + v dt          (k_lab_integrate)\n"
         "overlap push:   pen = size - |xi - xj|     correction = n * pen * share     share = 1/2 liquid-liquid, 1 liquid-solid\n"
         "friction:   t = relative motion minus its part along n      correction -= t * min(1, mu * pen / |t|) * share      mu = the material's grip (0.04 aluminium .. 0.15 clay)\n"
         "relaxation: x* += 0.7 * correction (at most one radius), 3 passes;  ground y = size/2 with the same friction on the slide\n"
         "thickness:   v *= exp(-visc dt (1 + 2 (1 - t)))   t = how far above the setting heat (cooler = stiffer)   visc: aluminium 0.8, slag 3, clay 4.5, glass 6 /s\n"
         "velocity from positions:   v = (x* - x) / dt * 0.9998 ;  if |v| < 0.2 m/s: v *= 0.9 ;  capped at 6 m/s\n"
         "only while something is hot or liquid; a liquid chunk is drawn as a glowing sphere", "k_lab_melt_project, k_lab_melt_apply, k_lab_melt_update"),
        ("Cannonball","a ball of 19 iron chunks (density 7800) fired from the camera toward the clicked point:  v = speed * direction\n"
         "its bonds never break; it hits with momentum p = m v and energy 1/2 m v^2", "DestructionLab.fire"),
        ("Wind drag", "u = wind - v          F = 1/2 * rho * Cd * A * |u| * u          rho = 1.2 * 10 (a game gale), Cd = 1.1, A = chunk face\n"
         "only chunks with an exposed side feel it; each gusts:  wind * (1 + 0.25 * sin(2.5 t + phase))", "k_lab_wind"),
        ("Earthquake", "the ground shakes sideways:   a = A * sin(2 pi f t)    f = 2.5 Hz\nadded to every chunk's acceleration (the ground stays still in the shaking frame)", "DestructionLab.step"),
        ("Gravity well (Newton again)", "F = G * M * m / (r^2 + 0.2^2)  pulling each chunk toward the held point", "k_lab_well"),
    )),
    ("State manager", (
        ("Belief ageing (sleep)", "confidence <- max( 0.15,  confidence * 0.5 ^ (minutes / 480) )", "Character._age_beliefs"),
        ("Memory fading", "effective = confidence * 0.5 ^ (age / life)\nlife = 1 day (7 days for diary entries); half-life is simulated time", "effective_confidence"),
    )),
    ("Language model (GGUF runner)", (
        ("RMS norm", "y = x / sqrt( mean(x^2) + eps ) * weight", "rms_norm"),
        ("Rotary position (RoPE)", "freq_i = base ^ (-2i / dim)       angle = position * freq_i\n"
         "(a, b) -> ( a*cos - b*sin ,  a*sin + b*cos )", "Model.rotary"),
        ("Attention", "scores = q . k / sqrt(head size)      (future positions masked out)\n"
         "probs = softmax(scores)      out = probs . V      x <- x + out", "Model.forward"),
        ("SiLU gate (feed-forward)", "y = g / (1 + e^-g) * u        x <- x + down(y)", "silu_mul"),
        ("Sampling", "p = softmax(logits / temperature)\nkeep the top-k;  keep the smallest set whose p adds up to top-p;  draw one", "Sampler"),
        ("Dequantising weights", "Q8_0:   x = scale * q\nQ4_K:   x = d * sc * q  -  dmin * mn\nQ6_K:   x = d * sc * (q - 32)", "dq_q8_0, dq_q4_k, dq_q6_k"),
    )),
)

PAGES = (("overview", "Overview"), ("model", "Model"), ("formulas", "Engine formulas"), ("thoughts", "Thoughts"), ("beliefs", "Beliefs"), ("memories", "Memories"),
         ("questions", "Questions"), ("observations", "Observations"), ("events", "Events"), ("raw", "Raw data"))
PAGE_HELP = {
    "thoughts": "Every change the state engine made, newest first: what it learned, believed, wondered or felt.",
    "beliefs": "What it believes is true, and how sure it is (0 to 1). Faded rows are stale.",
    "memories": "What it remembers. 'sure' is its confidence now, after fading with time.",
    "questions": "What it is still wondering about (yellow) and what it has worked out.",
    "observations": "The last things it perceived, and how many new things each one taught it.",
    "events": "The raw event log, newest first.",
}
TABLES = {            # page -> ((column, width, stretches?), ...)
    "thoughts": (("when", 100, 0), ("kind", 80, 0), ("change", 260, 1)),
    "beliefs": (("sure", 56, 0), ("belief", 240, 1), ("from", 110, 0)),
    "memories": (("when", 100, 0), ("sure", 56, 0), ("from", 110, 0), ("memory", 240, 1)),
    "questions": (("status", 70, 0), ("question", 260, 1), ("raised", 100, 0)),
    "observations": (("when", 100, 0), ("kind", 70, 0), ("learned", 66, 0), ("perceived", 240, 1)),
    "events": (("event", 400, 1),),
}


class Viewer:
    def __init__(self, root, path: str):
        self.root = root
        self.sf = StateFile(path)
        self.rows = {p: [] for p, _ in PAGES}          # page -> [(values, tag, full text)]
        self.page = "overview"
        self.nav = {}
        self._style()
        self._build()
        self.open_path(path, first=True)
        self.show("overview")
        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ look
    def _style(self):
        r = self.root
        r.title(APP)
        r.configure(bg=SV_BG)
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = min(1180, int(sw * 0.85)), min(780, int(sh * 0.85))
        r.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
        r.minsize(900, 580)
        st = ttk.Style(r)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=SV_PANEL, foreground=SV_FG, fieldbackground=SV_CARD, bordercolor=SV_LINE,
                     lightcolor=SV_PANEL, darkcolor=SV_PANEL, troughcolor=SV_PANEL, focuscolor=SV_PANEL)
        st.configure("SV.Treeview", background=SV_PANEL, fieldbackground=SV_PANEL, foreground=SV_FG, rowheight=28,
                     font=("Segoe UI", 10), borderwidth=0, relief="flat")
        st.map("SV.Treeview", background=[("selected", SV_SEL)], foreground=[("selected", "#ffffff")])
        st.configure("SV.Treeview.Heading", background=SV_SIDE, foreground=SV_DIM, font=("Segoe UI", 9, "bold"),
                     relief="flat", padding=(8, 6))
        st.map("SV.Treeview.Heading", background=[("active", SV_SIDE)])
        st.configure("Vertical.TScrollbar", background="#323a47", troughcolor=SV_PANEL, bordercolor=SV_PANEL,
                     arrowcolor=SV_DIM, relief="flat", gripcount=0)
        st.map("Vertical.TScrollbar", background=[("active", "#46505f"), ("pressed", "#46505f")])
        st.configure("SV.TCheckbutton", background=SV_BG, foreground=SV_DIM, font=("Segoe UI", 10))
        st.configure("SV.TCheckbutton", indicatorbackground=SV_CARD, indicatorforeground=SV_BG, indicatormargin=(0, 0, 6, 0),
                     indicatorrelief="flat", bordercolor=SV_LINE, lightcolor=SV_CARD, darkcolor=SV_CARD)
        st.map("SV.TCheckbutton", background=[("active", SV_BG)], foreground=[("active", SV_FG)],
               indicatorbackground=[("selected", SV_ACCENT), ("!selected", SV_CARD)],
               indicatorforeground=[("selected", "#08111f"), ("!selected", SV_CARD)])
        for name, bg, hov, fg in (("Accent", SV_ACCENT, "#7bb0ff", "#08111f"), ("Ghost", SV_CARD, "#252d39", SV_FG)):
            st.configure(f"{name}.TButton", background=bg, foreground=fg, borderwidth=0, padding=(14, 7),
                         font=("Segoe UI", 10, "bold"), relief="flat")
            st.map(f"{name}.TButton", background=[("active", hov), ("disabled", SV_SIDE)],
                   foreground=[("disabled", "#566070")])

    def _label(self, parent, text="", size=10, bold=False, fg=SV_FG, bg=None, **kw):
        return tk.Label(parent, text=text, bg=bg or parent.cget("bg"), fg=fg, anchor="w", justify="left",
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _entry(self, parent, var, width=10):
        return tk.Entry(parent, textvariable=var, bg=SV_CARD, fg=SV_FG, insertbackground=SV_FG, relief="flat",
                        font=("Segoe UI", 11), width=width, highlightthickness=1, highlightbackground=SV_LINE,
                        highlightcolor=SV_ACCENT)

    # ------------------------------------------------------------------ layout
    def _build(self):
        r = self.root
        r.grid_columnconfigure(0, weight=1)
        r.grid_rowconfigure(1, weight=1)
        self._build_header()
        body = tk.Frame(r, bg=SV_BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._build_sidebar(body)
        self.stage = tk.Frame(body, bg=SV_PANEL)
        self.stage.grid(row=0, column=1, sticky="nsew")
        self.stage.grid_columnconfigure(0, weight=1)
        self.stage.grid_rowconfigure(0, weight=1)
        self._build_overview()
        self._build_list()
        self._build_raw()
        self._build_model()
        self._build_formulas()
        self._build_feed()
        self.status_v = tk.StringVar()
        tk.Label(r, textvariable=self.status_v, bg=SV_BG, fg=SV_DIM, anchor="w", font=("Segoe UI", 9),
                 padx=14, pady=5).grid(row=3, column=0, sticky="ew")
        r.bind("<Control-f>", lambda e: (self.show("thoughts") if self.page in ("overview", "raw") else None,
                                         self.find_e.focus_set()))
        r.bind("<F5>", lambda e: self.open_path(self.sf.path))

    def _build_header(self):
        h = tk.Frame(self.root, bg=SV_BG, padx=16, pady=12)
        h.grid(row=0, column=0, sticky="ew")
        h.grid_columnconfigure(0, weight=1)
        left = tk.Frame(h, bg=SV_BG)
        left.grid(row=0, column=0, sticky="w")
        self.head_v, self.sub_v = tk.StringVar(), tk.StringVar()
        tk.Label(left, textvariable=self.head_v, bg=SV_BG, fg=SV_FG, font=("Segoe UI", 17, "bold"), anchor="w").pack(anchor="w")
        tk.Label(left, textvariable=self.sub_v, bg=SV_BG, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").pack(anchor="w")
        right = tk.Frame(h, bg=SV_BG)
        right.grid(row=0, column=1, sticky="e")
        self.mode_l = tk.Label(right, text="", font=("Segoe UI", 9, "bold"), padx=10, pady=3)
        self.mode_l.pack(side="left", padx=(0, 12))
        self.live_v = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Live", variable=self.live_v, style="SV.TCheckbutton").pack(side="left", padx=(0, 12))
        ttk.Button(right, text="Reload", style="Ghost.TButton", command=lambda: self.open_path(self.sf.path)).pack(side="left", padx=3)
        ttk.Button(right, text="Open file...", style="Ghost.TButton", command=self.choose_file).pack(side="left", padx=3)
        ttk.Button(right, text="Destruction lab", style="Ghost.TButton", command=self.open_lab).pack(side="left", padx=3)
        tk.Frame(self.root, bg=SV_LINE, height=1).grid(row=0, column=0, sticky="sew")

    def open_lab(self):
        """The lab is its own window and its own process, so a GPU problem there cannot take the viewer down."""
        import subprocess
        try:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--lab"])
        except OSError as e:
            messagebox.showerror("Destruction lab", f"Could not start the lab:\n{e}")

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=SV_SIDE, width=196)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        tk.Frame(side, bg=SV_SIDE, height=10).pack()
        for key, title in PAGES:
            row = tk.Frame(side, bg=SV_SIDE, cursor="hand2")
            row.pack(fill="x")
            bar = tk.Frame(row, bg=SV_SIDE, width=3)
            bar.pack(side="left", fill="y")
            name = tk.Label(row, text=title, bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 11), anchor="w", padx=14, pady=9)
            name.pack(side="left", fill="x", expand=True)
            count = tk.Label(row, text="", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 9), padx=12)
            count.pack(side="right")
            for w in (row, name, count):
                w.bind("<Button-1>", lambda e, k=key: self.show(k))
                w.bind("<Enter>", lambda e, k=key: self._hover(k, True))
                w.bind("<Leave>", lambda e, k=key: self._hover(k, False))
            self.nav[key] = (row, bar, name, count)
        tk.Frame(side, bg=SV_LINE, width=1).pack(side="right", fill="y")

    def _paint_nav(self, key):
        row, bar, name, count = self.nav[key]
        sel = key == self.page
        bg = SV_SEL if sel else SV_SIDE
        for w in (row, name, count):
            w.configure(bg=bg)
        bar.configure(bg=SV_ACCENT if sel else bg)
        name.configure(fg=SV_FG if sel else SV_DIM)
        count.configure(fg=SV_ACCENT if sel else SV_DIM)

    def _hover(self, key, on):
        if key != self.page:
            row, bar, name, count = self.nav[key]
            bg = "#171d26" if on else SV_SIDE
            for w in (row, bar, name, count):
                w.configure(bg=bg)

    # ------------------------------------------------------------------ pages
    def _build_overview(self):
        f = tk.Frame(self.stage, bg=SV_PANEL, padx=18, pady=16)
        f.grid(row=0, column=0, sticky="nsew")
        for c in (0, 1):
            f.grid_columnconfigure(c, weight=1, uniform="cards")
        for r_ in (0, 1, 2):
            f.grid_rowconfigure(r_, weight=1, uniform="rows")
        self.cards = {}
        for key, title, r_, c_, span in (("now", "RIGHT NOW", 0, 0, 1), ("feels", "FEELS", 0, 1, 1),
                                         ("wants", "WANTS", 1, 0, 1), ("wonder", "WONDERING", 1, 1, 1),
                                         ("beliefs", "STRONGEST BELIEFS", 2, 0, 1), ("thoughts", "LATEST THOUGHTS", 2, 1, 1)):
            card = tk.Frame(f, bg=SV_CARD, padx=14, pady=10, highlightthickness=1, highlightbackground=SV_LINE)
            card.grid(row=r_, column=c_, sticky="nsew", padx=6, pady=6)
            card.grid_propagate(False)
            tk.Label(card, text=title, bg=SV_CARD, fg=SV_DIM, font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")
            body = tk.Frame(card, bg=SV_CARD)
            body.pack(fill="both", expand=True, pady=(6, 0))
            body.bind("<Configure>", lambda e, b=body: self._wrap(b))
            self.cards[key] = body
        self.f_overview = f

    def _wrap(self, body):
        for w in body.winfo_children():
            if isinstance(w, tk.Label):
                w.configure(wraplength=max(120, body.winfo_width() - 4))

    def _fill_card(self, key, items, empty="(nothing)"):
        body = self.cards[key]
        for w in body.winfo_children():
            w.destroy()
        for text, style in (items or [(empty, "dim")]):
            fg = {"dim": SV_DIM, "warn": SV_WARN, "accent": SV_ACCENT, "good": SV_GOOD}.get(style, SV_FG)
            tk.Label(body, text=text, bg=SV_CARD, fg=fg, anchor="w", justify="left", font=("Segoe UI", 10),
                     wraplength=max(120, body.winfo_width() - 4)).pack(anchor="w", pady=1)

    def _build_list(self):
        f = tk.Frame(self.stage, bg=SV_PANEL)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        top = tk.Frame(f, bg=SV_PANEL, padx=16, pady=12)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        tbox = tk.Frame(top, bg=SV_PANEL)
        tbox.grid(row=0, column=0, sticky="w")
        self.page_title = tk.Label(tbox, text="", bg=SV_PANEL, fg=SV_FG, font=("Segoe UI", 14, "bold"), anchor="w")
        self.page_title.pack(anchor="w")
        self.page_help = tk.Label(tbox, text="", bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 9), anchor="w")
        self.page_help.pack(anchor="w")
        self.find_v = tk.StringVar()
        self.find_e = self._entry(top, self.find_v, 22)
        self.find_e.grid(row=0, column=1, sticky="e", ipady=4)
        self.find_v.trace_add("write", lambda *_: self._fill_tree())
        tk.Label(top, text="Search", bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 9)).grid(row=0, column=2, sticky="e", padx=(6, 0))
        split = self.split = tk.PanedWindow(f, orient="horizontal", bg=SV_LINE, sashwidth=1, bd=0, sashrelief="flat")
        split.grid(row=1, column=0, sticky="nsew")
        tf = tk.Frame(split, bg=SV_PANEL)
        self.tv = ttk.Treeview(tf, show="headings", selectmode="browse", style="SV.Treeview")
        sb = ttk.Scrollbar(tf, command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for fld, col in FIELD_COLOURS.items():
            self.tv.tag_configure(fld, foreground=col)
        self.tv.tag_configure("faded", foreground="#5b6470")
        self.tv.tag_configure("open", foreground=SV_WARN)
        self.tv.tag_configure("done", foreground=SV_DIM)
        self.tv.bind("<<TreeviewSelect>>", lambda e: self._show_selected())
        split.add(tf, stretch="always", minsize=360)
        df = tk.Frame(split, bg=SV_SIDE, padx=14, pady=12)
        tk.Label(df, text="DETAILS", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")
        self.detail = tk.Text(df, wrap="word", bg=SV_SIDE, fg=SV_FG, relief="flat", font=("Segoe UI", 10), padx=0, pady=8,
                              height=8, width=36, state="disabled", highlightthickness=0, cursor="arrow")
        self.detail.pack(fill="both", expand=True)
        split.add(df, minsize=240, width=330)
        self.f_list = f

    def _build_raw(self):
        f = tk.Frame(self.stage, bg=SV_PANEL, padx=16, pady=12)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        tk.Label(f, text="Raw data", bg=SV_PANEL, fg=SV_FG, font=("Segoe UI", 14, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        self.raw_txt = tk.Text(f, wrap="none", bg=SV_SIDE, fg=SV_FG, relief="flat", font=("Consolas", 10), padx=10, pady=8,
                               state="disabled", highlightthickness=0)
        sb = ttk.Scrollbar(f, command=self.raw_txt.yview)
        self.raw_txt.configure(yscrollcommand=sb.set)
        self.raw_txt.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        sb.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.f_raw = f

    def _build_formulas(self):
        f = tk.Frame(self.stage, bg=SV_PANEL, padx=16, pady=12)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)
        n = sum(len(items) for _, items in FORMULAS)
        tk.Label(f, text="Engine formulas", bg=SV_PANEL, fg=SV_FG, font=("Segoe UI", 14, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(f, text=f"Every formula this script uses ({n}), with the function that holds it.  Units: pixels, seconds.",
                 bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=1, column=0, sticky="w", pady=(2, 0))
        t = tk.Text(f, wrap="word", bg=SV_SIDE, fg=SV_FG, relief="flat", font=("Consolas", 10), padx=14, pady=10,
                    highlightthickness=0, cursor="arrow", spacing1=1)
        sb = ttk.Scrollbar(f, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        sb.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        t.tag_configure("sec", foreground=SV_ACCENT, font=("Segoe UI", 12, "bold"), spacing1=14, spacing3=4)
        t.tag_configure("name", foreground=SV_FG, font=("Segoe UI", 10, "bold"), spacing1=8)
        t.tag_configure("eq", foreground=SV_GOOD, lmargin1=18, lmargin2=18)
        t.tag_configure("where", foreground=SV_DIM, font=("Segoe UI", 9), lmargin1=18, lmargin2=18)
        for sec, items in FORMULAS:
            t.insert("end", sec + "\n", "sec")
            for name, eq, where in items:
                t.insert("end", name + "\n", "name")
                t.insert("end", eq + "\n", "eq")
                t.insert("end", "in: " + where + "\n", "where")
        t.configure(state="disabled")
        self.f_formulas = f

    # ------------------------------------------------------------------ the Model page (a GGUF language model)
    def _build_model(self):
        f = tk.Frame(self.stage, bg=SV_PANEL, padx=16, pady=12)
        f.grid(row=0, column=0, sticky="nsew")
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)
        self.f_model = f
        self.llm = None                       # the loaded LLMSession (owned by the worker thread)
        self.llm_busy = False
        self.llm_cancel = False
        self.llm_jobs = queue.Queue()
        self.llm_out = queue.Queue()
        self.llm_thread = None
        # row 0: title + load
        top = tk.Frame(f, bg=SV_PANEL)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        tbox = tk.Frame(top, bg=SV_PANEL)
        tbox.grid(row=0, column=0, sticky="w")
        tk.Label(tbox, text="Model", bg=SV_PANEL, fg=SV_FG, font=("Segoe UI", 14, "bold"), anchor="w").pack(anchor="w")
        self.llm_status = tk.StringVar(value="No model loaded.  Load a .gguf file to chat with it.")
        tk.Label(tbox, textvariable=self.llm_status, bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 9), anchor="w",
                 justify="left", wraplength=640).pack(anchor="w")
        self.b_load = ttk.Button(top, text="Load model...", style="Accent.TButton", command=self._llm_choose)
        self.b_load.grid(row=0, column=1, sticky="e")
        # row 1: settings
        st = tk.Frame(f, bg=SV_PANEL)
        st.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        self.t_temp, self.t_tokens, self.t_ctx = tk.StringVar(value="0.7"), tk.StringVar(value="256"), tk.StringVar(value="2048")
        self.t_system = tk.StringVar()
        has_cupy = False
        try:
            import importlib.util
            has_cupy = importlib.util.find_spec("cupy") is not None
        except Exception:
            pass
        self.t_gpu, self.t_dense = tk.BooleanVar(value=has_cupy), tk.BooleanVar(value=False)
        col = 0
        for lab, var, w in (("temperature", self.t_temp, 5), ("max reply tokens", self.t_tokens, 6),
                            ("context (at load)", self.t_ctx, 6)):
            box = tk.Frame(st, bg=SV_PANEL)
            box.grid(row=0, column=col, padx=(0, 12), sticky="w")
            tk.Label(box, text=lab, bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 9)).pack(anchor="w")
            self._entry(box, var, w).pack(ipady=3)
            col += 1
        box = tk.Frame(st, bg=SV_PANEL)
        box.grid(row=0, column=col, padx=(0, 12), sticky="ew")
        st.grid_columnconfigure(col, weight=1)
        tk.Label(box, text="system prompt (optional)", bg=SV_PANEL, fg=SV_DIM, font=("Segoe UI", 9)).pack(anchor="w")
        self._entry(box, self.t_system, 20).pack(fill="x", ipady=3)
        col += 1
        cb = tk.Frame(st, bg=SV_PANEL)
        cb.grid(row=0, column=col, sticky="w")
        g = ttk.Checkbutton(cb, text="GPU (CuPy)", variable=self.t_gpu, style="SV.TCheckbutton")
        g.pack(anchor="w")
        if not has_cupy:
            g.configure(state="disabled")
        ttk.Checkbutton(cb, text="Dense (faster, more memory)", variable=self.t_dense, style="SV.TCheckbutton").pack(anchor="w")
        # row 2: conversation
        cf = tk.Frame(f, bg=SV_SIDE)
        cf.grid(row=2, column=0, sticky="nsew")
        cf.grid_columnconfigure(0, weight=1)
        cf.grid_rowconfigure(0, weight=1)
        self.chat_txt = tk.Text(cf, wrap="word", bg=SV_SIDE, fg=SV_FG, relief="flat", font=("Segoe UI", 11), padx=14, pady=10,
                                state="disabled", highlightthickness=0, spacing3=6)
        sb = ttk.Scrollbar(cf, command=self.chat_txt.yview)
        self.chat_txt.configure(yscrollcommand=sb.set)
        self.chat_txt.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.chat_txt.tag_configure("you", foreground=SV_ACCENT, font=("Segoe UI", 11, "bold"))
        self.chat_txt.tag_configure("ai", foreground=SV_GOOD, font=("Segoe UI", 11, "bold"))
        self.chat_txt.tag_configure("dim", foreground=SV_DIM, font=("Segoe UI", 9))
        self.chat_txt.tag_configure("err", foreground="#ff7b72")
        # row 3: input
        bot = tk.Frame(f, bg=SV_PANEL)
        bot.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bot.grid_columnconfigure(0, weight=1)
        self.chat_v = tk.StringVar()
        self.chat_e = self._entry(bot, self.chat_v, 20)
        self.chat_e.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 8))
        self.chat_e.bind("<Return>", lambda e: self._llm_send())
        self.b_send = ttk.Button(bot, text="Send", style="Accent.TButton", command=self._llm_send)
        self.b_send.grid(row=0, column=1, padx=3)
        self.b_stop = ttk.Button(bot, text="Stop", style="Ghost.TButton", command=self._llm_stop)
        self.b_stop.grid(row=0, column=2, padx=3)
        self.b_clear = ttk.Button(bot, text="Clear chat", style="Ghost.TButton", command=self._llm_clear)
        self.b_clear.grid(row=0, column=3, padx=3)
        self._chat_write("Load a .gguf model with the button above, then type a message below.\n", "dim")
        self._llm_buttons()

    def _chat_write(self, text, tag=None):
        t = self.chat_txt
        t.configure(state="normal")
        t.insert("end", text, tag)
        t.configure(state="disabled")
        t.see("end")

    def _llm_buttons(self):
        ready = self.llm is not None and not self.llm_busy
        self.b_send.configure(state="normal" if ready else "disabled")
        self.b_stop.configure(state="normal" if (self.llm_busy and self.llm is not None) else "disabled")
        self.b_clear.configure(state="normal" if self.llm is not None else "disabled")
        self.b_load.configure(state="disabled" if self.llm_busy else "normal")

    def _llm_worker(self):
        """All model work happens on this one thread, so the GPU context is always used from the same place."""
        while True:
            kind, args = self.llm_jobs.get()
            try:
                if kind == "load":
                    path, use_gpu, dense, ctx = args
                    self.llm = None
                    sess = LLMSession(path, use_gpu=use_gpu, dense=dense, max_ctx=ctx)
                    self.llm_out.put(("loaded", sess))
                elif kind == "chat":
                    text, temp, tokens, system = args
                    stats = self.llm.chat(text, lambda piece: self.llm_out.put(("tok", piece)),
                                          cancel=lambda: self.llm_cancel, temp=temp, max_tokens=tokens, system=system)
                    self.llm_out.put(("done", stats))
                elif kind == "clear":
                    self.llm.clear()
                    self.llm_out.put(("cleared", None))
            except Exception as e:                        # report it in the window; never kill the worker
                self.llm_out.put(("error", str(e) or e.__class__.__name__))

    def _llm_poll(self):
        try:
            while True:
                kind, val = self.llm_out.get_nowait()
                if kind == "tok":
                    self._chat_write(val)
                elif kind == "loaded":
                    self.llm, self.llm_busy = val, False
                    self.llm_status.set(f"{val.name}   -   {val.summary}")
                    self._chat_write(f"Loaded {val.name}.\n\n", "dim")
                elif kind == "done":
                    self.llm_busy = False
                    self._chat_write(f"\n[{val}]\n\n", "dim")
                elif kind == "cleared":
                    self.llm_busy = False
                    self._chat_write("(conversation cleared)\n\n", "dim")
                elif kind == "error":
                    self.llm_busy = False
                    if self.llm is None:
                        self.llm_status.set("Error: " + val)
                    self._chat_write(f"\n{val}\n\n", "err")
                self._llm_buttons()
        except queue.Empty:
            pass
        if self.llm_busy:
            self.root.after(40, self._llm_poll)

    def _llm_start_thread(self):
        if self.llm_thread is None:
            self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
            self.llm_thread.start()

    def _llm_choose(self):
        p = filedialog.askopenfilename(title="Open a GGUF model", initialdir=os.path.dirname(self.sf.path),
                                       filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if p:
            self.llm_load(p)

    def llm_load(self, path):
        try:
            ctx = max(128, int(self.t_ctx.get()))
        except ValueError:
            ctx = 2048
        self._llm_start_thread()
        self.llm_busy = True
        self.llm_status.set(f"Loading {os.path.basename(path)} ...  (large models can take a while)")
        self._chat_write(f"Loading {os.path.basename(path)} ...\n", "dim")
        self.llm_jobs.put(("load", (path, bool(self.t_gpu.get()), bool(self.t_dense.get()), ctx)))
        self._llm_buttons()
        self._llm_poll()

    def _llm_send(self):
        text = self.chat_v.get().strip()
        if not text or self.llm is None or self.llm_busy:
            return
        try:
            temp = max(0.0, float(self.t_temp.get()))
        except ValueError:
            temp = 0.7
        try:
            tokens = max(8, int(self.t_tokens.get()))
        except ValueError:
            tokens = 256
        self.chat_v.set("")
        self._chat_write("You: ", "you")
        self._chat_write(text + "\n")
        self._chat_write("AI: ", "ai")
        self.llm_cancel = False
        self.llm_busy = True
        self.llm_jobs.put(("chat", (text, temp, tokens, self.t_system.get().strip())))
        self._llm_buttons()
        self._llm_poll()

    def _llm_stop(self):
        self.llm_cancel = True

    def _llm_clear(self):
        if self.llm is None or self.llm_busy:
            return
        self.llm_busy = True
        self.llm_jobs.put(("clear", None))
        self._llm_buttons()
        self._llm_poll()


    def _build_feed(self):
        bar = tk.Frame(self.root, bg=SV_SIDE, padx=16, pady=10)
        bar.grid(row=2, column=0, sticky="ew")
        self.feed_frame = bar
        self.subj_v, self.text_v, self.q_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
        bar.grid_columnconfigure(1, weight=4, minsize=200)
        bar.grid_columnconfigure(2, weight=3, minsize=170)
        bar.grid_columnconfigure(3, weight=1, minsize=120)
        tk.Label(bar, text="FEED IT", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 12))
        for col, (lab, var, w) in enumerate((("what happened", self.text_v, 8), ("question it raises (optional)", self.q_v, 8),
                                             ("about (optional)", self.subj_v, 8)), start=1):
            tk.Label(bar, text=lab, bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 9)).grid(row=0, column=col, sticky="w", padx=4)
            e = self._entry(bar, var, w)
            e.grid(row=1, column=col, sticky="ew", padx=4, ipady=4)
            e.bind("<Return>", lambda _e: self.feed())
            if col == 1:
                self.e_text = e
        self.believe_v = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="believe it", variable=self.believe_v, style="SV.TCheckbutton").grid(row=1, column=4, padx=8)
        self.b_feed = ttk.Button(bar, text="Feed", style="Accent.TButton", command=self.feed)
        self.b_feed.grid(row=1, column=5, padx=3)
        self.b_hour = ttk.Button(bar, text="+1 hour", style="Ghost.TButton", command=lambda: self.advance(60))
        self.b_hour.grid(row=1, column=6, padx=3)
        self.b_day = ttk.Button(bar, text="+1 day", style="Ghost.TButton", command=lambda: self.advance(24 * 60))
        self.b_day.grid(row=1, column=7, padx=3)

    # ------------------------------------------------------------------ navigation
    def show(self, page):
        self.page = page
        for key, _ in PAGES:
            self._paint_nav(key)
        frame = {"overview": self.f_overview, "raw": self.f_raw, "model": self.f_model,
                                   "formulas": self.f_formulas}.get(page, self.f_list)
        frame.tkraise()
        if page in ("model", "formulas") or not self.sf.writable:
            self.feed_frame.grid_remove()
        else:
            self.feed_frame.grid()
        if page == "model":
            self.chat_e.focus_set()
        if page in TABLES:
            cols = TABLES[page]
            self.tv.configure(columns=[c for c, _, _ in cols], displaycolumns=[c for c, _, _ in cols])
            for c, w, stretch in cols:
                self.tv.heading(c, text=c.upper(), anchor="w")
                self.tv.column(c, width=w, minwidth=40, anchor="w", stretch=bool(stretch))
            self.page_title.configure(text=dict(PAGES)[page])
            self.page_help.configure(text=PAGE_HELP.get(page, ""))
            self._fill_tree()
            self._set_text(self.detail, "Select a row to read it in full.")
            self.root.after_idle(self._place_sash)

    def _place_sash(self):
        try:
            w = self.split.winfo_width()
            if w > 700:
                self.split.sash_place(0, w - 320, 0)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ files
    def choose_file(self):
        p = filedialog.askopenfilename(title="Open a state file", initialdir=os.path.dirname(self.sf.path),
                                       filetypes=[("State / save files", "*.json"), ("All files", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, path, first=False):
        sf = StateFile(path)
        try:
            sf.load()
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.status_v.set(f"Could not open {os.path.basename(path)}: {e}")
            if first:
                sf = StateFile(DEFAULT_FILE)
                sf.load()
            else:
                return
        self.sf = sf
        self.refresh()
        self.status_v.set(f"Opened {sf.path}" if os.path.exists(sf.path) else
                          f"New test state - it will be saved to {sf.path} when you feed it")

    def _poll(self):
        try:
            if self.live_v.get() and self.sf.changed_on_disk():
                self.sf.load()
                self.refresh(keep=True)
                self.status_v.set(f"Updated {time.strftime('%H:%M:%S')}: the file changed")
        except (OSError, ValueError):
            pass                                      # caught it half-written: the next poll gets it
        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ actions
    def feed(self):
        text = self.text_v.get().strip()
        if not text:
            self.status_v.set("Type what happened first.")
            return
        try:
            lines = self.sf.feed(self.subj_v.get().strip(), text, self.believe_v.get(), self.q_v.get().strip(), 0.7)
        except PermissionError as e:
            self.status_v.set(str(e))
            return
        self.text_v.set("")
        self.q_v.set("")
        self.refresh(keep=True)
        self.show("thoughts")
        self.status_v.set("It took that in: " + (" | ".join(lines) if lines else "nothing new"))

    def advance(self, minutes):
        try:
            self.sf.advance(minutes)
        except PermissionError as e:
            self.status_v.set(str(e))
            return
        self.refresh(keep=True)
        self.status_v.set(f"Time moved on {minutes // 60} h. Memories fade with time: see 'sure' on the Memories page.")

    # ------------------------------------------------------------------ filling the views
    @staticmethod
    def _set_text(t, text):
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", text)
        t.configure(state="disabled")

    def refresh(self, keep=False):
        sf, cs, ch = self.sf, self.sf.cs, self.sf.char
        name = cs.identity.get("name") or "State"
        open_q = [u for u in cs.uncertainties if u.get("status") == "open"]
        self.head_v.set(name)
        self.sub_v.set(f"{cs.current_activity or 'idle'}   |   sim time {sf.clock.now.replace('T', ' ')}")
        self.root.title(f"{APP} - {name}")
        if sf.writable:
            self.mode_l.configure(text="TEST FILE - you can feed it", bg="#143325", fg=SV_GOOD)
        else:
            self.mode_l.configure(text="VIEW ONLY - belongs to a running app", bg="#3a2d0f", fg=SV_WARN)
        if self.page == "model" or not sf.writable:
            self.feed_frame.grid_remove()
        else:
            self.feed_frame.grid()

        self.rows["thoughts"] = [((_short_time(c.get("time")), FIELD_WORDS.get(c.get("field"), c.get("field")), c.get("change", "")),
                                  c.get("field", ""), f"{c.get('time')}  [{c.get('field')}]\n\n{c.get('change')}")
                                 for c in reversed(cs.state_changes)]
        self.rows["beliefs"] = [((f"{float(b.get('confidence', 0)):.2f}", b.get("text", ""), b.get("source", "")),
                                 "faded" if b.get("stale") else "beliefs", self._belief_detail(b))
                                for b in sorted(cs.beliefs, key=lambda b: -float(b.get("confidence", 0)))]
        mems = []
        for m in reversed(cs.memories):
            eff = ch.effective_confidence(m)
            tag = "faded" if (eff < 0.15 or m.get("superseded_by")) else "memories"
            note = (" (superseded)" if m.get("superseded_by") else "") + (" (incomplete)" if m.get("incomplete") else "")
            mems.append(((_short_time(m.get("time")), f"{eff:.2f}", m.get("source", ""), m.get("text", "") + note), tag,
                         f"{m.get('time')}\nformed at {m.get('confidence')}, now {eff:.2f} after fading\n"
                         f"source: {m.get('source')}   recalled {m.get('recalled', 0)} times\n\n{m.get('text')}"))
        self.rows["memories"] = mems
        qs = sorted(cs.uncertainties, key=lambda u: (u.get("status") != "open", str(u.get("raised"))))
        self.rows["questions"] = [((u.get("status", ""), u.get("question", ""), _short_time(u.get("raised"))),
                                   "open" if u.get("status") == "open" else "done",
                                   f"{u.get('question')}\n\nstatus: {u.get('status')}\nraised {u.get('raised')}"
                                   f"\nresolved {u.get('resolved') or '-'}\n\n{u.get('notes') or ''}") for u in qs]
        self.rows["observations"] = [((_short_time(o.get("time")), o.get("kind", ""),
                                       str(len(o.get("new", [])) + len(o.get("upgraded", []))), " | ".join(o.get("perceived", []))),
                                      "memories" if o.get("new") else "done",
                                      json.dumps(o, indent=1, ensure_ascii=False)) for o in reversed(cs.recent_observations)]
        self.rows["events"] = [((e.describe(),), "", e.describe()) for e in reversed(sf.events.events[-500:])]
        counts = {"thoughts": len(cs.state_changes), "beliefs": len(cs.beliefs), "memories": len(cs.memories),
                  "questions": len(open_q), "observations": len(cs.recent_observations), "events": len(sf.events.events),
                  "formulas": sum(len(items) for _, items in FORMULAS)}
        for key, (_, _, _, cl) in self.nav.items():
            cl.configure(text=str(counts.get(key, "")))
        self._fill_overview()
        if self.page in TABLES:
            self._fill_tree(keep)
        self._set_text(self.raw_txt, json.dumps(cs.to_dict(), indent=1, ensure_ascii=False))

    def _fill_overview(self):
        cs = self.sf.cs
        loc = cs.current_location or {}
        i = cs.identity
        place = (loc.get("room_label") or "somewhere") + (f"  -  {loc['spot_label']}" if loc.get("spot_label") else "")
        self._fill_card("now", [(f"{i.get('name', '?')}" + (f", {i['role']}" if i.get("role") else ""), "main"),
                                (cs.current_activity or "idle", "accent"), (place, "dim")])
        feel = [f"{BAND.get(int(v.get('band', 0)), '')} {n}" for n, v in cs.needs.items() if int(v.get("band", 0)) >= 1]
        self._fill_card("feels", [(f, "warn") for f in feel] or [("Fine", "good")])
        self._fill_card("wants", [("- " + self._cut(g.get("text", "")), "main") for g in cs.goals if g.get("status") == "active"][:5],
                        empty="(nothing)")
        self._fill_card("wonder", [("? " + self._cut(u.get("question", "")), "warn") for u in cs.uncertainties
                                   if u.get("status") == "open"][:5], empty="(nothing open)")
        self._fill_card("beliefs", [(f"{float(b.get('confidence', 0)):.2f}   {self._cut(b.get('text', ''), 48)}", "main")
                                    for b in sorted(cs.beliefs, key=lambda b: -float(b.get("confidence", 0)))[:4]],
                        empty="(no beliefs yet)")
        self._fill_card("thoughts", [(f"{_short_time(c.get('time'))}   {self._cut(c.get('change', ''), 44)}", "dim")
                                     for c in list(reversed(cs.state_changes))[:4]], empty="(none yet)")

    @staticmethod
    def _cut(text, n=64):
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[:n - 1] + "..."

    @staticmethod
    def _belief_detail(b) -> str:
        out = [b.get("text", ""), "", f"confidence {b.get('confidence')}   source {b.get('source')}",
               f"formed {b.get('formed')}   updated {b.get('updated')}"]
        for h in b.get("history", []):
            out.append(f"before {h.get('until')}: {h.get('text')}")
        if b.get("stale"):
            out.append("(stale: time passed without it being checked)")
        return "\n".join(out)

    def _fill_tree(self, keep=False):
        if self.page not in TABLES:
            return
        q = self.find_v.get().strip().lower()
        tv = self.tv
        pos = tv.yview()[0] if keep else 0
        sel = tv.selection()
        sel_vals = tv.item(sel[0], "values") if sel else None
        tv.delete(*tv.get_children())
        n = 0
        for vals, tag, full in self.rows[self.page]:
            if q and q not in " ".join(map(str, vals)).lower() and q not in full.lower():
                continue
            iid = tv.insert("", "end", values=vals, tags=(tag,) if tag else ())
            n += 1
            if keep and sel_vals and tuple(map(str, vals)) == tuple(map(str, sel_vals)):
                tv.selection_set(iid)
        tv.yview_moveto(pos)
        total = len(self.rows[self.page])
        self.page_title.configure(text=f"{dict(PAGES)[self.page]}   " + (f"{n} of {total}" if q else str(total)))

    def _show_selected(self):
        sel = self.tv.selection()
        if not sel:
            return
        vals = tuple(map(str, self.tv.item(sel[0], "values")))
        for v, _tag, full in self.rows[self.page]:
            if tuple(map(str, v)) == vals:
                self._set_text(self.detail, full)
                return



def viewer_main():   # (standalone use: python physics_engine.py [file.json])
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    if len(sys.argv) <= 1 and not os.path.exists(path):
        for name in ("state_save.json",):
            save = os.path.join(SAVE_HOME, name)
            if os.path.exists(save):
                path = save
                break
    root = tk.Tk()
    Viewer(root, path)
    root.mainloop()


# ####################################################################################################
##### PART: 3  GGUF MODEL RUNNER  (loads a .gguf language model and generates text; no llama.cpp)
# ####################################################################################################
# Reads the GGUF header, maps the weights from disk, keeps them quantised (F16, F32, Q8_0, Q4_K, Q6_K), multiplies
# them with fused CUDA kernels on the GPU (CuPy) or with numpy on the CPU, tokenises with the vocabulary in the file
# (SentencePiece or byte-level BPE), and chats with ChatML or [INST] templates.  Layouts: llama, mistral3, qwen2, qwen3.
# Needs:  pip install numpy regex   (GPU mode also needs CuPy matching your CUDA version, e.g. cupy-cuda12x)
# Run:    python physics_engine.py --gguf model.gguf [--prompt "..."] [--tokens N] [--temp T] [--top-p P] [--dense]
#                                  [--stats] [--cpu] [--info]
# Not supported yet: Q5_K, Q1_0, MXFP4, the GPT-OSS / GLM architectures.

# ----------------------------------------------------------------------------- array library (CuPy on the GPU, else numpy)
GPU = False
xp = np
cupy = None


def init_llm_backend(use_gpu=True):
    """Pick the array library for the language-model runner: CuPy (GPU) if it is installed and a GPU is found,
    otherwise numpy (CPU).  Called once before a model is loaded; importing this file never touches CuPy."""
    global GPU, xp, cupy, _silu_mul_gpu
    GPU, xp, cupy, _silu_mul_gpu = False, np, None, None
    if use_gpu:
        try:
            import cupy as _cp
            if _cp.cuda.runtime.getDeviceCount() > 0:
                cupy, xp, GPU = _cp, _cp, True
                _silu_mul_gpu = _cp.ElementwiseKernel("float32 g, float32 u", "float32 y",
                                                      "y = g / (1.0f + expf(-g)) * u", "silu_mul")
        except Exception:
            pass


# ----------------------------------------------------------------------------- GGUF reader
# metadata value type -> struct format (little-endian)
SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
T_STRING, T_ARRAY = 8, 9

# tensor type -> (name, elements per block, bytes per block).  Types not listed show as "type N".
TENSOR_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 30: ("BF16", 1, 2),
}


class GGUFError(Exception):
    pass


def _read(f, fmt):
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("the file ends early - the download may be incomplete")
    return struct.unpack("<" + fmt, data)[0]


def _read_string(f):
    n = _read(f, "Q")
    if n > 10_000_000:
        raise GGUFError(f"a text value claims to be {n} bytes long - the file is probably damaged")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    if vtype == T_STRING:
        return _read_string(f)
    if vtype == T_ARRAY:
        item_type = _read(f, "I")
        count = _read(f, "Q")
        return [_read_value(f, item_type) for _ in range(count)]
    if vtype in SCALAR_FORMATS:
        return _read(f, SCALAR_FORMATS[vtype])
    raise GGUFError(f"unknown settings value type {vtype}")


def read_gguf(f):
    """Parse the header: settings (metadata) and the list of tensors, with where their data starts."""
    if f.read(4) != b"GGUF":
        raise GGUFError("not a GGUF file (the first four bytes should be GGUF)")
    version = _read(f, "I")
    tensor_count = _read(f, "Q")
    kv_count = _read(f, "Q")

    metadata = {}
    for _ in range(kv_count):
        key = _read_string(f)
        vtype = _read(f, "I")
        metadata[key] = _read_value(f, vtype)

    tensors = []
    for _ in range(tensor_count):
        name = _read_string(f)
        n_dims = _read(f, "I")
        dims = [_read(f, "Q") for _ in range(n_dims)]
        ttype = _read(f, "I")
        offset = _read(f, "Q")
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

    # tensor data starts at the next multiple of the alignment after the header
    alignment = metadata.get("general.alignment", 32) or 32
    pos = f.tell()
    data_start = pos + (-pos % alignment)

    file_size = os.fstat(f.fileno()).st_size
    return {"version": version, "metadata": metadata, "tensors": tensors,
            "data_start": data_start, "file_size": file_size}


def tensor_bytes_of(t):
    """How many bytes this tensor's data takes, or None if the type is not in our table."""
    n_elems = 1
    for d in t["dims"]:
        n_elems *= d
    info = TENSOR_TYPES.get(t["type"])
    if info is None:
        return None
    _, block_elems, block_bytes = info
    return (n_elems // block_elems) * block_bytes


def human(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


def show_value(value, show_all):
    if isinstance(value, list):
        if not show_all and len(value) > 8:
            preview = ", ".join(repr(v)[:40] for v in value[:5])
            return f"[{len(value)} items: {preview}, ...]"
        return "[" + ", ".join(repr(v)[:200] for v in value) + "]"
    if isinstance(value, str):
        text = value if show_all or len(value) <= 300 else value[:300] + "..."
        return repr(text)
    return str(value)


def report(path, model, show_all):
    md, tensors = model["metadata"], model["tensors"]
    print("=" * 78)
    print(f"File:     {path}")
    print(f"Size:     {human(model['file_size'])}")
    print(f"GGUF version {model['version']}  |  {len(md)} settings  |  {len(tensors)} tensors")
    print("=" * 78)

    print("\nSETTINGS (metadata)\n")
    width = max((len(k) for k in md), default=10)
    for key in sorted(md):
        print(f"  {key:<{width}}  {show_value(md[key], show_all)}")

    print("\nTENSORS\n")
    print(f"  {'name':<46} {'shape':<22} {'type':<8} {'size':>10}")
    print("  " + "-" * 90)
    total = 0
    problems = []
    type_counts = {}
    for t in tensors:
        shape = " x ".join(str(d) for d in reversed(t["dims"]))   # shown the usual way round: rows x columns
        info = TENSOR_TYPES.get(t["type"])
        tname = info[0] if info else f"type {t['type']}"
        type_counts[tname] = type_counts.get(tname, 0) + 1
        nbytes = tensor_bytes_of(t)
        if nbytes is not None:
            total += nbytes
            if model["data_start"] + t["offset"] + nbytes > model["file_size"]:
                problems.append(t["name"])
        print(f"  {t['name']:<46} {shape:<22} {tname:<8} {human(nbytes):>10}")

    print("\nSUMMARY")
    print(f"  Total tensor data:  {human(total)}  (tensors of unknown types are not counted)")
    print("  Tensor types:       " + ", ".join(f"{k} x{v}" for k, v in sorted(type_counts.items())))
    arch = md.get("general.architecture", "unknown")
    print(f"  Architecture:       {arch}")
    if problems:
        print(f"\n  WARNING: {len(problems)} tensor(s) run past the end of the file, so the download is probably incomplete:")
        for name in problems[:10]:
            print(f"    - {name}")
    elif model["data_start"] + total > model["file_size"] + 1:
        print("\n  WARNING: the file is smaller than its tensors need, so the download is probably incomplete.")
    else:
        print("\n  File size matches what the tensors need.")



# ggml tensor type ids -> our names
F32, F16, Q8_0, Q4_K, Q6_K = 0, 1, 8, 12, 14
KIND = {F32: "f32", F16: "f16", Q8_0: "q8_0", Q4_K: "q4_k", Q6_K: "q6_k"}
PLAIN_DTYPE = {"f32": np.float32, "f16": np.float16}
BLOCKS = {"q8_0": (32, 34), "q4_k": (256, 144), "q6_k": (256, 210)}   # kind -> (elements per block, bytes per block)
ARCHITECTURES = ("llama", "mistral3", "qwen2", "qwen3")
SPACE = "▁"                                # SentencePiece writes a space as this character
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
TEKKEN_PATTERN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
                  r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
                  r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_PATTERNS = {"qwen2": QWEN_PATTERN, "": QWEN_PATTERN, "tekken": TEKKEN_PATTERN}
CHUNK_ROWS = 2048                          # matrices are decoded this many rows at a time, to limit memory use
PREFILL_CHUNK = 256                        # prompt tokens run through the network together
RESERVE_BYTES = 1536 * 1024 * 1024         # VRAM kept free for the CUDA context and working memory
ROWS_PER_BLOCK = 4                         # matrix rows per CUDA thread block (one warp of 32 threads per row)
REPLY_CAP = 512                            # longest chat reply in tokens


def tensor_bytes(ttype, n):
    kind = KIND[ttype]
    if kind in PLAIN_DTYPE:
        return n * np.dtype(PLAIN_DTYPE[kind]).itemsize
    elems, bb = BLOCKS[kind]
    return (n // elems) * bb


def to_host(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


def sync():
    if GPU:
        cupy.cuda.Stream.null.synchronize()


# ----------------------------------------------------------------------------- byte <-> character table (GPT-2 style)
def _byte_maps():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    enc = {b: chr(c) for b, c in zip(bs, cs)}
    return enc, {v: k for k, v in enc.items()}


BYTE_ENC, BYTE_DEC = _byte_maps()


# ----------------------------------------------------------------------------- tokenisers
class SentencePieceTokenizer:
    """Used by Llama and Mistral files (tokenizer.ggml.model = 'llama')."""

    def __init__(self, md):
        self.tokens = md["tokenizer.ggml.tokens"]
        self.scores = md.get("tokenizer.ggml.scores") or [0.0] * len(self.tokens)
        self.types = md.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.bos = md.get("tokenizer.ggml.bos_token_id", 1)
        self.add_bos = md.get("tokenizer.ggml.add_bos_token", True)
        self.eos = md.get("tokenizer.ggml.eos_token_id", 2)
        self.byte_ids = {}
        for i, t in enumerate(self.tokens):
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                self.byte_ids[int(t[3:5], 16)] = i

    def encode(self, text, bos=True):
        text = (" " + text).replace(" ", SPACE)
        syms = []
        for ch in text:
            if ch in self.index:
                syms.append(self.index[ch])
            else:
                syms.extend(self.byte_ids[b] for b in ch.encode("utf-8"))
        # Merge the highest-scoring neighbouring pair until none is left (leftmost wins a tie).
        # A heap of candidate pairs and a linked list keep this fast on long text.
        n = len(syms)
        if n > 1:
            nxt = list(range(1, n)) + [-1]
            prv = list(range(-1, n - 1))
            alive = [True] * n
            heap = []

            def push(i):
                j = nxt[i]
                if j < 0:
                    return
                m = self.index.get(self.tokens[syms[i]] + self.tokens[syms[j]])
                if m is not None:
                    heapq.heappush(heap, (-self.scores[m], i, syms[i], syms[j], m))

            for i in range(n - 1):
                push(i)
            while heap:
                _, i, a, b, m = heapq.heappop(heap)
                j = nxt[i]
                if not alive[i] or j < 0 or syms[i] != a or syms[j] != b:
                    continue                     # this pair has changed since it was queued
                syms[i] = m
                alive[j] = False
                nxt[i] = nxt[j]
                if nxt[j] >= 0:
                    prv[nxt[j]] = i
                if prv[i] >= 0:
                    push(prv[i])
                push(i)
            syms = [s for s, ok in zip(syms, alive) if ok]
        return ([self.bos] if (bos and self.add_bos) else []) + syms

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if self.types[i] == 3:
                continue
            t = self.tokens[i]
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                out.append(int(t[3:5], 16))
            else:
                out += t.replace(SPACE, " ").encode("utf-8")
        return out.decode("utf-8", errors="replace")


class ByteBPETokenizer:
    """Used by Qwen files (tokenizer.ggml.model = 'gpt2', pre = 'qwen2')."""

    def __init__(self, md):
        try:
            import regex
        except ImportError:
            sys.exit("This model needs the 'regex' package. Run:  pip install regex")
        pre = md.get("tokenizer.ggml.pre", "")
        if pre not in PRE_PATTERNS:
            sys.exit(f"Pre-tokeniser '{pre}' is not supported yet.")
        self.pre = pre
        self.pat = regex.compile(PRE_PATTERNS[pre])
        self.tokens = md["tokenizer.ggml.tokens"]
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.types = md.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.ranks = {}
        for r, m in enumerate(md.get("tokenizer.ggml.merges") or []):
            a, b = m.split(" ", 1)
            self.ranks[(a, b)] = r
        self.bos = md.get("tokenizer.ggml.bos_token_id")
        self.eos = md.get("tokenizer.ggml.eos_token_id", 2)
        self.add_bos = md.get("tokenizer.ggml.add_bos_token", False)
        # Control tokens such as <|im_start|> are matched whole, before the text is split into words.
        specials = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4) and len(t) > 1),
                          key=len, reverse=True)
        self.special = re.compile("(" + "|".join(map(re.escape, specials)) + ")") if specials else None
        self._piece = lru_cache(maxsize=1 << 16)(self._encode_piece)   # the same words come up again and again

    def _encode_piece(self, piece):
        syms = [BYTE_ENC[b] for b in piece.encode("utf-8")]
        while len(syms) > 1:
            best_rank, best_i = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        return tuple(self.index[s] for s in syms)

    def encode(self, text, bos=True):
        ids = [self.bos] if (bos and self.add_bos and self.bos is not None) else []
        parts = self.special.split(text) if self.special else [text]
        for n, part in enumerate(parts):
            if not part:
                continue
            if n % 2 == 1:                       # odd entries are the control tokens the split matched
                ids.append(self.index[part])
                continue
            for piece in self.pat.findall(part):
                ids.extend(self._piece(piece))
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if self.types[i] == 3:
                continue
            t = self.tokens[i]
            if self.types[i] == 4:
                out += t.encode("utf-8")
            else:
                out += bytes(BYTE_DEC[c] for c in t if c in BYTE_DEC)
        return out.decode("utf-8", errors="replace")


def make_tokenizer(md):
    kind = md.get("tokenizer.ggml.model")
    if kind == "llama":
        return SentencePieceTokenizer(md)
    if kind == "gpt2":
        return ByteBPETokenizer(md)
    sys.exit(f"Tokeniser '{kind}' is not supported yet.")


# ----------------------------------------------------------------------------- settings
class Config:
    def __init__(self, md):
        arch = md.get("general.architecture", "")
        if arch not in ARCHITECTURES:
            sys.exit(f"Architecture '{arch}' is not supported yet. Supported: {', '.join(ARCHITECTURES)}.")
        self.arch = arch

        def g(key, default=None):
            v = md.get(f"{arch}.{key}", default)
            if v is None:
                sys.exit(f"The file is missing the setting {arch}.{key}")
            return v
        n_head = g("attention.head_count")
        self.n_layer = g("block_count")
        self.n_embd = g("embedding_length")
        self.n_head = n_head
        self.n_kv = g("attention.head_count_kv", n_head)
        self.n_ff = g("feed_forward_length")
        self.eps = g("attention.layer_norm_rms_epsilon", 1e-5)
        self.base = g("rope.freq_base", 10000.0)
        self.ctx = g("context_length", 2048)
        self.hd = g("attention.key_length", self.n_embd // n_head)
        self.rope_dim = g("rope.dimension_count", self.hd)
        self.neox = arch in ("qwen2", "qwen3")
        self.qkv_bias = arch == "qwen2"
        self.qk_norm = arch == "qwen3"
        if f"{arch}.rope.scaling.type" in md:
            print("Note: this file sets rope scaling, which this script ignores. "
                  "Output may drift from the reference for long prompts.", file=sys.stderr)


# ----------------------------------------------------------------------------- decoding quantised blocks
# Each dq_* takes uint8 blocks of shape (n_blocks, bytes_per_block) and an array library m (numpy or CuPy),
# and returns float32 values of shape (n_blocks, elements_per_block).
# On the GPU these are only used for prompt batches and as the reference for the fused kernels below.

def _half(b, m):
    return b.copy().view(m.float16).astype(m.float32)


def dq_q8_0(b, m):
    """Blocks of 32: one float16 scale, then 32 signed bytes. value = scale * byte."""
    return _half(b[:, 0:2], m) * b[:, 2:34].copy().view(m.int8).astype(m.float32)


def dq_q4_k(b, m):
    """Blocks of 256: float16 d, float16 dmin, 12 bytes of packed 6-bit scales and minimums, 128 bytes of 4-bit values."""
    nb = b.shape[0]
    d = _half(b[:, 0:2], m)                       # (nb, 1)
    dmin = _half(b[:, 2:4], m)                    # (nb, 1)
    s = b[:, 4:16]
    sc = m.zeros((nb, 8), m.uint8)
    mn = m.zeros((nb, 8), m.uint8)
    sc[:, 0:4] = s[:, 0:4] & 63
    mn[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    qs = b[:, 16:144].reshape(nb, 4, 32)
    vals = m.stack([qs & 0x0F, qs >> 4], axis=2).reshape(nb, 8, 32).astype(m.float32)
    out = d[:, :, None] * sc[:, :, None].astype(m.float32) * vals \
        - dmin[:, :, None] * mn[:, :, None].astype(m.float32)
    return out.reshape(nb, 256)


def dq_q6_k(b, m):
    """Blocks of 256: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 signed 8-bit scales, float16 d."""
    nb = b.shape[0]
    ql = b[:, 0:128].reshape(nb, 2, 2, 32)
    qh = b[:, 128:192].reshape(nb, 2, 32)
    sc = b[:, 192:208].copy().view(m.int8).astype(m.float32).reshape(nb, 2, 4, 2)
    d = _half(b[:, 208:210], m).reshape(nb, 1, 1, 1)
    lo = ql & 0x0F
    hi = ql >> 4
    parts = [
        lo[:, :, 0, :] | (((qh >> 0) & 3) << 4),
        lo[:, :, 1, :] | (((qh >> 2) & 3) << 4),
        hi[:, :, 0, :] | (((qh >> 4) & 3) << 4),
        hi[:, :, 1, :] | (((qh >> 6) & 3) << 4),
    ]
    q = m.stack(parts, axis=2).astype(m.int16) - 32
    scale = m.repeat(sc, 16, axis=3)
    return (d * scale * q.astype(m.float32)).reshape(nb, 256)


DEQUANT = {"q8_0": dq_q8_0, "q4_k": dq_q4_k, "q6_k": dq_q6_k}


def decode_rows(data, kind, cols, m=None):
    """data: (rows, bytes per row) uint8 for quantised kinds, or (rows, cols) float for f16/f32.
    Returns float32 values of shape (rows, cols) using the array library m (default: the one in use)."""
    m = m or xp
    data = m.asarray(data)
    if kind in PLAIN_DTYPE:
        return data.astype(m.float32, copy=False)
    elems, bb = BLOCKS[kind]
    rows = data.shape[0]
    return DEQUANT[kind](data.reshape(-1, bb), m).reshape(rows, cols)


# ----------------------------------------------------------------------------- fused GPU kernels
# One warp (32 threads) per matrix row. Each thread decodes its share of the packed bytes in registers,
# multiplies by the input vector and the warp adds up the result. The decoded weights never touch VRAM,
# so each byte of the model is read once per token, which is what limits speed.
CUDA_SRC = r"""
__device__ __forceinline__ float half_to_float(unsigned int h) {
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int e = (h >> 10) & 0x1fu;
    unsigned int m = h & 0x3ffu;
    if (e == 0u) {                                   // zero or subnormal
        float f = (float)m * 5.9604644775390625e-8f;   // m * 2^-24
        return sign ? -f : f;
    }
    if (e == 31u) return __int_as_float((int)(sign | 0x7f800000u | (m << 13)));
    return __int_as_float((int)(sign | ((e + 112u) << 23) | (m << 13)));
}

__device__ __forceinline__ float load_half(const unsigned char* p) {
    return half_to_float((unsigned int)p[0] | ((unsigned int)p[1] << 8));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* s, float& sc, float& mn) {
    if (j < 4) {
        sc = (float)(s[j] & 63);
        mn = (float)(s[j + 4] & 63);
    } else {
        sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

extern "C" __global__ void mv_q8_0(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                   float* __restrict__ y, const int rows, const int cols) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int nb = cols >> 5;
    const unsigned char* r = W + (size_t)row * nb * 34;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = r + b * 34;
        sum += load_half(blk) * (float)(signed char)blk[2 + lane] * x[b * 32 + lane];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

extern "C" __global__ void mv_q4_k(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                   float* __restrict__ y, const int rows, const int cols) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int nb = cols >> 8;
    const unsigned char* r = W + (size_t)row * nb * 144;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = r + b * 144;
        const float d = load_half(blk);
        const float dmin = load_half(blk + 2);
        const unsigned char* s = blk + 4;
        const unsigned char* qs = blk + 16;
        const float* xb = x + b * 256;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const unsigned int q = qs[g * 32 + lane];
            float sc0, m0, sc1, m1;
            q4k_scale_min(2 * g, s, sc0, m0);
            q4k_scale_min(2 * g + 1, s, sc1, m1);
            sum += (d * sc0 * (float)(q & 15u) - dmin * m0) * xb[(2 * g) * 32 + lane];
            sum += (d * sc1 * (float)(q >> 4) - dmin * m1) * xb[(2 * g + 1) * 32 + lane];
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

extern "C" __global__ void mv_q6_k(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                   float* __restrict__ y, const int rows, const int cols) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int nb = cols >> 8;
    const unsigned char* r = W + (size_t)row * nb * 210;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = r + b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = (const signed char*)(blk + 192);
        const float d = load_half(blk + 208);
        const float* xb = x + b * 256;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int hb = qh[h * 32 + lane];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                const unsigned int lb = ql[h * 64 + (p & 1) * 32 + lane];
                const unsigned int lo = (p < 2) ? (lb & 15u) : (lb >> 4);
                const int q = (int)(lo | (((hb >> (2 * p)) & 3u) << 4)) - 32;
                sum += d * (float)sc[h * 8 + p * 2 + (lane >> 4)] * (float)q * xb[h * 128 + p * 32 + lane];
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

extern "C" __global__ void mv_f16(const unsigned short* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, const int rows, const int cols) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const unsigned short* r = W + (size_t)row * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += 32) sum += half_to_float(r[c]) * x[c];
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}
"""
FUSED = {}                                 # kind -> compiled kernel, filled by setup_fused()
STAGING = None                             # VRAM buffer that RAM-resident matrices are copied into before use


def fused_matvec(kind, data, x, rows, cols):
    y = cupy.empty(rows, dtype=cupy.float32)
    grid = ((rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)
    FUSED[kind](grid, (32, ROWS_PER_BLOCK), (data, x, y, np.int32(rows), np.int32(cols)))
    return y


def setup_fused(mats):
    """Compile the kernels and check each against the plain decoder on real weights from this model.
    A kernel that doesn't match is switched off, and that weight type falls back to the slower path."""
    try:
        mod = cupy.RawModule(code=CUDA_SRC)
        kernels = {k: mod.get_function("mv_" + k) for k in ("q8_0", "q4_k", "q6_k", "f16")}
    except Exception as e:
        print(f"Note: could not compile the fused GPU kernels, using the slower path. ({e})", file=sys.stderr)
        return
    rng = np.random.default_rng(0)
    for kind, kern in kernels.items():
        sample = next((t for t in mats if t.kind == kind), None)
        if sample is None:
            continue
        rows = min(sample.shape[0], 256)
        cols = sample.shape[1]
        data = device_data(sample)[:rows]
        x = xp.asarray(rng.standard_normal(cols).astype(np.float32))
        FUSED[kind] = kern
        try:
            got = to_host(fused_matvec(kind, data, x, rows, cols))
            ref = to_host(decode_rows(data, kind, cols) @ x)
            err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-6))
        except Exception as e:
            err, why = None, str(e)
        if err is None or not err < 1e-3:
            del FUSED[kind]
            detail = why if err is None else f"error {err:.2e}"
            print(f"Note: the fused {kind} kernel failed its check ({detail}); using the slower path for it.",
                  file=sys.stderr)


# ----------------------------------------------------------------------------- weights
class Tensor:
    """One weight matrix. data is numpy (RAM) or CuPy (VRAM); shape is (rows, cols)."""

    def __init__(self, kind, shape, data):
        self.kind, self.shape, self.data, self.on_gpu = kind, shape, data, False


def load_weights(path, info):
    """Map every tensor from the file without reading it all in. Nothing is decoded yet."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for t in info["tensors"]:
        ttype, name = t["type"], t["name"]
        if ttype not in KIND:
            sys.exit(f"'{name}' is stored as type {ttype}, which this script can't read yet. "
                     "Supported: F32, F16, Q8_0, Q4_K, Q6_K.")
        kind = KIND[ttype]
        shape = tuple(reversed(t["dims"]))           # GGUF lists dimensions fastest-first
        n = 1
        for d in shape:
            n *= d
        size = tensor_bytes(ttype, n)
        start = info["data_start"] + t["offset"]
        if start + size > mm.shape[0]:
            raise GGUFError(f"'{name}' is cut short - the download may be incomplete")
        raw = mm[start:start + size]
        if kind in PLAIN_DTYPE:
            data = raw.view(PLAIN_DTYPE[kind]).reshape(shape)
        else:
            if len(shape) != 2:
                sys.exit(f"'{name}' is quantised but is not a matrix, which this script does not handle.")
            elems, bb = BLOCKS[kind]
            data = raw.reshape(shape[0], (shape[1] // elems) * bb)
        tensors[name] = Tensor(kind, shape, data)
    return tensors


def device_data(t):
    """The tensor's bytes on the device that does the maths. RAM-resident matrices are copied into STAGING."""
    if not GPU or t.on_gpu:
        return t.data
    view = STAGING[:t.data.nbytes].view(t.data.dtype).reshape(t.data.shape)
    view.set(t.data)
    return view


def linear_rows(t, X):
    """X: (T, cols) float32. Returns X times the matrix's transpose: (T, rows)."""
    rows, cols = t.shape
    data = device_data(t)
    T = X.shape[0]
    if T == 1 and t.kind in FUSED:
        return fused_matvec(t.kind, data, xp.ascontiguousarray(X[0]), rows, cols).reshape(1, rows)
    if t.kind == "f32":
        return X @ data.T
    out = xp.empty((T, rows), dtype=xp.float32)
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(rows, r0 + CHUNK_ROWS)
        out[:, r0:r1] = X @ decode_rows(data[r0:r1], t.kind, cols).T
    return out


class Linear:
    """One or more weight matrices that read the same input (Q, K and V, or gate and up).
    Neighbours with the same storage type are joined into one matrix, so they run as one multiply."""

    def __init__(self, tensors, biases=None):
        self.sizes = [t.shape[0] for t in tensors]
        groups = []
        for t in tensors:
            if groups and groups[-1][-1].kind == t.kind:
                groups[-1].append(t)
            else:
                groups.append([t])
        self.parts = []
        for g in groups:
            if len(g) == 1:
                self.parts.append(g[0])
            else:
                data = np.concatenate([t.data for t in g], axis=0)
                self.parts.append(Tensor(g[0].kind, (sum(t.shape[0] for t in g), g[0].shape[1]), data))
        self.bias = None
        if biases and any(b is not None for b in biases):
            self.bias = xp.concatenate([b if b is not None else xp.zeros(n, xp.float32)
                                        for b, n in zip(biases, self.sizes)])

    def __call__(self, X):
        outs = [linear_rows(p, X) for p in self.parts]
        y = outs[0] if len(outs) == 1 else xp.concatenate(outs, axis=1)
        if self.bias is not None:
            y = y + self.bias
        if len(self.sizes) == 1:
            return [y]
        res, a = [], 0
        for n in self.sizes:
            res.append(y[:, a:a + n])
            a += n
        return res


class Layer:
    pass


class Weights:
    def __init__(self, cfg, T):
        def need(name):
            if name not in T:
                sys.exit(f"The file has no tensor '{name}', which the {cfg.arch} layout needs.")
            return T[name]

        def vec(name):
            return xp.asarray(np.asarray(need(name).data, dtype=np.float32))

        self.layers = []
        for l in range(cfg.n_layer):
            p = f"blk.{l}."
            L = Layer()
            L.attn_norm = vec(p + "attn_norm.weight")
            L.ffn_norm = vec(p + "ffn_norm.weight")
            L.q_norm = vec(p + "attn_q_norm.weight") if cfg.qk_norm else None
            L.k_norm = vec(p + "attn_k_norm.weight") if cfg.qk_norm else None
            biases = None
            if cfg.qkv_bias:
                biases = [vec(p + f"attn_{n}.bias") if (p + f"attn_{n}.bias") in T else None for n in "qkv"]
            L.qkv = Linear([need(p + "attn_q.weight"), need(p + "attn_k.weight"), need(p + "attn_v.weight")], biases)
            L.o = Linear([need(p + "attn_output.weight")])
            L.gate_up = Linear([need(p + "ffn_gate.weight"), need(p + "ffn_up.weight")])
            L.down = Linear([need(p + "ffn_down.weight")])
            self.layers.append(L)
        self.embd = need("token_embd.weight")
        self.tied = "output.weight" not in T
        self.out = Linear([self.embd if self.tied else T["output.weight"]])
        self.out_norm = vec("output_norm.weight")

    def matrices(self):
        """Every matrix that is multiplied, in the order they get VRAM. The embedding table is left out
        unless it doubles as the output matrix: only one row of it is read per token, so it stays in RAM."""
        seen, out = set(), []
        for L in self.layers:
            for lin in (L.qkv, L.o, L.gate_up, L.down):
                for t in lin.parts:
                    if id(t) not in seen:
                        seen.add(id(t))
                        out.append(t)
        for t in self.out.parts:
            if id(t) not in seen:
                out.append(t)
        return out


def _pinned(a):
    """A copy of a in page-locked RAM, which the GPU can copy from much faster. Plain RAM if that fails."""
    a = np.ascontiguousarray(a)
    try:
        mem = cupy.cuda.alloc_pinned_memory(a.nbytes)
        p = np.frombuffer(mem, dtype=a.dtype, count=a.size).reshape(a.shape)
        p[...] = a
        return p
    except Exception:
        return np.array(a)


def _to_dense(t, m, dtype):
    rows, cols = t.shape
    src = m.asarray(t.data)
    out = m.empty((rows, cols), dtype=dtype)
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(rows, r0 + CHUNK_ROWS)
        out[r0:r1] = decode_rows(src[r0:r1], t.kind, cols, m).astype(dtype)
    return out


def place(W, kv_bytes, dense):
    """GPU: move as many matrices as fit into VRAM (decoded to float16 first with --dense).
    The rest go to pinned RAM and are copied to the GPU each time they are used.
    CPU: with --dense, decode everything to float32 once, so no decoding is left for each token."""
    global STAGING
    mats = W.matrices()
    stats = {"gpu_n": 0, "gpu_bytes": 0, "ram_n": 0, "ram_bytes": 0}
    if not GPU:
        for t in mats:
            if dense and t.kind != "f32":
                t.data, t.kind = _to_dense(t, np, np.float32), "f32"
        return stats

    def gpu_size(t):
        return t.shape[0] * t.shape[1] * 2 if (dense and t.kind in BLOCKS) else t.data.nbytes
    free, _total = cupy.cuda.runtime.memGetInfo()
    budget = free - RESERVE_BYTES - kv_bytes
    sizes = [gpu_size(t) for t in mats]
    if sum(sizes) > budget:
        budget -= max(t.data.nbytes for t in mats)     # room for STAGING
    for t, size in zip(mats, sizes):
        if stats["gpu_bytes"] + size <= budget:
            if dense and t.kind in BLOCKS:
                t.data, t.kind = _to_dense(t, cupy, cupy.float16), "f16"
            else:
                t.data = cupy.asarray(t.data)
            t.on_gpu = True
            stats["gpu_n"] += 1
            stats["gpu_bytes"] += size
        else:
            t.data = _pinned(t.data)
            stats["ram_n"] += 1
            stats["ram_bytes"] += t.data.nbytes
    ram = [t.data.nbytes for t in mats if not t.on_gpu]
    if ram:
        STAGING = cupy.empty(max(ram), dtype=cupy.uint8)
    return stats


# ----------------------------------------------------------------------------- the model
def rms_norm(x, w, eps):
    return x / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + eps) * w


_silu_mul_gpu = None


def silu_mul(g, u):
    if _silu_mul_gpu is not None:
        return _silu_mul_gpu(g, u)
    with np.errstate(over="ignore"):
        return g / (1.0 + np.exp(-g)) * u


class Model:
    def __init__(self, cfg, W, max_ctx, kv_dtype):
        self.c, self.W = cfg, W
        self.max_ctx = min(max_ctx, cfg.ctx)
        shape = (cfg.n_layer, self.max_ctx, cfg.n_kv, cfg.hd)
        self.K = xp.zeros(shape, kv_dtype)
        self.V = xp.zeros(shape, kv_dtype)
        half = cfg.rope_dim // 2                 # rotary angles for every position, worked out once
        inv = 1.0 / (cfg.base ** (np.arange(half, dtype=np.float64) * 2.0 / cfg.rope_dim))
        ang = np.arange(self.max_ctx, dtype=np.float64)[:, None] * inv[None, :]
        self.cos = xp.asarray(np.cos(ang).astype(np.float32))
        self.sin = xp.asarray(np.sin(ang).astype(np.float32))
        self.scale = 1.0 / math.sqrt(cfg.hd)

    def embed(self, ids):
        t = self.W.embd
        if t.on_gpu:
            return decode_rows(t.data[xp.asarray(ids)], t.kind, t.shape[1])
        rows = np.ascontiguousarray(t.data[np.asarray(ids)])
        return xp.asarray(decode_rows(rows, t.kind, t.shape[1], np))

    def rotary(self, x, pos0):
        """x: (T, heads, head size) for positions pos0 .. pos0+T-1."""
        T = x.shape[0]
        rd, half = self.c.rope_dim, self.c.rope_dim // 2
        cos = self.cos[pos0:pos0 + T, None, :]
        sin = self.sin[pos0:pos0 + T, None, :]
        out = x.copy()
        if self.c.neox:
            a, b = x[..., :half], x[..., half:rd]
            out[..., :half] = a * cos - b * sin
            out[..., half:rd] = a * sin + b * cos
        else:
            a, b = x[..., 0:rd:2], x[..., 1:rd:2]
            out[..., 0:rd:2] = a * cos - b * sin
            out[..., 1:rd:2] = a * sin + b * cos
        return out

    def forward(self, ids, pos0, want_logits=True):
        """Run tokens ids, which sit at positions pos0, pos0+1, ... Returns the logits for the last one."""
        c, W = self.c, self.W
        T = len(ids)
        S = pos0 + T
        group = c.n_head // c.n_kv
        x = self.embed(ids)
        mask = None
        if T > 1:                                # token t may only look at positions up to pos0 + t
            mask = xp.arange(S)[None, :] > (pos0 + xp.arange(T))[:, None]
        for l, L in enumerate(W.layers):
            h = rms_norm(x, L.attn_norm, c.eps)
            q, k, v = L.qkv(h)
            q = q.reshape(T, c.n_head, c.hd)
            k = k.reshape(T, c.n_kv, c.hd)
            v = v.reshape(T, c.n_kv, c.hd)
            if c.qk_norm:
                q = rms_norm(q, L.q_norm, c.eps)
                k = rms_norm(k, L.k_norm, c.eps)
            q = self.rotary(q, pos0)
            k = self.rotary(k, pos0)
            self.K[l, pos0:S] = k
            self.V[l, pos0:S] = v
            Kc = self.K[l, :S].astype(xp.float32, copy=False)
            Vc = self.V[l, :S].astype(xp.float32, copy=False)
            qg = q.reshape(T, c.n_kv, group, c.hd)
            scores = xp.einsum("tkgd,skd->kgts", qg, Kc) * self.scale
            if mask is not None:
                scores = xp.where(mask, xp.float32(-1e30), scores)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = xp.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            att = xp.einsum("kgts,skd->tkgd", probs, Vc).reshape(T, -1)
            x = x + L.o(att)[0]
            h = rms_norm(x, L.ffn_norm, c.eps)
            gate, up = L.gate_up(h)
            x = x + L.down(silu_mul(xp.ascontiguousarray(gate), xp.ascontiguousarray(up)))[0]
        if not want_logits:
            return None
        x = rms_norm(x[-1:], W.out_norm, c.eps)
        return W.out(x)[0][0]

    def prefill(self, ids, pos0, want_logits=True):
        """Run a run of tokens in batches of PREFILL_CHUNK. Returns the logits for the last one."""
        logits = None
        for i in range(0, len(ids), PREFILL_CHUNK):
            last = i + PREFILL_CHUNK >= len(ids)
            logits = self.forward(ids[i:i + PREFILL_CHUNK], pos0 + i, want_logits and last)
        return logits


class Sampler:
    def __init__(self, temp, top_k, top_p, seed=None):
        self.temp, self.top_k, self.top_p = temp, top_k, top_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, logits):
        if self.temp <= 0:
            return int(xp.argmax(logits))        # only one number crosses back from the GPU
        n = logits.shape[0]
        if 0 < self.top_k < n:                   # pick the top k on the device, copy only those back
            idx_d = xp.argpartition(-logits, self.top_k)[:self.top_k]
            idx, lg = to_host(idx_d), to_host(logits[idx_d])
        else:
            idx, lg = np.arange(n), to_host(logits)
        order = np.argsort(-lg)
        idx, lg = idx[order], lg[order].astype(np.float64) / self.temp
        p = np.exp(lg - lg[0])
        p /= p.sum()
        if self.top_p < 1.0:                     # keep the smallest set whose chances add up to top_p
            keep = int(np.searchsorted(np.cumsum(p), self.top_p)) + 1
            idx, p = idx[:keep], p[:keep] / p[:keep].sum()
        return int(self.rng.choice(idx, p=p))


# ----------------------------------------------------------------------------- generation and chat
def generate(model, tok, logits, pos, limit, stops, sample):
    """Write up to limit tokens, printing as it goes. pos is how many tokens the cache holds.
    Returns (new tokens, new pos). The last token written is not run through the network yet."""
    reply, shown = [], ""
    while len(reply) < limit:
        nxt = sample(logits)
        if nxt in stops or pos >= model.max_ctx:
            break
        reply.append(nxt)
        text = tok.decode(reply)
        if not text.endswith("�"):
            print(text[len(shown):], end="", flush=True)
            shown = text
        if len(reply) == limit:
            break
        logits = model.forward([nxt], pos)
        pos += 1
    text = tok.decode(reply)
    print(text[len(shown):], end="", flush=True)
    return reply, pos


def speed(n, secs):
    return f"{n} tokens in {secs:.2f}s ({n / max(secs, 1e-6):.1f} tokens/s)"


class ChatFormat:
    """ChatML (<|im_start|>user ... <|im_end|>) when the vocabulary has it, otherwise Mistral's [INST] ... [/INST]."""

    def __init__(self, tok, system=None):
        self.tok = tok
        self.head = [tok.bos] if (tok.add_bos and tok.bos is not None) else []
        self.stops = {tok.eos}
        if "<|im_start|>" in tok.index and "<|im_end|>" in tok.index:
            self.kind = "chatml"
            self.end = tok.index["<|im_end|>"]
            self.stops.add(self.end)
            if "<|endoftext|>" in tok.index:
                self.stops.add(tok.index["<|endoftext|>"])
            if system:
                self.head += tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n", bos=False)
        else:
            self.kind = "inst"
            self.end = tok.eos
            self.system = system

    def user(self, text, first):
        if self.kind == "chatml":
            return self.tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n", bos=False)
        if first and self.system:
            text = f"{self.system}\n\n{text}"
        if getattr(self.tok, "pre", "") == "tekken":
            return self.tok.encode(f"[INST]{text}[/INST]", bos=False)
        return self.tok.encode(f"[INST] {text} [/INST]", bos=False)

    def close(self, reply):
        """The reply as stored in the conversation, with its end marker."""
        if self.kind == "chatml":
            return reply + [self.end] + self.tok.encode("\n", bos=False)
        return reply + [self.end]


def chat_loop(model, tok, args, sample):
    """turns holds (user tokens, reply tokens) pairs. pos is how many tokens the model's memory already holds,
    so each turn only runs the new tokens through the network."""
    fmt = ChatFormat(tok, args.system)
    head = fmt.head
    reply_cap = min(REPLY_CAP, model.max_ctx // 2)
    turns = []
    pos = 0
    print(f"Chat started ({'ChatML' if fmt.kind == 'chatml' else '[INST]'} format). "
          "Type a message and press Enter. Commands: /clear to forget, /quit to leave.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in ("/quit", "/exit"):
            return
        if user.lower() == "/clear":
            turns, pos = [], 0
            print("(conversation cleared)\n")
            continue

        new_user = fmt.user(user, first=not turns)

        def used(ts):
            return len(head) + sum(len(u) + len(r) for u, r in ts)
        trimmed = False
        while turns and used(turns) + len(new_user) + reply_cap > model.max_ctx:
            turns.pop(0)                         # forget the oldest exchange to make room
            trimmed = True
        if used(turns) + len(new_user) + reply_cap > model.max_ctx:
            print("That message is too long for --max-ctx. Use a bigger value or a shorter message.\n")
            continue
        if trimmed:
            pos = 0                              # the start of the conversation changed, so the memory is rebuilt

        seq = head + [t for u, r in turns for t in (u + r)] + new_user
        pending = seq[pos:]                      # tokens the model has not seen yet
        t0 = time.time()
        logits = model.prefill(pending, pos)
        sync()
        t_pre = time.time() - t0
        pos += len(pending)

        print("\nAI: ", end="", flush=True)
        t0 = time.time()
        reply, pos = generate(model, tok, logits, pos, reply_cap, fmt.stops, sample)
        t_gen = time.time() - t0
        print()
        if args.stats:
            print(f"[prompt {speed(len(pending), t_pre)}; reply {speed(len(reply), t_gen)}]")
        print()
        turns.append((new_user, fmt.close(reply)))   # tokens after pos are run at the start of the next turn


# ----------------------------------------------------------------------------- main
def generate_stream(model, tok, logits, pos, limit, stops, sample, emit, cancel):
    """Like generate(), but hands each piece of text to emit(text) and stops early when cancel() is true.
    Returns (new tokens, new pos)."""
    reply, shown = [], ""
    while len(reply) < limit:
        if cancel():
            break
        nxt = sample(logits)
        if nxt in stops or pos >= model.max_ctx:
            break
        reply.append(nxt)
        text = tok.decode(reply)
        if not text.endswith("\ufffd"):
            emit(text[len(shown):])
            shown = text
        if len(reply) == limit:
            break
        logits = model.forward([nxt], pos)
        pos += 1
    text = tok.decode(reply)
    emit(text[len(shown):])
    return reply, pos


class LLMSession:
    """A loaded model plus one conversation, for a UI to drive (the same work chat_loop does, without the console).
    Call every method from the same thread: the GUI uses one worker thread for all model work."""

    def __init__(self, path, use_gpu=True, dense=False, max_ctx=2048, kv32=False):
        init_llm_backend(use_gpu)
        t0 = time.time()
        try:
            with open(path, "rb") as f:
                info = read_gguf(f)
            md = info["metadata"]
            self.cfg = Config(md)
            self.tok = make_tokenizer(md)
            W = Weights(self.cfg, load_weights(path, info))
        except GGUFError as e:
            raise RuntimeError(f"Could not read the model: {e}")
        except SystemExit as e:                       # the loaders report problems with sys.exit(message)
            raise RuntimeError(str(e.code))
        cfg = self.cfg
        self.max_ctx = min(max_ctx, cfg.ctx)
        kv_dtype = xp.float16 if (GPU and not kv32) else xp.float32
        kv_bytes = cfg.n_layer * self.max_ctx * cfg.n_kv * cfg.hd * np.dtype(kv_dtype).itemsize * 2
        self.stats = place(W, kv_bytes, dense)
        if GPU:
            setup_fused(W.matrices())
        self.model = Model(cfg, W, self.max_ctx, kv_dtype)
        self.gpu = GPU
        self.name = os.path.basename(path)
        gb = 1024 ** 3
        self.summary = (f"{cfg.arch}, {cfg.n_layer} layers, {cfg.n_embd} wide, vocabulary {len(self.tok.tokens)}, "
                        f"context {self.max_ctx}.  " +
                        (f"GPU: {self.stats['gpu_n']} matrices in VRAM ({self.stats['gpu_bytes'] / gb:.1f} GB), "
                         f"{self.stats['ram_n']} in RAM." if GPU else "CPU.") +
                        f"  Loaded in {time.time() - t0:.1f}s.")
        self.fmt = None
        self.turns = []
        self.pos = 0

    def clear(self):
        self.turns, self.pos, self.fmt = [], 0, None

    def chat(self, text, emit, cancel=lambda: False, temp=0.7, top_k=40, top_p=0.95, max_tokens=256, system=None):
        """Send one user message, stream the reply through emit(text).  Returns a one-line speed summary."""
        tok, model = self.tok, self.model
        if not self.turns or self.fmt is None:
            self.fmt = ChatFormat(tok, system or None)
            self.turns, self.pos = [], 0
        fmt = self.fmt
        head = fmt.head
        reply_cap = max(8, min(int(max_tokens), model.max_ctx // 2))
        new_user = fmt.user(text, first=not self.turns)

        def used(ts):
            return len(head) + sum(len(u) + len(r) for u, r in ts)
        trimmed = False
        while self.turns and used(self.turns) + len(new_user) + reply_cap > model.max_ctx:
            self.turns.pop(0)                         # forget the oldest exchange to make room
            trimmed = True
        if used(self.turns) + len(new_user) + reply_cap > model.max_ctx:
            raise ValueError("That message is too long for the context size. Use a larger context or a shorter message.")
        if trimmed:
            self.pos = 0                              # the start of the conversation changed: rebuild the memory
        seq = head + [t for u, r in self.turns for t in (u + r)] + new_user
        pending = seq[self.pos:]
        t0 = time.time()
        logits = model.prefill(pending, self.pos)
        sync()
        t_pre = time.time() - t0
        self.pos += len(pending)
        t0 = time.time()
        reply, self.pos = generate_stream(model, tok, logits, self.pos, reply_cap, fmt.stops,
                                          Sampler(temp, top_k, top_p), emit, cancel)
        t_gen = time.time() - t0
        self.turns.append((new_user, fmt.close(reply)))
        return f"prompt {speed(len(pending), t_pre)}; reply {speed(len(reply), t_gen)}"



def gguf_main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    init_llm_backend("--cpu" not in argv and "--info" not in argv)
    ap = argparse.ArgumentParser(prog="physics_engine.py --gguf")
    ap.add_argument("model", nargs="?", help="path to a .gguf file")
    ap.add_argument("--prompt", default=None, help="run this one prompt and stop, instead of chatting")
    ap.add_argument("--tokens", type=int, default=60, help="how many tokens to write")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = always the most likely word; 0.7 = more varied")
    ap.add_argument("--top-k", type=int, default=40, help="sample only from this many likeliest tokens (0 = all)")
    ap.add_argument("--top-p", type=float, default=1.0, help="sample only from the likeliest tokens adding up to this")
    ap.add_argument("--seed", type=int, default=None, help="random seed, for repeatable sampling")
    ap.add_argument("--system", default=None, help="system prompt for chat")
    ap.add_argument("--max-ctx", type=int, default=2048, help="how much text it can hold at once (uses memory)")
    ap.add_argument("--dense", action="store_true",
                    help="decode weights once at load (float16 in VRAM, float32 on CPU): faster, more memory")
    ap.add_argument("--kv32", action="store_true", help="keep the GPU key/value cache in float32 instead of float16")
    ap.add_argument("--no-fused", action="store_true", help="don't use the fused GPU kernels (for comparison)")
    ap.add_argument("--stats", action="store_true", help="show speeds after each chat reply")
    ap.add_argument("--info", action="store_true", help="print the model's settings and tensor list, then stop")
    ap.add_argument("--all", action="store_true", help="with --info: show every settings value in full")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU even if a GPU is available")
    args = ap.parse_args(argv)

    path = args.model
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        found = sorted(f for f in os.listdir(here) if f.lower().endswith(".gguf"))
        if not found:
            sys.exit("Give a model path, or put a .gguf in this folder.")
        path = os.path.join(here, found[0])

    mode = "GPU (VRAM first, overflow to RAM)" if GPU else "CPU only"
    print(f"Running on: {mode}")
    print(f"Reading {os.path.basename(path)} ...")
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            info = read_gguf(f)
        md = info["metadata"]
        if args.info:
            report(path, info, args.all)
            return
        cfg = Config(md)
        tok = make_tokenizer(md)
        W = Weights(cfg, load_weights(path, info))
    except GGUFError as e:
        sys.exit(f"Could not read the model: {e}")

    max_ctx = min(args.max_ctx, cfg.ctx)
    kv_dtype = xp.float16 if (GPU and not args.kv32) else xp.float32
    kv_bytes = cfg.n_layer * max_ctx * cfg.n_kv * cfg.hd * np.dtype(kv_dtype).itemsize * 2
    stats = place(W, kv_bytes, args.dense)
    if GPU and not args.no_fused:
        setup_fused(W.matrices())
    gb = 1024 ** 3
    print(f"Loaded {cfg.arch}: {cfg.n_layer} layers, {cfg.n_embd} wide, {cfg.n_head} heads "
          f"({cfg.n_kv} key/value heads), vocabulary {len(tok.tokens)}, in {time.time() - t0:.1f}s.")
    if GPU:
        print(f"Weights: {stats['gpu_n']} matrices in VRAM ({stats['gpu_bytes'] / gb:.2f} GB), "
              f"{stats['ram_n']} in RAM ({stats['ram_bytes'] / gb:.2f} GB, copied to the GPU as needed).")
        print(f"Fused kernels: {', '.join(sorted(FUSED)) or 'off'}.")
    elif not args.dense:
        print("Tip: --dense decodes the weights once at load, which makes CPU runs much faster if you have the RAM.")

    model = Model(cfg, W, args.max_ctx, kv_dtype)
    sample = Sampler(args.temp, args.top_k, args.top_p, args.seed)
    if args.prompt is None:
        chat_loop(model, tok, args, sample)
        return

    ids = tok.encode(args.prompt)
    if len(ids) >= model.max_ctx:
        sys.exit("The prompt is longer than --max-ctx.")
    print(f"\n{args.prompt}", end="", flush=True)
    t0 = time.time()
    logits = model.prefill(ids, 0)
    sync()
    t_pre = time.time() - t0
    t0 = time.time()
    generated, _ = generate(model, tok, logits, len(ids), args.tokens, {tok.eos}, sample)
    t_gen = time.time() - t0
    print(f"\n\n[prompt {speed(len(ids), t_pre)}; generated {speed(len(generated), t_gen)}]")


# ####################################################################################################
##### PART: 4  DESTRUCTION LAB  (a 3D object made of chunks held by breakable bonds; hit it and watch it crumble)
# ####################################################################################################
# The object is a lattice of small chunks (spheres).  Each chunk is joined to its 18 nearest neighbours by a spring
# bond with its own strength and damage.  Hit it and the bonds near the hit stretch; a bond that stretches past its
# limit (or wears out from repeated stress) breaks.  Pieces that lose their bonds fall, bounce and pile up.
# Everything runs on the GPU in Warp, and the picture is drawn by Warp too (no extra libraries).
# Run:    python physics_engine.py --lab          (right-drag orbits, wheel zooms, left-click uses the tool)
#         python physics_engine.py --lab --gl     the same, with the picture drawn through OpenGL straight from the GPU
#                                                 (pip install pyglet; falls back to the Tk window if that is unavailable)
# Tools:  hammer, pick, blast, laser (heat + fire), cannon, machine gun, gravity well, weaken, mend.   Conditions: wind, earthquake.

LAB_NB = 18
LAB_OFFS = ([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            + [(a, b, 0) for a in (1, -1) for b in (1, -1)]
            + [(a, 0, b) for a in (1, -1) for b in (1, -1)]
            + [(0, a, b) for a in (1, -1) for b in (1, -1)])
LAB_REV = [LAB_OFFS.index((-a, -b, -c)) for a, b, c in LAB_OFFS]
LAB_CHUNK = 0.1                       # chunk size in metres (for the smaller detail levels; finer levels shrink it, see build())
LAB_G = 9.81
LAB_DT = 1.0 / 1440.0                 # one physics step; 24 of them = 1/60 s (stiff bonds need short steps)
LAB_SUBSTEPS = 24
LAB_G_RAMP = 0.8                      # seconds gravity takes to come in after the first interaction
LAB_STEPS_DEFAULT = 40                 # the Steps slider's value at launch
LAB_STEPS_MIN, LAB_STEPS_MAX = 5, 160      # range of the Steps slider (physics steps per 1/60 s)
LAB_REF_DENSITY = 2500.0              # hit strength is quoted for a stone chunk of the current size
LAB_GRID_DIM = 128                    # hash-grid cells per axis (was 64; the grid wraps, so a small one aliases far-flung debris onto the block)
LAB_STATS_EVERY = 12                  # stepped frames between stats readouts (the readout copies ~15 MB and counts pieces)
LAB_WAKE_FRAMES = 120                 # frames the physics keeps running after the last thing that could change it
LAB_WAKE_KE = 0.5                     # kinetic energy (J) above which the object still counts as moving
LAB_WAKE_HEAT = 0.02                  # hottest chunk above this keeps the physics running (heat spreads and burns by itself)
LAB_MELT_MU = 0.03                    # friction between liquid chunks and the ground (the particle solver's sand uses 0.5)
LAB_MELT_RELAX = 0.7                  # relaxation of the overlap push (same as the particle solver, P_RELAX)
LAB_MELT_ITERS = 3                    # overlap-push passes per substep (same as the particle solver, P_ITERS)
LAB_FREEZE = 0.7                      # a liquid chunk sets again when its heat falls below this fraction of the melt heat
LAB_LATENT = 0.35                     # liquid cools this much slower than solid (it holds its heat)
# Layout of the lab's small device-side parameter array.  The host writes it once per frame; the kernels read it, so
# the substep loop never needs a per-substep argument from Python and can be recorded once as a CUDA graph.
LP_T, LP_QAMP, LP_QHZ, LP_G = 0, 1, 2, 3          # sim time (advanced on the GPU), earthquake amplitude, its Hz, gravity
LP_WX, LP_WY, LP_WZ, LP_WIND_ON = 4, 5, 6, 7        # wind velocity, wind switch
LP_WELL_ON, LP_CX, LP_CY, LP_CZ, LP_GM = 8, 9, 10, 11, 12    # gravity well: switch, point, strength
LP_LOAD = 13                                        # 1: the last substep records the hardest-loaded bond
LP_GRAB_ON, LP_GAIN, LP_GX, LP_GY, LP_GZ = 14, 15, 16, 17, 18    # grab tools: switch, spring gain, where the held patch is pulled to
LP_N = 20
LAB_CC_ITERS = 400                    # most label-spreading rounds the GPU piece counter will run
LAB_AA_MODES = {"Off": (0, 1), "FXAA": (1, 1), "SSAA 2x": (0, 2), "SSAA 4x": (0, 4)}      # name -> (FXAA on, supersampling factor)
LAB_RES_SIZES = {"1080p": (1920, 1080), "720p": (1280, 720), "480p": (854, 480), "240p": (426, 240)}
LAB_RESOLUTIONS = tuple(LAB_RES_SIZES)           # the Resolution drop-down: the picture is rendered at this size
LAB_FULL_MODES = LAB_RESOLUTIONS                 # fullscreen: the same four, the picture is stretched to fill the screen


def lab_res_size(name):
    """(width, height) of a Resolution entry: '720p' etc., or an explicit 'W x H' (used by the tests)."""
    if name in LAB_RES_SIZES:
        return LAB_RES_SIZES[name]
    try:
        w, h = (int(x) for x in str(name).replace("x", " ").split())
        return w, h
    except ValueError:
        return LAB_RES_SIZES["720p"]


class LabMaterial:
    def __init__(self, name, density, k, eb, ey, rate, noise, colour, burn=0.0, melt=0.0, visc=0.0, grip=0.03,
                 ductile=0.0, hard=0.0, gfric=0.6, gstick=1.0):
        self.name, self.density, self.k = name, density, k
        self.eb, self.ey, self.rate, self.noise, self.colour = eb, ey, rate, noise, colour
        self.burn = burn                  # heat the material makes by itself once it is alight (wood: fire spreads)
        self.visc = visc                  # liquid thickness: velocity damping per second (water ~0, aluminium 0.8, slag 3, glass 6)
        self.grip = grip                  # friction of the liquid on the ground and on itself (the particle solver's mu)
        self.melt = melt                  # heat at which a chunk turns to liquid (0 = never: it burns away at h = 1 instead)
        self.ductile = ductile            # > 0: the bonds yield plastically (bend and stay bent); this is the plastic strain at which one snaps
        self.gfric, self.gstick = gfric, gstick   # ground grip: friction coefficient, and how hard it holds a slow slide (1 = as before)
        self.hard = hard                  # strain hardening: how much the yield strain grows per unit of plastic strain


LAB_MATERIALS = {m.name: m for m in (
    LabMaterial("Stone", 2500.0, 3.0e5, 0.025, 0.010, 3.0, 0.35, (0.64, 0.63, 0.60), melt=0.70, visc=3.0, grip=0.12),
    LabMaterial("Wood", 600.0, 7.0e4, 0.060, 0.025, 1.5, 0.45, (0.66, 0.45, 0.25), burn=1.3),
    LabMaterial("Glass", 2500.0, 3.5e5, 0.010, 0.005, 6.0, 0.10, (0.50, 0.78, 0.90), melt=0.55, visc=6.0, grip=0.08),
    LabMaterial("Aluminium", 2700.0, 3.3e6, 0.040, 0.004, 3.0, 0.10, (0.78, 0.80, 0.83), melt=0.45, visc=0.8, grip=0.04,
                ductile=0.25, hard=0.05, gfric=1.2, gstick=3.0),
    LabMaterial("Steel", 7850.0, 1.0e7, 0.050, 0.005, 3.0, 0.08, (0.36, 0.38, 0.42), melt=0.85, visc=1.5, grip=0.05,
                ductile=0.30, hard=0.05, gfric=1.4, gstick=3.0),
    LabMaterial("Clay", 1600.0, 1.5e5, 0.018, 0.006, 4.0, 0.50, (0.58, 0.38, 0.26), melt=0.80, visc=4.5, grip=0.15),
    # heavy, stiff and brittle; grips the ground hard (gfric, gstick) so a concrete base stays where it is put
    LabMaterial("Concrete", 2400.0, 3.5e5, 0.012, 0.007, 4.0, 0.30, (0.58, 0.58, 0.56), melt=0.80, visc=4.0, grip=0.15,
                gfric=1.6, gstick=6.0),
)}
LAB_SIZES = {"Small": 12, "Medium": 16}     # chunks along a block edge (Small ~1,700 chunks as a Block, Medium ~4,100)
LAB_DETAIL_MIN, LAB_DETAIL_MAX = 1500, 500000     # the Detail slider: target number of chunks (log scale)
LAB_DETAIL_CAUTION, LAB_DETAIL_WARN = 250000, 400000   # above these the panel shows CAUTIOUS / WARNING
LAB_BASE_N = 22                       # up to this many chunks per edge the chunk stays 0.1 m; beyond it the object keeps its size and the chunks shrink
LAB_SHAPES = ("Block", "Wall", "Pillar", "Arch")


def lab_shape_mask(shape, n):
    """Boolean grid (x, y, z) of where chunks exist.  y is up."""
    if shape == "Wall":
        nx, ny, nz = 2 * n, n, max(3, n // 3)
    elif shape == "Pillar":
        nx, ny, nz = max(4, n // 2), 2 * n, max(4, n // 2)
    elif shape == "Arch":
        nx, ny, nz = 2 * n, n + n // 2, max(3, n // 3)
    else:
        nx, ny, nz = n, n, n
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    m = np.ones((nx, ny, nz), bool)
    if shape == "Arch":
        cx, r = (nx - 1) / 2.0, n * 0.55
        hole = ((x - cx) ** 2 + (y - 0.0) ** 2 <= r * r) & (y < n * 0.9)
        top_round = y > ny - 1 - n // 6
        m = ~hole
        m &= ~(top_round & ((x - cx) ** 2 + (y - (ny - 1 - n // 6)) ** 2 > (nx / 2.0) ** 2))
    return m


_LAB_COUNT_CACHE = {}


def lab_shape_count(shape, n):
    """How many chunks `shape` has at edge size n (cached)."""
    key = (shape, int(n))
    if key not in _LAB_COUNT_CACHE:
        _LAB_COUNT_CACHE[key] = int(lab_shape_mask(shape, int(n)).sum())
    return _LAB_COUNT_CACHE[key]


def lab_edge_for(shape, target):
    """The edge size n whose chunk count for `shape` is closest to `target` (binary search; counts grow with n)."""
    lo, hi = 4, 200
    while lo < hi:
        mid = (lo + hi) // 2
        if lab_shape_count(shape, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if lo > 4 and abs(lab_shape_count(shape, lo - 1) - target) < abs(lab_shape_count(shape, lo) - target):
        lo -= 1
    return lo


def lab_detail_target(frac):
    """Slider position 0..1 -> target chunk count (log scale), rounded to 3 significant figures."""
    v = LAB_DETAIL_MIN * (LAB_DETAIL_MAX / LAB_DETAIL_MIN) ** min(1.0, max(0.0, float(frac)))
    return int(float(f"{v:.3g}"))


def lab_detail_level(count):
    """'' / 'CAUTIOUS' / 'WARNING' for a chunk count."""
    return "WARNING" if count > LAB_DETAIL_WARN else ("CAUTIOUS" if count > LAB_DETAIL_CAUTION else "")


@wp.kernel
def k_lab_bonds(k0: float, cdamp: float, dt: float, rate: float, last_sub: int,
                P: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                nbr: wp.array(dtype=int), alive: wp.array(dtype=int), dmg: wp.array(dtype=float),
                strength: wp.array(dtype=float), rest: wp.array(dtype=float), kscale: wp.array(dtype=float),
                force: wp.array(dtype=wp.vec3), gone: wp.array(dtype=int), load: wp.array(dtype=float),
                full: wp.array(dtype=int), plast: wp.array(dtype=float), bprop: wp.array(dtype=wp.vec4), hard: float):
    """Bond forces AND bond breaking in one pass.  For every live bond of this chunk:
    strain = (L - L0) / L0.  Past the yield strain the bond collects damage; past the break strain (or full damage, or
    hard compression) it snaps and gives no force.  Otherwise it pulls with the spring + damper, weakened by damage:
        F = k (1 - 0.6 d) (L - L0) + c (v_rel . n)
    Every bond exists twice (once in each end chunk's list) and each end updates ITS OWN copy of the damage and the
    alive flag.  Both ends compute the same strain from the same positions, so the copies always agree and no thread
    writes to a neighbour's data.  (Anything else that changes a bond's damage must change both copies by the same
    amount: see k_lab_heat_b and k_lab_weaken.)
    Ductile materials (ep_max > 0, e.g. steel) yield instead: the bond keeps a permanent plastic strain ep, the elastic
    strain is ee = strain - ep, and when |ee| passes the yield limit y = ey*st + hard*|ep| the excess flows into ep
    (return mapping:  ep += (|ee| - y) / (1 + hard)).  The bond only snaps when |ep| > ep_max*st, so a beam bends, stays
    bent, and fails after a large deformation rather than all at once.  The force uses the elastic part only.
    Each bond carries its own limits in bprop = (break strain eb, yield strain ey, ep_max, stiffness multiplier), so one
    structure can be made of several materials: the host fills them from the two chunks a bond joins.
    last_sub = 1 on the last substep of a frame: if the host asked (P[LP_LOAD]) this chunk also reports how close its
    most stretched bond is to snapping (1.0 = at the limit), as one atomic per chunk."""
    i = wp.tid()
    f = wp.vec3(0.0, 0.0, 0.0)
    if gone[i] == 1:
        force[i] = f
        full[i] = 0
        return
    track = int(0)
    nlive = int(0)
    if last_sub == 1 and P[LP_LOAD] > 0.5:
        track = 1
    worst = float(0.0)
    for k in range(18):
        e = i * 18 + k
        if alive[e] == 1:
            j = nbr[e]
            d = pos[j] - pos[i]
            ln = wp.length(d)
            eps = (ln - rest[k]) / rest[k]
            st = strength[e]
            dm = dmg[e]
            ep = plast[e]
            bpe = bprop[e]
            eb = bpe[0]
            ey = bpe[1]
            ep_max = bpe[2]
            kmul = bpe[3]
            broke = int(0)
            if ep_max > 0.0:
                ee = eps - ep
                yl = ey * st + hard * wp.abs(ep)
                if ee > yl:
                    ep = ep + (ee - yl) / (1.0 + hard)
                    plast[e] = ep
                elif ee < -yl:
                    ep = ep + (ee + yl) / (1.0 + hard)
                    plast[e] = ep
                if track == 1:
                    worst = wp.max(worst, wp.abs(ep) / (ep_max * st))
                if wp.abs(ep) > ep_max * st or dm >= 1.0:
                    broke = 1
            else:
                if track == 1:
                    worst = wp.max(worst, wp.max(eps / (eb * st), -eps / (3.0 * eb * st)))
                if eps > ey * st:
                    dm = dm + rate * dt * (eps - ey * st) / ((eb - ey) * st)
                    dmg[e] = dm
                if eps > eb * st or eps < -3.0 * eb * st or dm >= 1.0:
                    broke = 1
            if broke == 1:
                alive[e] = 0                                   # snapped: no force from this bond any more
            elif ln > 1.0e-6:
                nlive += 1
                n = d / ln
                soft = 1.0 - 0.6 * dm
                vr = wp.dot(vel[j] - vel[i], n)
                f = f + n * (kscale[k] * (k0 * kmul * soft * (ln - rest[k] * (1.0 + ep)) + cdamp * vr))
    force[i] = f
    full[i] = 1 if nlive == 18 else 0                  # buried: every neighbour is bonded (k_lab_contacts skips these)
    if track == 1 and worst > 0.0:
        wp.atomic_max(load, 0, worst)


@wp.kernel
def k_lab_contacts(grid: wp.uint64, s: float, kc: float, cc: float, mu: float, gprop: wp.array(dtype=wp.vec2),
                   pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), force: wp.array(dtype=wp.vec3),
                   gone: wp.array(dtype=int), full: wp.array(dtype=int), molten: wp.array(dtype=int)):
    """Chunks that overlap push apart (penalty spring + damper + Coulomb friction); the ground is a plane at y = 0.
    A liquid chunk gets no contact force here: its overlaps and the ground are handled by k_lab_melt_*."""
    i = wp.tid()
    if gone[i] == 1 or molten[i] == 1:
        return
    if full[i] == 1:
        return                                       # buried: every neighbour is bonded, nothing else can reach it
    pi = pos[i]
    vi = vel[i]
    f = force[i]
    q = wp.hash_grid_query(grid, pi, s)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j == i or gone[j] == 1:
            continue
        d = pi - pos[j]
        dist = wp.length(d)
        pen = s - dist
        if pen > 0.0 and dist > 1.0e-6:
            n = d / dist
            rv = vi - vel[j]
            vn = wp.dot(rv, n)
            fn = wp.max(kc * pen - cc * vn, 0.0)
            f = f + n * fn
            vt = rv - n * vn
            vtl = wp.length(vt)
            if vtl > 1.0e-6:
                f = f - vt / vtl * wp.min(mu * fn, cc * vtl)
    pen = 0.5 * s - pi[1]
    if pen > 0.0:
        fn = wp.max(kc * pen - cc * vi[1], 0.0)
        f = f + wp.vec3(0.0, fn, 0.0)
        vt = wp.vec3(vi[0], 0.0, vi[2])
        vtl = wp.length(vt)
        if vtl > 1.0e-6:
            gp = gprop[i]
            f = f - vt / vtl * wp.min(gp[0] * fn, gp[1] * cc * vtl)    # the ground grips harder than chunk on chunk (gprop = gmu, gstick: per chunk's material)
    force[i] = f


@wp.kernel
def k_lab_integrate(dt: float, P: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                    force: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), gone: wp.array(dtype=int),
                    molten: wp.array(dtype=int), pold: wp.array(dtype=wp.vec3), n_obj: int, shot_cap: float):
    """Semi-implicit Euler:  v += (F/m + a) dt ;  x += v dt,   where a = gravity + the earthquake's push.
    Gravity is eased in by the host over 0.8 s once it is on; both come from the parameter array.
    For a liquid chunk this is the particle solver's 'predict' step: the old position is kept (pold) and pos becomes the
    predicted position, which k_lab_melt_* then correct."""
    i = wp.tid()
    if gone[i] == 1:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    t = P[LP_T]
    g = P[LP_G]                                      # the host eases it in (DestructionLab.step)
    qa = P[LP_QAMP] * wp.sin(6.283185307 * P[LP_QHZ] * t)
    acc = wp.vec3(qa * 0.92, -g, qa * 0.39)
    v = vel[i] + (force[i] * inv_m[i] + acc) * dt
    sp = wp.length(v)
    cap = 6.0                                        # the object's chunks never exceed 6 m/s (keeps the contacts stable) ...
    if i >= n_obj:
        cap = shot_cap                               # ... but a cannonball or bullet flies at the speed it was fired
    if sp > cap:
        v = v * (cap / sp)
    v = v * 0.9995
    if sp < 0.06 and pos[i][1] < 0.06:
        v = v * 0.9                                  # rubble settles instead of creeping
    if molten[i] == 1:
        pold[i] = pos[i]
    vel[i] = v
    pos[i] = pos[i] + v * dt


@wp.kernel
def k_lab_melt_project(grid: wp.uint64, s: float, mu: float, molten: wp.array(dtype=int), gone: wp.array(dtype=int),
                       pos: wp.array(dtype=wp.vec3), pold: wp.array(dtype=wp.vec3), corr: wp.array(dtype=wp.vec3)):
    """The particle solver's overlap push (k_p_project), in 3D, for liquid chunks:
        pen = size - |xi - xj|            correction = n * pen * share      (two liquid chunks each take half)
        friction:  t = relative motion minus its part along n      correction -= t * min(1, mu * pen / |t|) * share
    A solid chunk does not move for the liquid (like a peg in the particle solver), so the liquid takes all of it."""
    i = wp.tid()
    if molten[i] == 0 or gone[i] == 1:
        return
    pi = pos[i]
    rel_i = pi - pold[i]
    c = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, pi, s)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j == i or gone[j] == 1:
            continue
        d = pi - pos[j]
        dist = wp.length(d)
        pen = s - dist
        if pen > 0.0:
            nrm = wp.vec3(0.0, 1.0, 0.0)
            if dist > 1.0e-6:
                nrm = d / dist
            elif i < j:
                nrm = wp.vec3(0.0, -1.0, 0.0)
            share = float(1.0)
            rel = rel_i
            if molten[j] == 1:
                share = 0.5
                rel = rel_i - (pos[j] - pold[j])
            c = c + nrm * (pen * share)
            tang = rel - nrm * wp.dot(rel, nrm)
            tl = wp.length(tang)
            if tl > 1.0e-9:
                c = c - tang * (wp.min(1.0, mu * pen / tl) * share)
    corr[i] = c


@wp.kernel
def k_lab_melt_apply(s: float, mu: float, relax: float, molten: wp.array(dtype=int), gone: wp.array(dtype=int),
                     pos: wp.array(dtype=wp.vec3), pold: wp.array(dtype=wp.vec3), corr: wp.array(dtype=wp.vec3)):
    """The particle solver's k_p_apply: relax the correction (x0.7, at most one radius), then the ground:
    pen = r - y,  the horizontal slide is reduced by min(1, mu * pen / |slide|)  (the ground is the plane y = 0)."""
    i = wp.tid()
    if molten[i] == 0 or gone[i] == 1:
        return
    r = 0.5 * s
    c = corr[i] * relax
    m = wp.length(c)
    if m > r:
        c = c * (r / m)
    p = pos[i] + c
    if p[1] < r:
        pen = r - p[1]
        d = p - pold[i]
        dh = wp.sqrt(d[0] * d[0] + d[2] * d[2])
        f = wp.min(1.0, mu * pen / (dh + 1.0e-6))
        p = wp.vec3(p[0] - d[0] * f, r, p[2] - d[2] * f)
    pos[i] = p


@wp.kernel
def k_lab_melt_update(dt: float, visc: float, meltc: wp.array(dtype=float), freeze_k: float, heat: wp.array(dtype=float), molten: wp.array(dtype=int),
                      gone: wp.array(dtype=int), pos: wp.array(dtype=wp.vec3), pold: wp.array(dtype=wp.vec3),
                      vel: wp.array(dtype=wp.vec3)):
    """The particle solver's k_p_update:  v = (x* - x) / dt * 0.9998 ;  if |v| < 0.2 m/s: v *= 0.9  (piles settle instead of
    creeping: the particle solver's 15 px/s) ;  capped at 6 m/s (lava is thick, unlike the 2D water)."""
    i = wp.tid()
    if molten[i] == 0 or gone[i] == 1:
        return
    v = (pos[i] - pold[i]) / dt * 0.9998
    freeze = meltc[i] * freeze_k
    # thickness: slag, molten glass and metal are not water.  Velocity is damped by exp(-visc dt), and the liquid
    # stiffens as it cools: the damping is x3 at the setting point and x1 at full heat.
    tt = wp.clamp((heat[i] - freeze) / wp.max(1.0 - freeze, 1.0e-3), 0.0, 1.0)
    v = v * wp.exp(-visc * (1.0 + 2.0 * (1.0 - tt)) * dt)
    sp = wp.length(v)
    if sp < 0.2:
        v = v * 0.9
    if sp > 6.0:
        v = v * (6.0 / sp)
    vel[i] = v


@wp.kernel
def k_lab_impulse(mode: int, center: wp.vec3, dirn: wp.vec3, radius: float, speed: float, mref: float,
                  pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float)):
    """A hit: every chunk within `radius` of the point gets a velocity change that fades with distance,
    dv = speed * (1 - d/R)^2 * (m_ref / m), along the hit direction (mode 0) or away from the point (mode 1)."""
    i = wp.tid()
    d = pos[i] - center
    dist = wp.length(d)
    if dist < radius:
        w = (1.0 - dist / radius) * (1.0 - dist / radius)
        u = dirn
        if mode == 1:
            u = d / wp.max(dist, 1.0e-4)
        vel[i] = vel[i] + u * (speed * w * mref * inv_m[i])


@wp.kernel
def k_lab_weaken(centre: wp.vec3, radius: float, amount: float, pos: wp.array(dtype=wp.vec3),
                 nbr: wp.array(dtype=int), alive: wp.array(dtype=int), dmg: wp.array(dtype=float)):
    """Paint damage onto (or mend) the bonds inside a sphere, without hitting anything."""
    e = wp.tid()
    if alive[e] == 0:
        return
    i = e / 18
    j = nbr[e]
    mid = (pos[i] + pos[j]) * 0.5
    dist = wp.length(mid - centre)
    if dist < radius:
        w = 1.0 - dist / radius
        dmg[e] = wp.clamp(dmg[e] + amount * w, 0.0, 0.98)


@wp.kernel
def k_lab_slice(eye: wp.vec3, d1: wp.vec3, d2: wp.vec3, nrm: wp.vec3, det: float, n_obj: int,
                pos: wp.array(dtype=wp.vec3), nbr: wp.array(dtype=int), peer: wp.array(dtype=int),
                alive: wp.array(dtype=int)):
    """The slicer: cut every bond that crosses the plane through the eye and the two screen points (d1, d2 are the
    rays through the ends of the dragged line, nrm = d1 x d2) and whose middle lies between those two rays, so only
    what the line was dragged across is cut, however deep."""
    e = wp.tid()
    i = e / 18
    if i >= n_obj or alive[e] == 0:
        return
    j = nbr[e]
    di = wp.dot(pos[i] - eye, nrm)
    dj = wp.dot(pos[j] - eye, nrm)
    if di * dj >= 0.0:
        return
    u = (pos[i] + pos[j]) * 0.5 - eye
    a = wp.dot(wp.cross(u, d2), nrm) / det                 # u = a d1 + b d2 + c nrm : inside the wedge when a, b >= 0
    b = wp.dot(wp.cross(d1, u), nrm) / det
    if a >= 0.0 and b >= 0.0:
        alive[e] = 0
        alive[peer[e]] = 0


@wp.kernel
def k_lab_kinetic(mass: wp.array(dtype=float), vel: wp.array(dtype=wp.vec3), total: wp.array(dtype=float)):
    """Kinetic energy 1/2 m v^2, summed over every chunk."""
    i = wp.tid()
    v = vel[i]
    wp.atomic_add(total, 0, 0.5 * mass[i] * wp.dot(v, v))


@wp.kernel
def k_lab_surface(pos: wp.array(dtype=wp.vec3), nbr: wp.array(dtype=int), alive: wp.array(dtype=int),
                  dmg: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3), dmgc: wp.array(dtype=float),
                  draw: wp.array(dtype=int), gone: wp.array(dtype=int)):
    """Per chunk: draw it only if it is on the surface (a bond is missing), its outward normal (away from the live
    neighbours' centre), and how damaged it is (broken bonds count as fully damaged)."""
    i = wp.tid()
    if gone[i] == 1:
        draw[i] = 0
        return
    s = wp.vec3(0.0, 0.0, 0.0)
    live = int(0)
    had = int(0)
    dsum = float(0.0)
    for k in range(18):
        e = i * 18 + k
        if nbr[e] >= 0:
            had += 1
            if alive[e] == 1:
                s = s + (pos[nbr[e]] - pos[i])
                live += 1
                dsum += dmg[e]
            else:
                dsum += 1.0                      # a broken bond counts as fully damaged
    n = wp.vec3(0.0, 1.0, 0.0)
    sl = wp.length(s)
    if live > 0 and sl > 1.0e-5:
        n = -s / sl
    nrm[i] = n
    dmgc[i] = dsum / float(wp.max(had, 1))
    if live == 0:
        dmgc[i] = 0.08                           # a loose chunk is just a lump of the material
    if live < 18:
        draw[i] = 1
    else:
        draw[i] = 0


@wp.kernel
def k_lab_sky(w: int, h: int, f: float, cx: float, cy: float, eye: wp.vec3, fwd: wp.vec3, right: wp.vec3,
              up: wp.vec3, col: wp.array(dtype=wp.vec3), depth: wp.array(dtype=int), ids: wp.array(dtype=int)):
    """Background: sky gradient and a checkered ground plane that fades into the distance.
    Also clears the depth buffer and the chunk-id buffer (-1 = no chunk under this pixel)."""
    t = wp.tid()
    py = t / w
    px = t - py * w
    d = fwd + right * ((float(px) - cx) / f) + up * (-(float(py) - cy) / f)
    d = d / wp.length(d)
    depth[t] = 1073741824
    ids[t] = -1
    sky = wp.vec3(0.20, 0.25, 0.32) * (1.0 - wp.clamp(d[1] * 1.6, 0.0, 1.0)) + wp.vec3(0.05, 0.07, 0.11) * wp.clamp(d[1] * 1.6, 0.0, 1.0)
    c = sky
    if d[1] < -0.001:
        tt = -eye[1] / d[1]
        gx = eye[0] + d[0] * tt
        gz = eye[2] + d[2] * tt
        chk = (int(wp.floor(gx * 2.0)) + int(wp.floor(gz * 2.0))) & 1
        g = wp.vec3(0.16, 0.18, 0.21)
        if chk == 1:
            g = wp.vec3(0.19, 0.21, 0.25)
        fog = wp.clamp(tt / 30.0, 0.0, 1.0)
        c = g * (1.0 - fog) + sky * fog
    col[t] = c


@wp.func
def _lab_bbox(c: wp.vec3, eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, f: float, cx: float, cy: float,
              h: float):
    """Screen rectangle (x0, y0, x1, y1) that covers the cube of half-size h at c.  Empty (x0 > x1) if it is behind us."""
    lox = float(1.0e9)
    loy = float(1.0e9)
    hix = float(-1.0e9)
    hiy = float(-1.0e9)
    ok = int(1)
    for k in range(8):
        sx = float(-1.0)
        sy = float(-1.0)
        sz = float(-1.0)
        if (k & 1) != 0:
            sx = 1.0
        if (k & 2) != 0:
            sy = 1.0
        if (k & 4) != 0:
            sz = 1.0
        rel = c + wp.vec3(sx * h, sy * h, sz * h) - eye
        zc = wp.dot(rel, fwd)
        if zc < 0.15:
            ok = 0
        else:
            x = cx + f * wp.dot(rel, right) / zc
            y = cy - f * wp.dot(rel, up) / zc
            lox = wp.min(lox, x)
            hix = wp.max(hix, x)
            loy = wp.min(loy, y)
            hiy = wp.max(hiy, y)
    if ok == 0:
        return wp.vec4(1.0, 1.0, 0.0, 0.0)
    return wp.vec4(lox, loy, hix, hiy)


@wp.func
def _lab_raybox(o: wp.vec3, d: wp.vec3, h: float):
    """Where a ray (origin o relative to the cube centre, direction d) first enters an axis-aligned cube of half-size h.
    Returns (distance, face normal x, y, z); distance < 0 means a miss.  (Slab method.)"""
    tmin = float(-1.0e9)
    tmax = float(1.0e9)
    nx = float(0.0)
    ny = float(0.0)
    nz = float(0.0)
    for a in range(3):
        da = d[a]
        oa = o[a]
        if wp.abs(da) < 1.0e-9:
            if wp.abs(oa) > h:
                return wp.vec4(-1.0, 0.0, 0.0, 0.0)
        else:
            t1 = (-h - oa) / da
            t2 = (h - oa) / da
            tn = wp.min(t1, t2)
            tf = wp.max(t1, t2)
            if tn > tmin:
                tmin = tn
                sg = float(1.0)
                if da > 0.0:
                    sg = -1.0
                nx = 0.0
                ny = 0.0
                nz = 0.0
                if a == 0:
                    nx = sg
                elif a == 1:
                    ny = sg
                else:
                    nz = sg
            tmax = wp.min(tmax, tf)
    if tmax < tmin or tmin < 0.0:
        return wp.vec4(-1.0, 0.0, 0.0, 0.0)
    return wp.vec4(tmin, nx, ny, nz)


@wp.func
def _lab_raysphere(o: wp.vec3, d: wp.vec3, rad: float):
    """Same contract as _lab_raybox, for a sphere of radius `rad` at the origin (used for liquid chunks):
    (distance, normal x, y, z); distance < 0 means a miss."""
    a = wp.dot(d, d)
    b = 2.0 * wp.dot(o, d)
    c = wp.dot(o, o) - rad * rad
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return wp.vec4(-1.0, 0.0, 0.0, 0.0)
    t = (-b - wp.sqrt(disc)) / (2.0 * a)
    if t < 0.0:
        return wp.vec4(-1.0, 0.0, 0.0, 0.0)
    n = (o + d * t) / rad
    return wp.vec4(t, n[0], n[1], n[2])


@wp.kernel
def k_lab_depth(w: int, h: int, f: float, cx: float, cy: float, s: float, eye: wp.vec3, fwd: wp.vec3,
                right: wp.vec3, up: wp.vec3, pos: wp.array(dtype=wp.vec3), draw: wp.array(dtype=int),
                depth: wp.array(dtype=int), molten: wp.array(dtype=int), n_obj: int, maxb: int):
    """Each surface chunk is a true cube: for every pixel inside its screen rectangle, shoot a ray at the cube and keep
    the nearest hit.  (The ray direction has length 1 along the view axis, so the hit distance is the depth.)
    A liquid chunk is a sphere a little wider than the cube, so neighbouring liquid chunks run together."""
    i = wp.tid()
    if draw[i] == 0:
        return
    hs = 0.5 * s
    bh = hs
    rnd = molten[i] == 1 or i >= n_obj                    # liquid chunks and cannonball / bullet chunks are spheres
    if molten[i] == 1:
        bh = hs * 1.15
    elif i >= n_obj:
        bh = hs * 1.1
    bb = _lab_bbox(pos[i], eye, fwd, right, up, f, cx, cy, bh)
    x0 = wp.max(int(bb[0]) - 1, 0)
    x1 = wp.min(int(bb[2]) + 1, w - 1)
    y0 = wp.max(int(bb[1]) - 1, 0)
    y1 = wp.min(int(bb[3]) + 1, h - 1)
    if x1 < x0 or y1 < y0 or x1 - x0 > maxb or y1 - y0 > maxb:
        return
    o = eye - pos[i]
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            d = fwd + right * ((float(px) + 0.5 - cx) / f) - up * ((float(py) + 0.5 - cy) / f)
            r = wp.vec4(-1.0, 0.0, 0.0, 0.0)
            if rnd:
                r = _lab_raysphere(o, d, bh)
            else:
                r = _lab_raybox(o, d, hs)
            if r[0] > 0.0:
                wp.atomic_min(depth, py * w + px, int(r[0] * 4000.0))


# ---- view modes: false-colour the same simulation state ----------------------------------------------------
# 0 Normal, 1 Damage, 2 Strain, 3 Pieces, 4 Velocity, 5 Heat.  k_lab_viewval works out one colour per chunk;
# k_lab_colour then only shades it (flat faces + edge outline), so every mode costs one small per-chunk pass.
@wp.func
def _lab_stop(pal: int, k: int):
    """Colour stops of the two ramps (pal 0: blue -> cyan -> green -> yellow -> red; pal 1: thermal camera)."""
    c = wp.vec3(0.0, 0.0, 0.0)
    if pal == 0:
        if k == 0:
            c = wp.vec3(0.10, 0.16, 0.35)
        elif k == 1:
            c = wp.vec3(0.10, 0.50, 0.70)
        elif k == 2:
            c = wp.vec3(0.35, 0.80, 0.40)
        elif k == 3:
            c = wp.vec3(0.98, 0.85, 0.20)
        else:
            c = wp.vec3(0.95, 0.20, 0.10)
    else:
        if k == 0:
            c = wp.vec3(0.07, 0.09, 0.20)
        elif k == 1:
            c = wp.vec3(0.45, 0.07, 0.42)
        elif k == 2:
            c = wp.vec3(0.90, 0.22, 0.12)
        elif k == 3:
            c = wp.vec3(1.00, 0.72, 0.12)
        else:
            c = wp.vec3(1.00, 0.97, 0.80)
    return c


@wp.func
def _lab_ramp(pal: int, t: float):
    x = wp.clamp(t, 0.0, 1.0) * 4.0
    k = wp.min(int(x), 3)
    fr = x - float(k)
    return _lab_stop(pal, k) * (1.0 - fr) + _lab_stop(pal, k + 1) * fr


@wp.func
def _lab_hsv(h: float, s: float, v: float):
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    b = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(((r - 1.0) * s + 1.0) * v, ((g - 1.0) * s + 1.0) * v, ((b - 1.0) * s + 1.0) * v)


@wp.kernel
def k_lab_ramp_sample(pal: int, out: wp.array(dtype=wp.vec3)):
    """Samples a ramp so the window's legend uses exactly the colours the GPU draws with."""
    i = wp.tid()
    out[i] = _lab_ramp(pal, float(i) / float(out.shape[0] - 1))


@wp.kernel
def k_lab_viewval(mode: int, n_obj: int, eb: float, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                  nbr: wp.array(dtype=int), alive: wp.array(dtype=int), dmg: wp.array(dtype=float),
                  strength: wp.array(dtype=float), rest: wp.array(dtype=float), heat: wp.array(dtype=float),
                  label: wp.array(dtype=int), sizes: wp.array(dtype=int), vcol: wp.array(dtype=wp.vec3),
                  plast: wp.array(dtype=float), ep_max: float):
    """One false colour per chunk.
    Damage:   t = sqrt(mean bond damage), a broken bond counts as 1  (a loose chunk is fully damaged)
    Strain:   t = sqrt(max over live bonds of eps / (eb * strength))  - the same ratio the break test uses, 1 = about to snap
    Pieces:   a hashed colour per connected piece (labels come from the GPU piece counter); loose single chunks are pale grey
    Velocity: t = sqrt(speed / 15 m/s)
    Heat:     t = sqrt(heat)  on the thermal ramp"""
    i = wp.tid()
    c = wp.vec3(0.5, 0.5, 0.5)
    if i >= n_obj:
        c = wp.vec3(0.30, 0.32, 0.36)                    # cannonballs
    elif mode == 1:
        had = int(0)
        dsum = float(0.0)
        for k in range(18):
            e = i * 18 + k
            if nbr[e] >= 0:
                had += 1
                if alive[e] == 1:
                    dsum += dmg[e]
                else:
                    dsum += 1.0
        c = _lab_ramp(0, wp.sqrt(dsum / float(wp.max(had, 1))))
    elif mode == 2:
        worst = float(0.0)
        for k in range(18):
            e = i * 18 + k
            if alive[e] == 1:
                ln = wp.length(pos[nbr[e]] - pos[i])
                eps = (ln - rest[k]) / rest[k]
                st = eb * strength[e]
                if ep_max > 0.0:
                    worst = wp.max(worst, wp.abs(plast[e]) / (ep_max * strength[e]))      # ductile: how far it has flowed
                else:
                    worst = wp.max(worst, wp.max(eps / st, -eps / (3.0 * st)))
        c = _lab_ramp(0, wp.sqrt(wp.clamp(worst, 0.0, 1.0)))
    elif mode == 3:
        lab = label[i]
        if sizes[lab] == 1:
            c = wp.vec3(0.82, 0.82, 0.86)
        else:
            rs = wp.rand_init(7, lab)
            c = _lab_hsv(wp.randf(rs), 0.60, 0.95)
    elif mode == 4:
        c = _lab_ramp(0, wp.sqrt(wp.clamp(wp.length(vel[i]) / 15.0, 0.0, 1.0)))
    elif mode == 5:
        c = _lab_ramp(1, wp.sqrt(wp.clamp(heat[i], 0.0, 1.0)))
    vcol[i] = c


@wp.kernel
def k_lab_colour(w: int, h: int, f: float, cx: float, cy: float, s: float, eye: wp.vec3, fwd: wp.vec3,
                 right: wp.vec3, up: wp.vec3, light: wp.vec3, hover: wp.vec3, hover_r: float, hover_col: wp.vec3,
                 pos: wp.array(dtype=wp.vec3), draw: wp.array(dtype=int), base: wp.array(dtype=wp.vec3),
                 dmgc: wp.array(dtype=float), heat: wp.array(dtype=float), depth: wp.array(dtype=int),
                 col: wp.array(dtype=wp.vec3), ids: wp.array(dtype=int), vmode: int, vcol: wp.array(dtype=wp.vec3),
                 molten: wp.array(dtype=int), n_obj: int, maxb: int):
    """Second pass: colour the pixels each cube won.  Flat faces lit from a fixed light, a thin dark outline on every
    face edge, darker and redder where damaged, tinted inside the tool's reach.  Each winning pixel also records which
    chunk it shows (ids), so the mouse can pick a chunk by reading one pixel instead of searching on the CPU.
    vmode > 0 swaps the material colour for the chunk's false colour from k_lab_viewval (same shading and outline).
    A liquid chunk is drawn as a smooth glowing sphere (no outline): deep red as it cools, yellow-white while hot."""
    i = wp.tid()
    if draw[i] == 0:
        return
    hs = 0.5 * s
    bh = hs
    rnd = molten[i] == 1 or i >= n_obj                    # liquid chunks and cannonball / bullet chunks are spheres
    if molten[i] == 1:
        bh = hs * 1.15
    elif i >= n_obj:
        bh = hs * 1.1
    bb = _lab_bbox(pos[i], eye, fwd, right, up, f, cx, cy, bh)
    x0 = wp.max(int(bb[0]) - 1, 0)
    x1 = wp.min(int(bb[2]) + 1, w - 1)
    y0 = wp.max(int(bb[1]) - 1, 0)
    y1 = wp.min(int(bb[3]) + 1, h - 1)
    if x1 < x0 or y1 < y0 or x1 - x0 > maxb or y1 - y0 > maxb:
        return
    d_amt = dmgc[i]
    hh = wp.clamp(heat[i], 0.0, 1.0)
    reach = float(0.0)
    if hover_r > 0.0:
        hd = wp.length(pos[i] - hover)
        if hd < hover_r:
            reach = (1.0 - hd / hover_r) * 0.45 + 0.1
    o = eye - pos[i]
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            d = fwd + right * ((float(px) + 0.5 - cx) / f) - up * ((float(py) + 0.5 - cy) / f)
            r = wp.vec4(-1.0, 0.0, 0.0, 0.0)
            if rnd:
                r = _lab_raysphere(o, d, bh)
            else:
                r = _lab_raybox(o, d, hs)
            if r[0] > 0.0 and depth[py * w + px] == int(r[0] * 4000.0):
                n = wp.vec3(r[1], r[2], r[3])
                q = (o + d * r[0]) / hs                       # where on the cube we hit, -1 .. 1 on each axis
                edge = wp.max(wp.max(wp.abs(q[0]) * (1.0 - wp.abs(n[0])), wp.abs(q[1]) * (1.0 - wp.abs(n[1]))),
                              wp.abs(q[2]) * (1.0 - wp.abs(n[2])))
                if rnd:
                    edge = 0.0                                # a liquid or a ball has no cube edges
                lam = 0.30 + 0.70 * wp.max(0.0, wp.dot(n, light))
                c = base[i] * lam
                if vmode > 0:
                    c = vcol[i] * (0.55 + 0.45 * lam)          # false colour: keep the face shading so the cubes still read
                    if edge > 0.86:
                        c = c * 0.80
                else:
                    if edge > 0.86:
                        c = c * 0.80
                    c = c * (1.0 - 0.55 * d_amt) + wp.vec3(0.22, 0.03, 0.0) * d_amt
                    c = c * (1.0 - 0.6 * hh) + wp.vec3(1.0, 0.35 + 0.45 * hh, 0.08 + 0.4 * hh * hh) * (hh * 1.3)    # glows as it heats
                    if molten[i] == 1:
                        hot = wp.clamp((hh - 0.35) / 0.65, 0.0, 1.0)
                        lava = wp.vec3(0.55, 0.07, 0.02) * (1.0 - hot) + wp.vec3(1.0, 0.82, 0.30) * hot
                        c = lava * (0.55 + 0.45 * lam)
                c = c * (1.0 - reach) + hover_col * reach
                col[py * w + px] = c
                ids[py * w + px] = i


# ---- picture quality: supersampling, FXAA and a contrast-adaptive sharpener (all on the GPU, after the picture is drawn) ----
@wp.func
def _lab_at(src: wp.array(dtype=wp.vec3), w: int, h: int, x: int, y: int):
    return src[wp.clamp(y, 0, h - 1) * w + wp.clamp(x, 0, w - 1)]


@wp.func
def _lab_luma(c: wp.vec3):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


@wp.kernel
def k_lab_downsample(w: int, ss: int, W: int, src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    """Supersampling: the picture was drawn ss times larger in each direction; every output pixel is the average of its
    ss x ss block of samples."""
    t = wp.tid()
    y = t / w
    x = t - y * w
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(ss):
        for i in range(ss):
            acc = acc + src[(y * ss + j) * W + x * ss + i]
    dst[t] = acc / float(ss * ss)


@wp.kernel
def k_lab_fxaa(w: int, h: int, src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    """FXAA, in the spirit of version 3.11: pixels on a high-contrast edge are blended with their neighbours.
      1. skip flat areas (local luma range < max(1/32, 1/8 of the brightest neighbour));
      2. a sub-pixel blend toward the 3x3 average, by how far the centre is from its neighbours' average;
      3. the edge's direction (horizontal or vertical, from second differences of luma) and its steeper side;
         the pixel is averaged with its neighbour across that side, which smooths 1-pixel stair-steps."""
    t = wp.tid()
    y = t / w
    x = t - y * w
    cm = src[t]
    cn = _lab_at(src, w, h, x, y - 1)
    cs = _lab_at(src, w, h, x, y + 1)
    ce = _lab_at(src, w, h, x + 1, y)
    cw = _lab_at(src, w, h, x - 1, y)
    lm = _lab_luma(cm)
    ln = _lab_luma(cn)
    ls = _lab_luma(cs)
    le = _lab_luma(ce)
    lw = _lab_luma(cw)
    mn = wp.min(lm, wp.min(wp.min(ln, ls), wp.min(le, lw)))
    mx = wp.max(lm, wp.max(wp.max(ln, ls), wp.max(le, lw)))
    rng = mx - mn
    if rng < wp.max(0.0312, mx * 0.125):
        dst[t] = cm
        return
    cnw = _lab_at(src, w, h, x - 1, y - 1)
    cne = _lab_at(src, w, h, x + 1, y - 1)
    csw = _lab_at(src, w, h, x - 1, y + 1)
    cse = _lab_at(src, w, h, x + 1, y + 1)
    lnw = _lab_luma(cnw)
    lne = _lab_luma(cne)
    lsw = _lab_luma(csw)
    lse = _lab_luma(cse)
    avg = (cm + cn + cs + ce + cw + cnw + cne + csw + cse) / 9.0
    lavg = (ln + ls + le + lw) * 0.25
    blend = wp.clamp(wp.abs(lavg - lm) / rng, 0.0, 1.0)
    blend = blend * blend * 0.75
    edge_h = wp.abs(ln + ls - 2.0 * lm) * 2.0 + wp.abs(lne + lse - 2.0 * le) + wp.abs(lnw + lsw - 2.0 * lw)
    edge_v = wp.abs(le + lw - 2.0 * lm) * 2.0 + wp.abs(lne + lnw - 2.0 * ln) + wp.abs(lse + lsw - 2.0 * ls)
    nb = cm
    if edge_h >= edge_v:
        if wp.abs(ln - lm) >= wp.abs(ls - lm):
            nb = cn
        else:
            nb = cs
    else:
        if wp.abs(lw - lm) >= wp.abs(le - lm):
            nb = cw
        else:
            nb = ce
    edge = (cm + nb) * 0.5
    dst[t] = edge * (1.0 - blend) + avg * blend


@wp.kernel
def k_lab_cas(w: int, h: int, sharp: float, src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    """Contrast-adaptive sharpening (the idea of AMD's CAS): a 5-tap cross; the sharpening weight shrinks where the local
    contrast is already high, so edges do not grow halos.   w = -amp / (8 - 3 sharp),   out = (w (N+S+E+W) + C) / (1 + 4 w)."""
    t = wp.tid()
    y = t / w
    x = t - y * w
    c = src[t]
    n = _lab_at(src, w, h, x, y - 1)
    s = _lab_at(src, w, h, x, y + 1)
    e = _lab_at(src, w, h, x + 1, y)
    ww = _lab_at(src, w, h, x - 1, y)
    out = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        mn = wp.min(c[k], wp.min(wp.min(n[k], s[k]), wp.min(e[k], ww[k])))
        mx = wp.max(c[k], wp.max(wp.max(n[k], s[k]), wp.max(e[k], ww[k])))
        amp = wp.sqrt(wp.clamp(wp.min(mn, 1.0 - mx) / wp.max(mx, 1.0e-4), 0.0, 1.0))
        wgt = -amp / (8.0 - 3.0 * sharp)
        out[k] = (wgt * (n[k] + s[k] + e[k] + ww[k]) + c[k]) / (1.0 + 4.0 * wgt)
    dst[t] = out


@wp.kernel
def k_lab_u8(col: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.uint8)):
    t = wp.tid()
    c = col[t]
    out[t * 3] = wp.uint8(wp.clamp(c[0], 0.0, 1.0) * 255.0)
    out[t * 3 + 1] = wp.uint8(wp.clamp(c[1], 0.0, 1.0) * 255.0)
    out[t * 3 + 2] = wp.uint8(wp.clamp(c[2], 0.0, 1.0) * 255.0)


@wp.kernel
def k_lab_u8_flip(w: int, h: int, col: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.uint8)):
    """Same as k_lab_u8 but with the rows upside down: OpenGL textures start at the bottom row.  Used when the picture
    is written straight into an OpenGL buffer (see GLView)."""
    t = wp.tid()
    y = t / w
    x = t - y * w
    o = ((h - 1 - y) * w + x) * 3
    c = col[t]
    out[o] = wp.uint8(wp.clamp(c[0], 0.0, 1.0) * 255.0)
    out[o + 1] = wp.uint8(wp.clamp(c[1], 0.0, 1.0) * 255.0)
    out[o + 2] = wp.uint8(wp.clamp(c[2], 0.0, 1.0) * 255.0)


@wp.kernel
def k_lab_clock(dt: float, P: wp.array(dtype=float)):
    """End of a substep: advance the simulation clock held in P (so a recorded graph needs no per-step argument)."""
    P[LP_T] = P[LP_T] + dt


@wp.kernel
def k_lab_pick(px: int, py: int, w: int, n_obj: int, eye: wp.vec3, dirn: wp.vec3, ids: wp.array(dtype=int),
               pos: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    """Which chunk of the object is under pixel (px, py)?  Reads the id buffer the last picture wrote.
    out = (hit?, x, y, z): the point on the view ray level with that chunk's centre.  Cannonballs are not targets."""
    i = ids[py * w + px]
    out[0] = 0.0
    if i < 0 or i >= n_obj:
        return
    t = wp.dot(pos[i] - eye, dirn)
    if t < 0.1:
        return
    p = eye + dirn * t
    out[0] = 1.0
    out[1] = p[0]
    out[2] = p[1]
    out[3] = p[2]


@wp.kernel
def k_lab_count(n_obj: int, nbr: wp.array(dtype=int), alive: wp.array(dtype=int), gone: wp.array(dtype=int),
                heat: wp.array(dtype=float), cnt: wp.array(dtype=int), hmax: wp.array(dtype=float),
                molten: wp.array(dtype=int)):
    """Readout counters: cnt[0] = bond ends that existed and are now broken, cnt[1] = chunks burned away,
    cnt[2] = liquid chunks, hmax[0] = the hottest chunk.  Only chunks that have something to report touch the shared counters."""
    i = wp.tid()
    if i >= n_obj:
        return
    dead = int(0)
    for k in range(18):
        e = i * 18 + k
        if nbr[e] >= 0 and alive[e] == 0:
            dead += 1
    if dead > 0:
        wp.atomic_add(cnt, 0, dead)
    if gone[i] == 1:
        wp.atomic_add(cnt, 1, 1)
    if molten[i] == 1 and gone[i] == 0:
        wp.atomic_add(cnt, 2, 1)
    if heat[i] > 0.0:
        wp.atomic_max(hmax, 0, heat[i])


@wp.kernel
def k_cc_init(label: wp.array(dtype=int)):
    i = wp.tid()
    label[i] = i


@wp.kernel
def k_cc_min(nbr: wp.array(dtype=int), alive: wp.array(dtype=int), gone: wp.array(dtype=int),
             label: wp.array(dtype=int), label2: wp.array(dtype=int), changed: wp.array(dtype=int)):
    """Connected pieces, step 1: every chunk takes the smallest label among itself and its live-bonded neighbours."""
    i = wp.tid()
    l = label[i]
    m = l
    if gone[i] == 0:
        for k in range(18):
            e = i * 18 + k
            if alive[e] == 1:
                lj = label[nbr[e]]
                if lj < m:
                    m = lj
    if m < l:
        changed[0] = 1
    label2[i] = m


@wp.kernel
def k_cc_jump(label2: wp.array(dtype=int), label: wp.array(dtype=int)):
    """Step 2 (pointer jumping): follow the label one more hop, so labels cross a whole piece in few rounds."""
    i = wp.tid()
    label[i] = label2[label2[i]]


@wp.kernel
def k_cc_hist(label: wp.array(dtype=int), gone: wp.array(dtype=int), sizes: wp.array(dtype=int)):
    i = wp.tid()
    if gone[i] == 0:
        wp.atomic_add(sizes, label[i], 1)


@wp.kernel
def k_cc_summary(sizes: wp.array(dtype=int), out: wp.array(dtype=int)):
    """out = (pieces of 2 or more chunks, single loose chunks, size of the biggest piece)."""
    i = wp.tid()
    s = sizes[i]
    if s >= 2:
        wp.atomic_add(out, 0, 1)
    elif s == 1:
        wp.atomic_add(out, 1, 1)
    if s > 0:
        wp.atomic_max(out, 2, s)


@wp.kernel
def k_lab_wind(coef: float, P: wp.array(dtype=float), nbr: wp.array(dtype=int), alive: wp.array(dtype=int),
               vel: wp.array(dtype=wp.vec3), force: wp.array(dtype=wp.vec3), gone: wp.array(dtype=int)):
    """Wind drag on every chunk that has an exposed side:  u = wind - v ;  F = 1/2 rho Cd A |u| u   (coef = that constant).
    The wind gusts a little, differently for each chunk.  Does nothing while the wind switch in P is off."""
    i = wp.tid()
    if P[LP_WIND_ON] < 0.5 or gone[i] == 1:
        return
    wv = wp.vec3(P[LP_WX], P[LP_WY], P[LP_WZ])
    t = P[LP_T]
    live = int(0)
    for k in range(18):
        if alive[i * 18 + k] == 1:
            live += 1
    if live >= 18:
        return
    u = wv * (1.0 + 0.25 * wp.sin(2.5 * t + float(i) * 0.37)) - vel[i]
    force[i] = force[i] + u * (wp.length(u) * coef)


@wp.kernel
def k_lab_well(soft: float, P: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
               force: wp.array(dtype=wp.vec3), gone: wp.array(dtype=int)):
    """Newton's gravitation toward a point:  F = G M m / (r^2 + soft^2), pointing at the point.
    Does nothing while the well switch in P is off."""
    i = wp.tid()
    if P[LP_WELL_ON] < 0.5 or gone[i] == 1:
        return
    centre = wp.vec3(P[LP_CX], P[LP_CY], P[LP_CZ])
    gm = P[LP_GM]
    d = centre - pos[i]
    r2 = wp.dot(d, d) + soft * soft
    force[i] = force[i] + d / wp.sqrt(r2) * (gm * mass[i] / r2)


@wp.kernel
def k_lab_grab(P: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
               mass: wp.array(dtype=float), off: wp.array(dtype=wp.vec3), flag: wp.array(dtype=int),
               force: wp.array(dtype=wp.vec3), gone: wp.array(dtype=int)):
    """The grab tools: every held chunk is pulled by a damped spring toward (cursor point + its own offset), so the held
    patch moves as one and drags the rest along through the bonds.   a = k (target - x) - c v ,  c = 1.6 sqrt(k)."""
    i = wp.tid()
    if P[LP_GRAB_ON] < 0.5 or flag[i] == 0 or gone[i] == 1:
        return
    tgt = wp.vec3(P[LP_GX], P[LP_GY], P[LP_GZ]) + off[i]
    k = P[LP_GAIN]
    force[i] = force[i] + mass[i] * ((tgt - pos[i]) * k - vel[i] * (1.6 * wp.sqrt(k)))


@wp.kernel
def k_lab_laser(origin: wp.vec3, dirn: wp.vec3, t0: float, reach: float, radius: float, dose: float,
                n_obj: int, pos: wp.array(dtype=wp.vec3), heat: wp.array(dtype=float), gone: wp.array(dtype=int)):
    """A laser beam: heats the chunks inside the beam that lie within `reach` of the first surface it hits.
    heat += power * dt * (1 - d/R)   (dose = power * dt)."""
    i = wp.tid()
    if i >= n_obj or gone[i] == 1:
        return
    rel = pos[i] - origin
    t = wp.dot(rel, dirn)
    if t < t0 - 0.06 or t > t0 + reach:
        return
    perp = wp.length(rel - dirn * t)
    if perp < radius:
        heat[i] = heat[i] + dose * (1.0 - perp / radius)


@wp.kernel
def k_lab_heat_a(dt: float, cond: float, cool: float, burn: wp.array(dtype=float), nbr: wp.array(dtype=int),
                 alive: wp.array(dtype=int), heat: wp.array(dtype=float), heat2: wp.array(dtype=float),
                 gone: wp.array(dtype=int), molten: wp.array(dtype=int), latent: float):
    """Heat conducts to the neighbours, cools, and (for wood that is alight) feeds itself:
    h += dt * cond * (mean neighbour h - h) ;  h -= dt * cool * h ;  h += dt * burn  while h > 0.12.
    A liquid chunk has no bonds, so it does not conduct, and it cools more slowly:  cool * latent."""
    i = wp.tid()
    if gone[i] == 1:
        heat2[i] = 0.0
        return
    h = heat[i]
    tot = float(0.0)
    cnt = int(0)
    for k in range(18):
        e = i * 18 + k
        if alive[e] == 1:
            tot += heat[nbr[e]]
            cnt += 1
    if cnt > 0:
        h = h + dt * cond * (tot / float(cnt) - h)
    c = cool
    if molten[i] == 1:
        c = cool * latent
    h = h - dt * c * h
    if burn[i] > 0.0 and h > 0.12:
        h = h + dt * burn[i]
    heat2[i] = wp.max(h, 0.0)


@wp.kernel
def k_lab_heat_b(dt: float, rate: float, nbr: wp.array(dtype=int), peer: wp.array(dtype=int),
                 alive: wp.array(dtype=int), heat2: wp.array(dtype=float), heat: wp.array(dtype=float),
                 dmg: wp.array(dtype=float), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                 gone: wp.array(dtype=int), park: wp.vec3, meltc: wp.array(dtype=float), freeze_k: float,
                 molten: wp.array(dtype=int), jit: float):
    """Heat does its damage: a bond wears by  rate dt ((h_a - 0.4)+ + (h_b - 0.4)+)  from the heat of the chunks at both
    its ends.  Then the chunk's fate:
      a material that melts (melt > 0):  at h >= melt the chunk turns to liquid: all its bonds are cut (both ends) and it
          flows by the particle rules (k_lab_melt_*);  it sets again (a loose solid chunk) when h < freeze;
      any other material:  at h >= 1.0 the chunk burns away (its bonds are cut and it is removed).
    Each chunk updates its own copy of each bond's damage (see k_lab_bonds).  The sum is the same number at both ends
    (a + b == b + a exactly), so the two copies stay identical without atomics."""
    i = wp.tid()
    h = heat2[i]
    heat[i] = h
    if gone[i] == 1:
        return
    melt = meltc[i]                                  # this chunk's own material (0 = does not melt)
    freeze = melt * freeze_k
    wi = wp.max(h - 0.4, 0.0)
    for k in range(18):
        e = i * 18 + k
        if alive[e] == 1:
            x = rate * dt * (wi + wp.max(heat2[nbr[e]] - 0.4, 0.0))
            if x > 0.0:
                dmg[e] = dmg[e] + x
    if melt > 0.0:
        if molten[i] == 0 and h >= melt:
            molten[i] = 1
            for k in range(18):
                e = i * 18 + k
                if alive[e] == 1:
                    alive[e] = 0
                    alive[peer[e]] = 0
            # a perfect lattice of spheres is a stable stack, which no liquid is: nudge the chunk off its lattice site
            # (a fixed pseudo-random offset of `jit`) so that columns can slide apart
            rs = wp.rand_init(7919, i)
            pos[i] = pos[i] + wp.vec3(wp.randf(rs) - 0.5, 0.0, wp.randf(rs) - 0.5) * jit
        elif molten[i] == 1 and h < freeze:
            molten[i] = 0
        return
    if h >= 1.0:
        for k in range(18):
            e = i * 18 + k
            if alive[e] == 1:
                alive[e] = 0
                alive[peer[e]] = 0
        gone[i] = 1
        pos[i] = park
        vel[i] = wp.vec3(0.0, 0.0, 0.0)


class LabCamera:
    """Orbit camera: yaw / pitch around a target point at a distance."""

    def __init__(self):
        self.yaw, self.pitch, self.dist = 0.7, 0.35, 3.2
        self.target = np.array([0.0, 0.6, 0.0])

    def basis(self):
        cp, sp, cy, sy = math.cos(self.pitch), math.sin(self.pitch), math.cos(self.yaw), math.sin(self.yaw)
        eye = self.target + self.dist * np.array([sy * cp, sp, cy * cp])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def project(self, p, w, h, f):
        """Screen position of a world point, or None if it is behind the camera."""
        eye, fwd, right, up = self.basis()
        rel = np.asarray(p, float) - eye
        z = float(rel @ fwd)
        if z < 0.1:
            return None
        return w / 2.0 + f * float(rel @ right) / z, h / 2.0 - f * float(rel @ up) / z

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.008
        self.pitch = max(-0.1, min(1.45, self.pitch + dy * 0.008))

    def zoom(self, notches):
        self.dist = max(0.6, min(14.0, self.dist * (0.9 ** notches)))


LAB_SS_MAX_PIXELS = 16_000_000          # supersampling is lowered if the internal picture would have more pixels than this
LAB_BALLS = 10                        # cannonballs that can be in flight at once (the oldest is reused)
LAB_BALL_R = 0.16                     # a cannonball is a 0.16 m radius ball of chunks
LAB_SHOT_CAP = 120.0                  # fastest a cannonball or bullet may fly (m/s): faster than ~1 chunk per step tunnels straight through; the object's own chunks stop at 6
LAB_BULLETS = 24                      # machine-gun bullets that can be in flight at once (the oldest is reused)
LAB_BULLET_R = 0.055                  # a bullet is a 5.5 cm radius ball of chunks: one 10 cm chunk at the normal chunk size
LAB_BALL_DENSITY = 7800.0             # iron
WIND_BOOST = 10.0                     # air is this many times denser than real, so a gale is visible at game scale
LAB_QUAKE_HZ = 2.5


def lab_ball_mask(chunk, radius=LAB_BALL_R):
    """The chunks of a cannonball (a ball of fixed physical size, so finer chunks make a rounder ball)."""
    rc = radius / chunk
    n = 2 * int(math.ceil(rc)) + 1
    c = n // 2
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    return ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) <= (rc + 0.01) ** 2


class DestructionLab:
    """The simulation and its renderer.  build() places an object; step() advances; hit() / laser() / fire() / weaken()
    act on it; wind, quake and well are standing conditions; render() returns the picture as PPM bytes."""

    def __init__(self, device=None, width=760, height=520):
        wp.init()
        self.device = wp.get_device(device) if device else wp.get_preferred_device()
        self.w, self.h = width, height
        d = self.device
        self.aa, self.ss_req, self.sharp = 1, 1, 0.3                      # FXAA on, no supersampling, a little sharpening
        self._alloc_view(width, height)
        # The substep loop is recorded once as a CUDA graph per substep count (and re-recorded after build()).
        self.graphs = {}
        try:
            self.use_graph = bool(d.is_cuda and wp.is_mempool_enabled(d))
        except Exception:
            self.use_graph = False
        self.s, self.dt, self.substeps, self.mref = LAB_CHUNK, LAB_DT, LAB_SUBSTEPS, LAB_REF_DENSITY * LAB_CHUNK ** 3
        self.melt_lo, self.any_ductile, self.bond_hard = 0.0, False, 0.0
        self.stable_steps, self.soft = LAB_SUBSTEPS, 1.0
        self.hold_gravity = False          # gravity is on as soon as an object is placed (touch() would switch it on if it were ever held off)
        self.gravity_on, self.g_t0 = True, 0.0
        self.park = np.array([6432.5 * LAB_CHUNK] * 3, np.float32)
        self.rest = wp.array([LAB_CHUNK * math.sqrt(sum(abs(c) for c in o)) for o in LAB_OFFS], dtype=float, device=d)
        self.kscale = wp.array([1.0 if sum(abs(c) for c in o) == 1 else 0.5 for o in LAB_OFFS], dtype=float, device=d)
        self.grid = wp.HashGrid(LAB_GRID_DIM, LAB_GRID_DIM, LAB_GRID_DIM, device=d)
        self.nbr_host = None
        self.last_comp = 0.0
        self._sx = self._cc = self._sx_done = None
        self.async_stats = False
        self.inflight = deque()                                      # events of the frames the GPU has not finished yet
        self.n = 0
        self.n_obj = 0
        self.mat = LAB_MATERIALS["Stone"]
        self.rng = np.random.default_rng(1)
        self.sim_time = 0.0
        self.wind = (0.0, 0.0, 0.0)        # wind velocity in m/s (world axes)
        self.quake_amp = 0.0               # ground shaking, m/s^2
        self.well = None                   # (point, GM) while a gravity well is held
        self.ball_next = 0
        self.bullet_next = 0
        self.melt_on = False               # True while something is hot enough to matter: then each substep also runs the liquid solver
        # CUDA-event stopwatches (GpuTimer, defined with the particles): the GPU time really spent on the physics
        # kernels of a frame and on the picture.  Read later with take(), so measuring never forces a synchronisation.
        self.t_phys = GpuTimer(d)
        self.t_render = GpuTimer(d)

    # ---- placing an object ---------------------------------------------------------------------------------
    @staticmethod
    def _fit_ss(width, height, ss):
        """The biggest supersampling factor <= ss whose internal picture stays under LAB_SS_MAX_PIXELS."""
        while ss > 1 and width * height * ss * ss > LAB_SS_MAX_PIXELS:
            ss -= 1 if ss <= 2 else 2
        return max(1, ss)

    def _alloc_view(self, width, height):
        """Everything whose size follows the picture: colour, depth, id buffer, output bytes and the PPM buffer."""
        d = self.device
        self.w, self.h = width, height
        self.f = 0.95 * height
        self.ss = self._fit_ss(width, height, self.ss_req)                # supersampling factor in use (1 = off)
        W, H = width * self.ss, height * self.ss                          # the internal picture
        self.W, self.H = W, H
        self.col = wp.zeros(W * H, dtype=wp.vec3, device=d)
        self.depth = wp.zeros(W * H, dtype=int, device=d)
        self.col_a = wp.zeros(width * height, dtype=wp.vec3, device=d)    # the picture at screen size, and a spare for the filters
        self.col_b = wp.zeros(width * height, dtype=wp.vec3, device=d)
        self.out = wp.zeros(width * height * 3, dtype=wp.uint8, device=d)
        self.ids = wp.full(W * H, -1, dtype=int, device=d)                # which chunk each pixel shows (for picking)
        self.pick_out = wp.zeros(4, dtype=float, device=d)
        self.P = wp.zeros(LP_N, dtype=float, device=d)                   # per-frame parameters the kernels read (see LP_*)
        self.host_P = np.zeros(LP_N, np.float32)
        # The finished picture is copied straight into one PPM buffer (header + pixels): no per-frame allocation,
        # no .numpy() copy and no header concatenation.  h_img is a Warp array that aliases the pixel part of it.
        head = b"P6\n%d %d\n255\n" % (width, height)
        self.ppm = bytearray(len(head) + width * height * 3)
        self.ppm[:len(head)] = head
        self.ppm_px = np.frombuffer(self.ppm, dtype=np.uint8, offset=len(head))
        try:
            self.h_img = wp.array(self.ppm_px, dtype=wp.uint8, device="cpu", copy=False)
        except Exception:
            self.h_img = None                                            # older Warp: render() falls back to one numpy copy

    def set_quality(self, aa=None, ss=None, sharp=None):
        """Picture quality: aa 0/1 (FXAA), ss = supersampling factor 1, 2 or 4, sharp 0..1.  Returns the supersampling factor
        really in use (it is lowered when the internal picture would be too big)."""
        if aa is not None:
            self.aa = int(aa)
        if sharp is not None:
            self.sharp = float(max(0.0, min(1.0, sharp)))
        if ss is not None and int(ss) != self.ss_req:
            self.ss_req = int(ss)
            self._alloc_view(self.w, self.h)
            wp.synchronize_device(self.device)
        return self.ss

    def resize(self, width, height):
        """Change the picture size (the simulation is untouched)."""
        width, height = max(64, int(width)), max(64, int(height))
        if (width, height) != (self.w, self.h):
            self._alloc_view(width, height)
            wp.synchronize_device(self.device)

    def _lattice(self, mask, noise):
        """Chunk coordinates and bond tables for a boolean grid: (coords, nbr, alive, peer, strength per bond)."""
        coords = np.argwhere(mask)
        N = len(coords)
        idx = -np.ones(mask.shape, int)
        idx[mask] = np.arange(N)
        nbr = -np.ones((N, LAB_NB), np.int32)
        for k, o in enumerate(LAB_OFFS):
            c2 = coords + np.array(o)
            ok = np.all((c2 >= 0) & (c2 < np.array(mask.shape)), axis=1)
            j = -np.ones(N, int)
            j[ok] = idx[c2[ok, 0], c2[ok, 1], c2[ok, 2]]
            nbr[:, k] = j
        alive = (nbr >= 0).astype(np.int32)
        peer = np.where(nbr >= 0, nbr * LAB_NB + np.array(LAB_REV)[None, :], 0).astype(np.int32)
        if noise is None:
            strength = np.full((N, LAB_NB), 1000.0, np.float32)          # unbreakable
        else:
            strength = np.clip(self.rng.normal(1.0, noise, (N, LAB_NB)), 0.35, 2.0).astype(np.float32)
            flat, pflat = strength.reshape(-1), peer.reshape(-1)
            owner = np.repeat(np.arange(N), LAB_NB)
            low = (nbr.reshape(-1) >= 0) & (nbr.reshape(-1) < owner)
            flat[low] = flat[pflat[low]]                                  # both ends of a bond share one strength
        return coords, nbr, alive, peer, strength

    def build(self, shape="Block", material="Stone", size="Medium"):
        d = self.device
        n_edge = int(size) if isinstance(size, (int, np.integer)) else LAB_SIZES[size]
        mask = lab_shape_mask(shape, n_edge)
        # one material per chunk: the whole object is the chosen material.
        # self.mat is the main one (most chunks): it sets the settings that are still the same for the whole object
        # (liquid thickness, damage rate, the damping and step sizes).  Everything that decides how it breaks, how
        # heavy it is, how it grips the ground, burns or melts is read per chunk or per bond below.
        names = tuple(LAB_MATERIALS)
        M = [LAB_MATERIALS[n] for n in names]
        ids_grid = np.full(mask.shape, names.index(material), np.int64)
        mid = ids_grid[mask]                                         # same order as np.argwhere(mask)
        mat = M[int(np.bincount(mid, minlength=len(names)).argmax())]
        self.mat = mat
        used = np.unique(mid)
        tab = lambda f: np.array([f(m) for m in M], np.float32)
        t_dens, t_k, t_eb, t_ey, t_ep = tab(lambda m: m.density), tab(lambda m: m.k), tab(lambda m: m.eb), tab(lambda m: m.ey), tab(lambda m: m.ductile)
        t_bur, t_mel, t_gf, t_gs, t_noi = tab(lambda m: m.burn), tab(lambda m: m.melt), tab(lambda m: m.gfric), tab(lambda m: m.gstick), tab(lambda m: m.noise)
        t_col = np.array([m.colour for m in M], np.float32)
        t_hard = tab(lambda m: m.hard)
        melts = t_mel[used]
        self.melt_lo = float(melts[melts > 0].min()) if (melts > 0).any() else 0.0     # the lowest melting point present (0 = nothing melts)
        self.any_ductile = bool((t_ep[used] > 0).any())
        self.bond_hard = float(t_hard[used][t_ep[used] > 0].max()) if self.any_ductile else float(mat.hard)
        s = LAB_CHUNK * LAB_BASE_N / max(LAB_BASE_N, n_edge)           # finer detail: same object, smaller chunks
        r = s / LAB_CHUNK
        self.s = s
        self.dt = LAB_DT * r                                          # stability: the step shrinks with the chunk
        self.substeps = int(math.ceil(LAB_SUBSTEPS / r - 1e-6))
        ratio = float(max(t_k[u] / t_dens[u] for u in used)) / (3.0e5 / 2500.0)    # bond frequency squared (stiffest material present), relative to stone
        self.stable_steps = int(math.ceil(self.substeps * math.sqrt(max(1.0, ratio / 1.3))))   # the steps per 1/60 s the full-stiffness bonds need
        self.mref = LAB_REF_DENSITY * s ** 3
        self.park = np.array([6432.5 * s] * 3, np.float32)            # a hash-grid cell nothing else uses
        self.rest = wp.array([s * math.sqrt(sum(abs(c) for c in o)) for o in LAB_OFFS], dtype=float, device=d)
        nx, ny, nz = mask.shape
        coords, nbr, alive, peer, strength = self._lattice(mask, t_noi[mid][:, None])
        N = len(coords)
        # per-bond limits from the two chunks a bond joins: it breaks where the weaker one would, it is as stiff as the
        # two in series, and it is ductile (bends) only if both are.  One material: exactly that material's numbers.
        jj = np.maximum(nbr, 0)
        ia, ib = mid[:, None], mid[jj]
        ep_a, ep_b = t_ep[ia], t_ep[ib]
        bp_o = np.stack([np.minimum(t_eb[ia], t_eb[ib]), np.minimum(t_ey[ia], t_ey[ib]),
                         np.where((ep_a > 0) & (ep_b > 0), np.minimum(ep_a, ep_b), 0.0),
                         (2.0 * t_k[ia] * t_k[ib] / (t_k[ia] + t_k[ib])) / mat.k], axis=-1).astype(np.float32)   # (N, 18, 4)
        pos = np.zeros((N, 3), np.float32)
        pos[:, 0] = (coords[:, 0] - (nx - 1) / 2.0) * s
        pos[:, 1] = (coords[:, 1] - coords[:, 1].min() + 0.5) * s     # lowest layer rests on the ground
        pos[:, 2] = (coords[:, 2] - (nz - 1) / 2.0) * s
        mass = (t_dens[mid] * s ** 3).astype(np.float32)
        base = t_col[mid] * (0.88 + 0.24 * self.rng.random((N, 1))).astype(np.float32)
        gp_o = np.stack([t_gf[mid], t_gs[mid]], axis=1).astype(np.float32)
        # two pools of iron projectiles, parked far away until fired: cannonballs, then smaller machine-gun bullets
        nbr_l, alive_l, peer_l, str_l, pos_l, mass_l, base_l = [nbr], [alive], [peer], [strength], [pos], [mass], [base]
        bp_l, gp_l, burn_l, melt_l = [bp_o], [gp_o], [t_bur[mid]], [t_mel[mid]]
        proj_bp = np.array([mat.eb, mat.ey, mat.ductile, 1.0], np.float32)         # the projectiles' bonds never break (strength 1000)
        off = N
        pools = []
        for count, radius in ((LAB_BALLS, LAB_BALL_R), (LAB_BULLETS, LAB_BULLET_R)):
            bc, bn, ba, bp, bs = self._lattice(lab_ball_mask(s, radius), None)
            K = len(bc)
            rel = ((bc - bc.mean(axis=0)) * s).astype(np.float32)
            pools.append((off, K, rel))
            for q in range(count):
                o = off + q * K
                nbr_l.append(np.where(bn >= 0, bn + o, -1).astype(np.int32))
                alive_l.append(ba.copy())
                peer_l.append(np.where(bn >= 0, bp + o * LAB_NB, 0).astype(np.int32))
                str_l.append(bs.copy())
                pos_l.append(self.park[None, :] + rel)
                mass_l.append(np.full(K, LAB_BALL_DENSITY * s ** 3, np.float32))
                base_l.append(np.array([0.27, 0.29, 0.33], np.float32)[None, :] * (0.9 + 0.2 * self.rng.random((K, 1))).astype(np.float32))
                bp_l.append(np.tile(proj_bp, (K, LAB_NB, 1)))
                gp_l.append(np.tile(np.array([mat.gfric, mat.gstick], np.float32), (K, 1)))
                burn_l.append(np.full(K, mat.burn, np.float32))
                melt_l.append(np.full(K, mat.melt, np.float32))
            off += count * K
        self.ball_k, self.ball_rel = pools[0][1], pools[0][2]
        self.pools = pools                                            # (first index, chunks per projectile, offsets) of each
        Ntot = off
        gone = np.zeros(Ntot, np.int32)
        gone[N:] = 1
        cat = lambda L: np.concatenate(L, axis=0)
        nbr_a, alive_a, peer_a, str_a = cat(nbr_l), cat(alive_l), cat(peer_l), cat(str_l)
        self.n_obj, self.n, self.dims, self.shape_name = N, Ntot, (nx, ny, nz), shape
        self.total_bonds = int(alive_a[:N].sum() // 2)
        mass_a = cat(mass_l)
        self.pos = wp.array(cat(pos_l), dtype=wp.vec3, device=d)
        self.vel = wp.zeros(Ntot, dtype=wp.vec3, device=d)
        self.force = wp.zeros(Ntot, dtype=wp.vec3, device=d)
        self.mass = wp.array(mass_a, dtype=float, device=d)
        self.inv_m = wp.array(1.0 / mass_a, dtype=float, device=d)
        self.nbr = wp.array(nbr_a.reshape(-1), dtype=int, device=d)
        self.peer = wp.array(peer_a.reshape(-1), dtype=int, device=d)
        self.alive = wp.array(alive_a.reshape(-1), dtype=int, device=d)
        self.dmg = wp.zeros(Ntot * LAB_NB, dtype=float, device=d)
        self.plast = wp.zeros(Ntot * LAB_NB, dtype=float, device=d)        # permanent (plastic) strain of each bond end
        self.strength = wp.array(str_a.reshape(-1), dtype=float, device=d)
        self.bprop = wp.array(cat(bp_l).reshape(-1, 4), dtype=wp.vec4, device=d)      # per bond: eb, ey, ep_max, stiffness multiplier
        self.gprop = wp.array(cat(gp_l), dtype=wp.vec2, device=d)                      # per chunk: ground friction, ground stick
        self.burn_c = wp.array(cat(burn_l), dtype=float, device=d)                     # per chunk: heat it makes once alight
        self.melt_c = wp.array(cat(melt_l), dtype=float, device=d)                     # per chunk: melting heat (0 = never)
        self.base = wp.array(cat(base_l), dtype=wp.vec3, device=d)
        self.gone = wp.array(gone, dtype=int, device=d)
        self.heat = wp.zeros(Ntot, dtype=float, device=d)
        self.heat2 = wp.zeros(Ntot, dtype=float, device=d)
        self.molten = wp.zeros(Ntot, dtype=int, device=d)             # 1 = this chunk is liquid (see k_lab_melt_*)
        self.pold = wp.zeros(Ntot, dtype=wp.vec3, device=d)           # a liquid chunk's position before the step's prediction
        self.mcorr = wp.zeros(Ntot, dtype=wp.vec3, device=d)          # the liquid solver's per-chunk correction
        self.nrm = wp.zeros(Ntot, dtype=wp.vec3, device=d)
        self.dmgc = wp.zeros(Ntot, dtype=float, device=d)
        self.full = wp.zeros(Ntot, dtype=int, device=d)             # 1 = all 18 bonds alive (set by k_lab_bonds each step)
        self.vcol = wp.zeros(Ntot, dtype=wp.vec3, device=d)           # per-chunk false colour for the view modes
        self.draw = wp.zeros(Ntot, dtype=int, device=d)
        self.load = wp.zeros(1, dtype=float, device=d)
        self.ke = wp.zeros(1, dtype=float, device=d)
        # GPU-side readout and piece counting
        self.cnt = wp.zeros(3, dtype=int, device=d)                   # broken bond ends, burned chunks, liquid chunks
        self.hmax = wp.zeros(1, dtype=float, device=d)                # hottest chunk
        self.cc_a = wp.zeros(N, dtype=int, device=d)                  # piece labels (and their scratch copy)
        self.cc_b = wp.zeros(N, dtype=int, device=d)
        self.cc_changed = wp.zeros(1, dtype=int, device=d)
        self.cc_sizes = wp.zeros(N, dtype=int, device=d)
        self.cc_out = wp.zeros(3, dtype=int, device=d)                # pieces, loose chunks, biggest piece
        self.ids.fill_(-1)
        self.graphs = {}                                              # recorded graphs point at the old arrays
        self.set_steps(self.stable_steps)                             # bond stiffness, damping and the step length
        self.nbr_host = nbr_a.reshape(-1, LAB_NB)[:N]
        self.last_comp = 0.0
        self.sim_time = 0.0
        self.well = None
        self.grab = None
        self.grab_flag = wp.zeros(Ntot, dtype=int, device=d)          # 1 = held by the grab tools
        self.grab_off = wp.zeros(Ntot, dtype=wp.vec3, device=d)       # each held chunk's offset from the grab point
        self.ball_next = 0
        self.bullet_next = 0
        self.shot_mass = {(b, q): 1.0 for b in (False, True) for q in range(max(LAB_BALLS, LAB_BULLETS))}   # mass multiplier of each projectile slot
        self.gravity_on, self.g_t0 = (not self.hold_gravity), 0.0       # gravity waits for the first interaction
        self.melt_on = False                                          # the liquid passes only run while something is hot
        self.pieces = (1, 0, N, 0)
        self.last_key = None
        # the readout that does not wait for the GPU (stats_request / stats_poll): small pinned host buffers and its state
        pin = d.is_cuda
        self.async_stats = bool(d.is_cuda)
        self.h_ke, self.h_load, self.h_hmax = (wp.zeros(1, dtype=float, device="cpu", pinned=pin) for _ in range(3))
        self.h_cnt = wp.zeros(3, dtype=int, device="cpu", pinned=pin)
        self.h_ccchg = wp.zeros(1, dtype=int, device="cpu", pinned=pin)
        self.h_ccout = wp.zeros(3, dtype=int, device="cpu", pinned=pin)
        self._sx = self._cc = self._sx_done = None
        self.height = (coords[:, 1].max() - coords[:, 1].min() + 1) * s
        return N

    def set_steps(self, n):
        """Physics steps per 1/60 s (5 .. 100).  The step is 1/60 s / n.  Stability needs the step to be shorter than the
        bonds' fastest vibration, which for the chunk size in use takes `stable_steps` of them.  With fewer, the bonds are
        made softer by (n / stable_steps)^2 so the vibration slows to match: faster, but the material gets rubbery.
        With more, they stay at full stiffness and the integration is simply more accurate.  Bond stiffness scales with
        chunk size (the same material), and the damping and contact constants follow from it."""
        n = int(max(LAB_STEPS_MIN, min(LAB_STEPS_MAX, n)))
        r = self.s / LAB_CHUNK
        m = self.mat.density * self.s ** 3
        self.substeps = n
        self.dt = (1.0 / 60.0) / n
        self.soft = min(1.0, (n / float(self.stable_steps)) ** 2)    # stiffness left (1 = full)
        self.kb = self.mat.k * r * self.soft
        self.cdamp = 0.10 * 2.0 * math.sqrt(self.kb * m)
        self.kc = 2.0 * self.kb
        self.cc = 0.30 * 2.0 * math.sqrt(self.kc * m)
        self.graphs = {}                                              # recorded graphs hold the old constants

    # ---- stepping ------------------------------------------------------------------------------------------
    def step(self, substeps=None, want_load=False):
        """Advance one frame (`substeps` physics steps).  The frame's inputs (wind, quake, well, clock) are written to
        the small device array P once; after that the whole substep loop runs on the GPU as one CUDA graph launch.
        want_load: also record the hardest-loaded bond, on the last substep only (it is read by stats())."""
        substeps = substeps or self.substeps
        hp = self.host_P
        windy = self._write_params(want_load)
        self.sim_time += substeps * self.dt
        melt = bool(self.melt_on and self.melt_lo > 0.0)
        self._run_frame(substeps, windy, melt, hp)
        if self.async_stats:                                # (CUDA only) don't let the CPU run frames ahead: that makes the pacing lurch
            ev = wp.Event(self.device)
            wp.record_event(ev)
            self.inflight.append(ev)
            while len(self.inflight) > LAB_MAX_INFLIGHT:
                wp.synchronize_event(self.inflight.popleft())

    def _write_params(self, want_load):
        """Write this frame's inputs (time, gravity, quake, wind, well) into the device array P.  Returns True if windy."""
        hp = self.host_P
        windy = any(abs(x) > 1.0e-6 for x in self.wind)
        hp[LP_T] = self.sim_time
        g = 0.0
        if self.gravity_on:
            g = LAB_G * min(1.0, max(0.0, (self.sim_time - self.g_t0) / LAB_G_RAMP))       # eases in over 0.8 s once on
        hp[LP_QAMP], hp[LP_QHZ], hp[LP_G] = self.quake_amp, LAB_QUAKE_HZ, g
        hp[LP_WX], hp[LP_WY], hp[LP_WZ] = (float(x) for x in self.wind)
        hp[LP_WIND_ON] = 1.0 if windy else 0.0
        if self.well is not None:
            pt, gm = self.well
            hp[LP_WELL_ON], hp[LP_CX], hp[LP_CY], hp[LP_CZ], hp[LP_GM] = 1.0, float(pt[0]), float(pt[1]), float(pt[2]), float(gm)
        else:
            hp[LP_WELL_ON] = 0.0
        if self.grab is not None:
            hp[LP_GRAB_ON], hp[LP_GAIN] = 1.0, float(self.grab["gain"])
            hp[LP_GX], hp[LP_GY], hp[LP_GZ] = (float(x) for x in self.grab["target"])
        else:
            hp[LP_GRAB_ON] = 0.0
        hp[LP_LOAD] = 1.0 if want_load else 0.0
        self.P.assign(hp)
        return windy

    def _run_frame(self, substeps, windy, melt, hp):
        if self.use_graph:
            try:
                g = self.graphs.get((substeps, melt))             # one recorded graph per (steps, liquid solver on/off)
                if g is None:
                    self.grid.build(self.pos, self.s)           # make the grid's internal buffers exist before recording
                    with wp.ScopedCapture(device=self.device) as cap:
                        self._frame(substeps, always=True, melt=melt)
                    g = self.graphs[(substeps, melt)] = cap.graph
                self.t_phys.begin()
                wp.capture_launch(g)
                self.t_phys.end()
                return
            except Exception as e:                              # recording is not possible here: carry on without it
                self.use_graph, self.graphs = False, {}
                print(f"Note: CUDA graph recording failed ({e}); launching the kernels one by one instead.", file=sys.stderr)
                try:
                    wp.synchronize_device(self.device)
                except Exception:
                    pass
                hp[LP_T] = self.sim_time - substeps * self.dt   # the failed attempt may or may not have run: restart this frame
                self.P.assign(hp)
        self.t_phys.begin()
        self._frame(substeps, always=False, windy=windy, well=self.well is not None, melt=melt)
        self.t_phys.end()

    def _frame(self, substeps, always, windy=False, well=False, melt=False):
        """The kernels of one frame.  always=True launches the wind and well kernels every substep (they switch
        themselves off through P), which is what lets the loop be recorded as a graph.  melt=True also runs the liquid
        solver every substep (see k_lab_melt_*); it is left out while nothing is hot, so normal scenes cost the same."""
        d, m, N = self.device, self.mat, self.n
        for sub in range(substeps):
            self._substep(sub == substeps - 1, always, windy, well, melt)
        self._heat_pass(substeps * self.dt)                     # heat moves once per frame

    def _heat_pass(self, dt):
        d, m, N = self.device, self.mat, self.n
        wp.launch(k_lab_heat_a, dim=N, inputs=[dt, 4.0, 1.0, self.burn_c, self.nbr, self.alive, self.heat, self.heat2,
                                               self.gone, self.molten, LAB_LATENT], device=d)
        wp.launch(k_lab_heat_b, dim=N, inputs=[dt, 3.0, self.nbr, self.peer, self.alive, self.heat2, self.heat, self.dmg,
                                               self.pos, self.vel, self.gone, wp.vec3(*[float(x) for x in self.park]),
                                               self.melt_c, float(LAB_FREEZE), self.molten, 0.06 * self.s], device=d)

    def _substep(self, last, always, windy, well, melt=False, mark=None):
        """mark: optional callable mark(name), called after each phase (only by profile(); never while recording a graph)."""
        d, m, N, s, dt0 = self.device, self.mat, self.n, self.s, self.dt
        coef = 0.5 * 1.2 * WIND_BOOST * 1.1 * s ** 2            # 1/2 rho Cd A
        self.grid.build(self.pos, s)
        if mark:
            mark("grid")
        wp.launch(k_lab_bonds, dim=N, inputs=[self.kb, self.cdamp, dt0, m.rate, 1 if last else 0, self.P,
                                              self.pos, self.vel, self.nbr, self.alive, self.dmg, self.strength,
                                              self.rest, self.kscale, self.force, self.gone, self.load, self.full,
                                              self.plast, self.bprop, float(self.bond_hard)], device=d)
        if mark:
            mark("bonds")
        if always or windy:
            wp.launch(k_lab_wind, dim=N, inputs=[coef, self.P, self.nbr, self.alive, self.vel, self.force,
                                                 self.gone], device=d)
        if always or well:
            wp.launch(k_lab_well, dim=N, inputs=[0.2, self.P, self.pos, self.mass, self.force, self.gone], device=d)
        wp.launch(k_lab_grab, dim=N, inputs=[self.P, self.pos, self.vel, self.mass, self.grab_off, self.grab_flag, self.force,
                                             self.gone], device=d)
        if mark:
            mark("wind+well")
        wp.launch(k_lab_contacts, dim=N, inputs=[self.grid.id, s, self.kc, self.cc, 0.6, self.gprop, self.pos,
                                                 self.vel, self.force, self.gone, self.full, self.molten], device=d)
        if mark:
            mark("contacts")
        wp.launch(k_lab_integrate, dim=N, inputs=[dt0, self.P, self.pos, self.vel, self.force, self.inv_m,
                                                  self.gone, self.molten, self.pold, self.n_obj, LAB_SHOT_CAP], device=d)
        if mark:
            mark("integrate")
        if melt:                                                # the particle solver's passes, for liquid chunks only
            n_obj = self.n_obj
            for _ in range(LAB_MELT_ITERS):
                wp.launch(k_lab_melt_project, dim=n_obj, inputs=[self.grid.id, s, self.mat.grip, self.molten, self.gone,
                                                                 self.pos, self.pold, self.mcorr], device=d)
                wp.launch(k_lab_melt_apply, dim=n_obj, inputs=[s, self.mat.grip, LAB_MELT_RELAX, self.molten, self.gone,
                                                               self.pos, self.pold, self.mcorr], device=d)
            wp.launch(k_lab_melt_update, dim=n_obj, inputs=[dt0, self.mat.visc, self.melt_c, float(LAB_FREEZE), self.heat, self.molten, self.gone, self.pos,
                                                              self.pold, self.vel], device=d)
            if mark:
                mark("liquid")
        wp.launch(k_lab_clock, dim=1, inputs=[dt0, self.P], device=d)
        if mark:
            mark("clock")

    def profile(self, steps=8):
        """Time each phase of a physics step on the GPU with CUDA events (the kernels are launched one by one here, not as
        a graph, so every phase can be bracketed).  Runs `steps` real physics steps, so the scene advances a little.
        Returns {phase: ms per 1/60 s frame}, plus 'heat' (once per frame) and 'steps' = the steps in a frame.  The sum is
        a little more than the graph's time, because a graph has no launch gaps."""
        d = self.device
        steps = max(1, min(int(steps), self.substeps))
        windy = self._write_params(False)
        melt = bool(self.melt_on and self.melt_lo > 0.0)
        well = self.well is not None
        marks = []
        cuda = d.is_cuda

        def mark(name):
            if cuda:
                e = wp.Event(d, enable_timing=True)
                wp.record_event(e)
                marks.append((name, e))
            else:
                wp.synchronize_device(d)
                marks.append((name, time.perf_counter()))
        self._substep(False, False, windy, well, melt)               # warm up (first launches are slower)
        wp.synchronize_device(d)
        mark("start")
        mark("overhead")                                             # an empty phase: what one mark costs by itself
        mark("overhead")
        for _ in range(steps):
            self._substep(False, False, windy, well, melt, mark)
        self._heat_pass(steps * self.dt)
        mark("heat")
        wp.synchronize_device(d)
        self.sim_time += (steps + 1) * self.dt
        raw = []
        for (n0, a), (n1, b) in zip(marks[:-1], marks[1:]):
            raw.append((n1, wp.get_event_elapsed_time(a, b) if cuda else (b - a) * 1000.0))
        ov = max((ms for n, ms in raw if n == "overhead"), default=0.0)   # cost of one mark (and its gap)
        out = {}
        for n, ms in raw:
            if n != "overhead":
                out[n] = out.get(n, 0.0) + max(0.0, ms - ov)
        for k in list(out):
            if k != "heat":
                out[k] = out[k] / steps * self.substeps
        out["mark_ms"] = ov
        out["steps"] = self.substeps
        return out

    # ---- acting on the object ------------------------------------------------------------------------------
    @staticmethod
    def _v(a):
        return wp.vec3(float(a[0]), float(a[1]), float(a[2]))

    def touch(self):
        """The first interaction switches gravity on (a new object waits, gravity off, so it stands exactly as placed)."""
        if not self.gravity_on:
            self.gravity_on, self.g_t0 = True, self.sim_time

    def hit(self, point, direction, radius, speed, blast=False):
        """radius in metres.  speed is the velocity change of a stone chunk at the centre of the hit."""
        self.touch()
        wp.launch(k_lab_impulse, dim=self.n_obj, inputs=[1 if blast else 0, self._v(point), self._v(direction),
                                                        float(radius), float(speed), float(self.mref),
                                                        self.pos, self.vel, self.inv_m], device=self.device)

    def weaken(self, point, radius, amount):
        self.touch()
        wp.launch(k_lab_weaken, dim=self.n_obj * LAB_NB, inputs=[self._v(point), float(radius),
                                                                float(amount), self.pos, self.nbr, self.alive,
                                                                self.dmg], device=self.device)

    def _piece_of(self, i0):
        """Indices of every chunk joined to chunk i0 through bonds that still hold."""
        N = self.n_obj
        alive = self.alive.numpy().reshape(-1, LAB_NB)[:N]
        adj = np.where(alive == 1, self.nbr_host[:N], -1)
        seen = np.zeros(N, bool)
        seen[i0] = True
        front = np.array([i0])
        while front.size:
            nb = np.unique(adj[front].ravel())
            nb = nb[nb >= 0]
            nb = nb[~seen[nb]]
            seen[nb] = True
            front = nb
        return np.nonzero(seen)[0]

    def grab_start(self, cam, point, gain, whole=True, radius=0.3):
        """Take hold of the chunk nearest `point`: its whole connected piece (whole=True) or the patch within `radius`."""
        P = self.pos.numpy()[:self.n_obj]
        i0 = int(np.argmin(np.sum((P - np.asarray(point)) ** 2, axis=1)))
        idx = self._piece_of(i0) if whole else np.nonzero(np.sum((P - P[i0]) ** 2, axis=1) < radius * radius)[0]
        flag = np.zeros(self.n, np.int32)
        flag[idx] = 1
        off = np.zeros((self.n, 3), np.float32)
        off[idx] = P[idx] - P[i0]
        self.grab_flag.assign(flag)
        self.grab_off.assign(off)
        eye, fwd, _, _ = cam.basis()
        boost = 1.0 if whole else float(min(12.0, math.sqrt(self.n_obj / max(1, len(idx)))))     # a small patch must be able to haul the rest
        self.grab = {"anchor": P[i0].astype(float), "depth": float(np.dot(P[i0] - eye, fwd)), "gain": float(gain) * boost,
                     "boost": boost, "target": P[i0].astype(float), "n": int(len(idx))}
        self.touch()
        return len(idx)

    def grab_move(self, cam, px, py, gain=None):
        """Point the held patch at the cursor: the spot on the plane through the grab point that faces the camera."""
        if self.grab is None:
            return
        eye, dirn = self.ray(cam, px, py)
        fwd = cam.basis()[1]
        den = float(np.dot(dirn, fwd))
        if den > 1.0e-4:
            self.grab["target"] = eye + dirn * (self.grab["depth"] / den)
        if gain is not None:
            self.grab["gain"] = float(gain) * self.grab["boost"]

    def grab_end(self):
        self.grab = None

    def slice(self, cam, p0, p1):
        """Cut the object along a line dragged on screen from pixel p0 to p1 (see k_lab_slice)."""
        eye, d1 = self.ray(cam, *p0)
        _, d2 = self.ray(cam, *p1)
        nrm = np.cross(d1, d2)
        ln = float(np.linalg.norm(nrm))
        if ln < 1.0e-6:
            return
        nrm /= ln
        det = float(np.dot(np.cross(d1, d2), nrm))
        self.touch()
        wp.launch(k_lab_slice, dim=self.n_obj * LAB_NB, inputs=[self._v(eye), self._v(d1), self._v(d2), self._v(nrm), det,
                                                               self.n_obj, self.pos, self.nbr, self.peer, self.alive],
                  device=self.device)

    def mend(self, point, radius, amount):
        """Repair bonds that still hold (a broken bond does not come back)."""
        self.weaken(point, radius, -amount)

    def laser(self, eye, dirn, point, radius, power, dt):
        """Burn: heat the chunks in the beam near the first surface it meets.  power is heat per second at the centre."""
        self.touch()
        t0 = float(np.dot(np.asarray(point) - np.asarray(eye), dirn))
        if self.melt_lo > 0.0:
            self.melt_on = True                              # heat is going in: run the liquid solver until it has all cooled
        wp.launch(k_lab_laser, dim=self.n, inputs=[self._v(eye), self._v(dirn), t0, 0.15,
                                                   float(radius), float(power * dt), self.n_obj,
                                                   self.pos, self.heat, self.gone], device=self.device)

    def fire(self, cam, target, speed, bullet=False, mass_mult=1.0):
        """Fire the next cannonball (or, with bullet=True, the next small machine-gun bullet) from just in front of the
        camera toward `target`."""
        d = self.device
        self.touch()
        eye, fwd, right, up = cam.basis()
        start = eye + fwd * 0.9 - right * 0.45 - up * 0.3
        dirn = np.asarray(target, float) - start
        dirn /= np.linalg.norm(dirn)
        first, K, rel = self.pools[1 if bullet else 0]
        count = LAB_BULLETS if bullet else LAB_BALLS
        k = self.bullet_next if bullet else self.ball_next
        a = first + (k % count) * K
        if bullet:
            self.bullet_next += 1
        else:
            self.ball_next += 1
        mass_mult = float(mass_mult)
        slot = (bool(bullet), k % count)
        if mass_mult != self.shot_mass[slot]:                 # a heavier / lighter projectile: the damage multiplier
            ms = self.mass.numpy().copy()
            ms[a:a + K] = LAB_BALL_DENSITY * self.s ** 3 * mass_mult
            self.mass.assign(ms)
            self.inv_m.assign(1.0 / ms)
            self.shot_mass[slot] = mass_mult
        pos, vel, gone = self.pos.numpy(), self.vel.numpy(), self.gone.numpy()
        pos[a:a + K] = (start + rel).astype(np.float32)
        vel[a:a + K] = (dirn * speed).astype(np.float32)
        gone[a:a + K] = 0
        wp.copy(self.pos, wp.array(pos, dtype=wp.vec3, device=d))
        wp.copy(self.vel, wp.array(vel, dtype=wp.vec3, device=d))
        wp.copy(self.gone, wp.array(gone, dtype=int, device=d))

    # ---- reading numbers back ------------------------------------------------------------------------------
    def stats(self):
        """Kinetic energy, bonds broken, the hardest-loaded bond (percent of its limit), and the piece counts.
        Everything is summed on the GPU; only a handful of numbers come back.  (This one waits for the GPU; the frame
        loop uses stats_request() / stats_poll() below, which never do.)"""
        d = self.device
        self._sx = self._cc = None                          # a readout in flight is dropped: this one replaces it
        self.ke.zero_()
        self.cnt.zero_()
        self.hmax.zero_()
        wp.launch(k_lab_kinetic, dim=self.n, inputs=[self.mass, self.vel, self.ke], device=d)
        wp.launch(k_lab_count, dim=self.n_obj, inputs=[self.n_obj, self.nbr, self.alive, self.gone, self.heat,
                                                       self.cnt, self.hmax, self.molten], device=d)
        wp.synchronize_device(d)
        ke = float(self.ke.numpy()[0])
        load = float(self.load.numpy()[0]) * 100.0
        self.load.zero_()
        cnt = self.cnt.numpy()
        broken = int(cnt[0]) // 2
        burned = int(cnt[1])
        molten = int(cnt[2])
        hot = float(self.hmax.numpy()[0])
        # the liquid solver runs while any chunk is liquid or hot enough to become so (the laser also switches it on)
        self.melt_on = bool(self.melt_lo > 0.0 and (molten > 0 or hot >= 0.5 * self.melt_lo))
        now = time.perf_counter()
        if (broken, burned) != self.last_key and now >= self.last_comp:
            self.last_key = (broken, burned)
            self.pieces = self._components_gpu(burned)
            self.last_comp = time.perf_counter() + 4.0 * (time.perf_counter() - now)     # never spend more than ~1/5 of the time on it
        return {"ke": ke, "broken": broken, "total": self.total_bonds, "load": load, "pieces": self.pieces[0],
                "loose": self.pieces[1], "biggest": self.pieces[2], "burned": self.pieces[3],
                "heat": hot, "molten": molten}

    # ---- the same readout without waiting for the GPU ------------------------------------------------------
    # The GPU is usually several frames of physics ahead of the CPU.  A readout that synchronises makes the CPU wait for
    # all of it, which is what shows up as a stall every few frames.  Here the kernels are queued behind the physics,
    # their results are copied into pinned memory by the GPU itself, and a CUDA event says when they have landed; the
    # frame loop just asks again next frame.  The piece count (label spreading, many rounds) is spread over frames too.
    def stats_busy(self):
        return self._sx is not None or self._sx_done is not None

    def stats_request(self):
        """Queue a readout (does nothing if one is already on its way).  Fetch it with stats_poll()."""
        if self.stats_busy():
            return
        d = self.device
        if not self.async_stats:
            self._sx_done = self.stats()                    # CPU backend: nothing to overlap, just do it
            return
        self.ke.zero_()
        self.cnt.zero_()
        self.hmax.zero_()
        wp.launch(k_lab_kinetic, dim=self.n, inputs=[self.mass, self.vel, self.ke], device=d)
        wp.launch(k_lab_count, dim=self.n_obj, inputs=[self.n_obj, self.nbr, self.alive, self.gone, self.heat,
                                                       self.cnt, self.hmax, self.molten], device=d)
        wp.copy(self.h_ke, self.ke)
        wp.copy(self.h_load, self.load)
        wp.copy(self.h_cnt, self.cnt)
        wp.copy(self.h_hmax, self.hmax)
        self.load.zero_()                                   # (stream order: after the copy above)
        ev = wp.Event(d)
        wp.record_event(ev)
        self._sx = ev

    def stats_poll(self):
        """Never blocks.  Returns the stats dict once the requested readout has arrived, otherwise None."""
        if not self.async_stats:
            out, self._sx_done = self._sx_done, None
            return out
        self._cc_advance()
        ev = self._sx
        if ev is None or not ev.is_complete:
            return None
        self._sx = None
        ke = float(self.h_ke.numpy()[0])
        load = float(self.h_load.numpy()[0]) * 100.0
        cnt = self.h_cnt.numpy()
        hot = float(self.h_hmax.numpy()[0])
        broken, burned, molten = int(cnt[0]) // 2, int(cnt[1]), int(cnt[2])
        self.melt_on = bool(self.melt_lo > 0.0 and (molten > 0 or hot >= 0.5 * self.melt_lo))
        if self._cc is None and (broken, burned) != self.last_key and time.perf_counter() >= self.last_comp:
            self._cc_start((broken, burned))
        return {"ke": ke, "broken": broken, "total": self.total_bonds, "load": load, "pieces": self.pieces[0],
                "loose": self.pieces[1], "biggest": self.pieces[2], "burned": self.pieces[3],
                "heat": hot, "molten": molten}

    def _cc_batch(self):
        """Queue 8 rounds of label spreading and a copy of 'did anything change'; returns the event for it."""
        d, N = self.device, self.n_obj
        self.cc_changed.zero_()
        for _ in range(8):
            wp.launch(k_cc_min, dim=N, inputs=[self.nbr, self.alive, self.gone, self.cc_a, self.cc_b, self.cc_changed], device=d)
            wp.launch(k_cc_jump, dim=N, inputs=[self.cc_b, self.cc_a], device=d)
        wp.copy(self.h_ccchg, self.cc_changed)
        ev = wp.Event(d)
        wp.record_event(ev)
        return ev

    def _cc_start(self, key):
        wp.launch(k_cc_init, dim=self.n_obj, inputs=[self.cc_a], device=self.device)
        self._cc = {"phase": "iter", "it": 0, "key": key, "ev": self._cc_batch()}

    def _cc_advance(self):
        """Move the piece count along by one step if the GPU has finished the last one (it keeps going on later frames
        until the labels stop changing).  The bonds may change a little while it runs; the next readout then starts again."""
        cc = self._cc
        if cc is None or not cc["ev"].is_complete:
            return
        d, N = self.device, self.n_obj
        if cc["phase"] == "iter":
            cc["it"] += 8
            if int(self.h_ccchg.numpy()[0]) == 0 or cc["it"] >= LAB_CC_ITERS:
                self.cc_sizes.zero_()
                self.cc_out.zero_()
                wp.launch(k_cc_hist, dim=N, inputs=[self.cc_a, self.gone, self.cc_sizes], device=d)
                wp.launch(k_cc_summary, dim=N, inputs=[self.cc_sizes, self.cc_out], device=d)
                wp.copy(self.h_ccout, self.cc_out)
                ev = wp.Event(d)
                wp.record_event(ev)
                cc["phase"], cc["ev"] = "out", ev
            else:
                cc["ev"] = self._cc_batch()
        else:
            o = self.h_ccout.numpy()
            self.pieces = (int(o[0]), int(o[1]), int(o[2]), int(cc["key"][1]))
            self.last_key = cc["key"]
            self.last_comp = time.perf_counter() + 0.3     # at most a few piece counts a second
            self._cc = None

    def _components_gpu(self, burned):
        """Connected groups of chunks over the bonds that still hold, found on the GPU (label propagation with pointer
        jumping).  Returns (pieces of 2+, single chunks, biggest, burned)."""
        d, N = self.device, self.n_obj
        wp.launch(k_cc_init, dim=N, inputs=[self.cc_a], device=d)
        for it in range(LAB_CC_ITERS):
            if it % 4 == 0:
                self.cc_changed.zero_()
            wp.launch(k_cc_min, dim=N, inputs=[self.nbr, self.alive, self.gone, self.cc_a, self.cc_b, self.cc_changed], device=d)
            wp.launch(k_cc_jump, dim=N, inputs=[self.cc_b, self.cc_a], device=d)
            if it % 4 == 3 and int(self.cc_changed.numpy()[0]) == 0:
                break
        self.cc_sizes.zero_()
        self.cc_out.zero_()
        wp.launch(k_cc_hist, dim=N, inputs=[self.cc_a, self.gone, self.cc_sizes], device=d)
        wp.launch(k_cc_summary, dim=N, inputs=[self.cc_sizes, self.cc_out], device=d)
        o = self.cc_out.numpy()
        return (int(o[0]), int(o[1]), int(o[2]), int(burned))

    def _components(self, alive, gone):
        """CPU reference for _components_gpu (unused by the app, kept for checking): (pieces of 2+, single chunks, biggest, burned)."""
        N = self.n_obj
        nbr = self.nbr_host
        a = alive.reshape(-1, LAB_NB)[:N] == 1
        ii, kk = np.nonzero(a)
        jj = nbr[ii, kk]
        keep = jj > ii
        ii, jj = ii[keep], jj[keep]
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            _, label = connected_components(coo_matrix((np.ones(len(ii), np.int8), (ii, jj)), shape=(N, N)), directed=False)
        except ImportError:
            label = np.arange(N)
            for _ in range(400):
                lo = np.minimum(label[ii], label[jj])
                new = label.copy()
                np.minimum.at(new, ii, lo)
                np.minimum.at(new, jj, lo)
                new = new[new]                               # pointer jumping
                if np.array_equal(new, label):
                    break
                label = new
        g = gone[:N] == 1
        sizes = np.bincount(label[~g], minlength=N)
        sizes = sizes[sizes > 0]
        return (int((sizes >= 2).sum()), int((sizes == 1).sum()), int(sizes.max()) if len(sizes) else 0, int(g.sum()))

    # ---- drawing -------------------------------------------------------------------------------------------
    def legend_colours(self, pal, n=24):
        """n colours along ramp `pal` (0 = blue to red, 1 = thermal), read from the same GPU function the picture uses."""
        out = wp.zeros(n, dtype=wp.vec3, device=self.device)
        wp.launch(k_lab_ramp_sample, dim=n, inputs=[int(pal), out], device=self.device)
        return [tuple(float(x) for x in row) for row in out.numpy()]

    def render(self, cam, hover=None, hover_radius=0.0, hover_colour=(0.35, 0.65, 1.0), view=0, target=None):
        """view: 0 Normal, 1 Damage, 2 Strain, 3 Pieces, 4 Velocity, 5 Heat (see LAB_VIEWS).
        target: a uint8 Warp array of w*h*3 bytes (an OpenGL buffer mapped by GLView).  The picture is then written
        straight into it, upside down, with no host copy and no synchronisation, and render() returns None."""
        d, w, h, V = self.device, self.w, self.h, self._v
        W, H, ss = self.W, self.H, self.ss
        eye, fwd, right, up = cam.basis()
        cx, cy, fi, maxb = W / 2.0, H / 2.0, self.f * ss, 160 * ss        # the picture is drawn at W x H (ss times larger)
        light = np.array([0.45, 0.80, 0.40])
        light /= np.linalg.norm(light)
        self.t_render.begin()
        wp.launch(k_lab_surface, dim=self.n, inputs=[self.pos, self.nbr, self.alive, self.dmg, self.nrm, self.dmgc,
                                                     self.draw, self.gone], device=d)
        wp.launch(k_lab_sky, dim=W * H, inputs=[W, H, fi, cx, cy, V(eye), V(fwd), V(right), V(up), self.col,
                                                self.depth, self.ids], device=d)
        wp.launch(k_lab_depth, dim=self.n, inputs=[W, H, fi, cx, cy, self.s, V(eye), V(fwd), V(right), V(up),
                                                   self.pos, self.draw, self.depth, self.molten, self.n_obj, maxb], device=d)
        hv = V(hover) if hover is not None else V((0, 0, 0))
        view = int(view)
        if view > 0:
            wp.launch(k_lab_viewval, dim=self.n, inputs=[view, self.n_obj, self.mat.eb, self.pos, self.vel, self.nbr,
                                                         self.alive, self.dmg, self.strength, self.rest, self.heat,
                                                         self.cc_a, self.cc_sizes, self.vcol,
                                                         self.plast, float(self.mat.ductile)], device=d)
        wp.launch(k_lab_colour, dim=self.n, inputs=[W, H, fi, cx, cy, self.s, V(eye), V(fwd), V(right), V(up),
                                                    V(light), hv, float(hover_radius if hover is not None else 0.0),
                                                    V(hover_colour), self.pos, self.draw, self.base,
                                                    self.dmgc, self.heat, self.depth, self.col, self.ids,
                                                    view, self.vcol, self.molten, self.n_obj, maxb], device=d)
        src = self.col
        if ss > 1:                                               # supersampling: average ss x ss samples per pixel
            wp.launch(k_lab_downsample, dim=w * h, inputs=[w, ss, W, src, self.col_a], device=d)
            src = self.col_a
        spare = self.col_b if src is self.col_a else self.col_a
        if self.aa:                                              # FXAA
            wp.launch(k_lab_fxaa, dim=w * h, inputs=[w, h, src, spare], device=d)
            src, spare = spare, src
        if self.sharp > 0.0:                                     # sharpen
            wp.launch(k_lab_cas, dim=w * h, inputs=[w, h, float(self.sharp), src, spare], device=d)
            src = spare
        if target is not None:
            wp.launch(k_lab_u8_flip, dim=w * h, inputs=[w, h, src, target], device=d)
            self.t_render.end()
            return None
        wp.launch(k_lab_u8, dim=w * h, inputs=[src, self.out], device=d)
        self.t_render.end()
        if self.h_img is not None:
            wp.copy(self.h_img, self.out)                # straight into the PPM buffer: one copy, no temporaries
            wp.synchronize_device(d)
        else:
            self.ppm_px[:] = self.out.numpy()
        return self.ppm

    def ray(self, cam, px, py):
        """The camera's eye and the direction of the ray through screen point (px, py)."""
        eye, fwd, right, up = cam.basis()
        dirn = fwd + right * ((px - self.w / 2.0) / self.f) - up * ((py - self.h / 2.0) / self.f)
        return eye, dirn / np.linalg.norm(dirn)

    def pick(self, cam, px, py):
        """Which surface chunk of the object is under the mouse?  Returns (hit point, ray direction) or None.
        Read from the id buffer the last picture wrote, so the GPU does the search and only 16 bytes come back."""
        px, py = int(px), int(py)
        if not (0 <= px < self.w and 0 <= py < self.h):
            return None
        eye, dirn = self.ray(cam, px, py)
        V = self._v
        ss = self.ss                                            # the id buffer is ss times larger: read the sample at the pixel's centre
        wp.launch(k_lab_pick, dim=1, inputs=[px * ss + ss // 2, py * ss + ss // 2, self.W, self.n_obj, V(eye), V(dirn), self.ids, self.pos,
                                             self.pick_out], device=self.device)
        o = self.pick_out.numpy()
        if o[0] < 0.5:
            return None
        return np.array(o[1:4], dtype=np.float64), dirn


# ----------------------------------------------------------------------------- the lab window
LAB_TOOLS = (("hammer", "Hammer", "A blunt hit along your view direction.", 3.0, 45.0, (0.35, 0.65, 1.0)),
             ("pick", "Pick", "A small, precise hit.", 1.2, 40.0, (0.55, 0.85, 1.0)),
             ("blast", "Blast", "A shockwave pushing outward from the point.", 5.0, 30.0, (1.0, 0.65, 0.25)),
             ("laser", "Laser", "Hold to burn. Heat spreads, cracks the surface and burns it away. Wood catches fire.", 1.0, 45.0, (1.0, 0.30, 0.20)),
             ("cannon", "Cannon", "Fires an iron cannonball at the spot you click.", 0.0, 40.0, (0.70, 0.72, 0.78)),
             ("gun", "Machine gun", "Hold to fire: one small iron bullet every second at the point under the cursor.", 0.0, 50.0, (0.85, 0.80, 0.45)),
             ("well", "Gravity", "Hold to pull a gravity well: every chunk is pulled toward the point (Newton's law of gravitation).", 0.0, 20.0, (0.70, 0.45, 1.0)),
             ("slice", "Slicer", "Drag a line across the object and let go: every bond the line crosses is cut, so it comes apart along it.", 0.0, 0.0, (1.0, 0.85, 0.30)),
             ("grab", "Grab piece", "Hold and drag: picks up the whole connected piece under the cursor and moves it about. Force = how hard it is pulled.", 0.0, 40.0, (0.45, 0.90, 0.60)),
             ("tug", "Tug", "Hold and drag: pulls on a patch of the object (Size of the area); the rest follows through the bonds, so it bends and tears.", 3.0, 40.0, (0.95, 0.60, 0.80)),
             ("weaken", "Weaken", "Paint damage on an area. Nothing is hit.", 3.0, 0.0, (1.0, 0.35, 0.30)),
             ("mend", "Mend", "Repair bonds that still hold.", 3.0, 0.0, (0.35, 0.90, 0.50)))
LAB_EXTRA_DMG = 0.15                  # bond damage per unit of (multiplier - 1), painted over a hit's area
LAB_MULTS = (0.5, 1.0, 2.0, 5.0, 10.0)     # the Damage x buttons (no ceiling: the point includes stress-testing)
LAB_FORCE_LABEL = {"grab": "Pull strength", "tug": "Pull strength", "laser": "Power", "cannon": "Speed (m/s)", "gun": "Speed (m/s)", "well": "Pull (GM)"}
LAB_FPS_CAP = 60.0                    # the GL loop sleeps rather than draw faster than this (vsync may hold it lower)
LAB_MAX_CATCHUP = (1.0 + 1.0e-6) / 60.0   # most real time one frame will simulate (s): exactly one 1/60 s tick.  When the GPU can't keep up
                                      # the sim slows down but every frame still moves by the same amount, so the motion stays even
LAB_MAX_INFLIGHT = 2                  # frames of physics the CPU may queue ahead of the GPU (more = more input lag and uneven pacing)
LAB_GUN_PERIOD = 1.0                  # the machine gun fires once per second while the button is held
LAB_MIN_SPEED = 0.002                 # slowest simulation speed on the slider (1.0 = real time)
LAB_VIEWS = ("Normal", "Damage", "Strain", "Pieces", "Velocity", "Heat")
# view -> (legend title, low label, high label, ramp);  ramp None = no gradient bar
LAB_VIEW_LEGEND = {1: ("Damage", "intact", "broken", 0), 2: ("Strain", "relaxed", "at its limit", 0),
                   3: ("Pieces", "each colour is one piece", "pale grey = loose chunk", None),
                   4: ("Velocity", "still", "15+ m/s", 0), 5: ("Heat", "cold", "burning", 1)}


class LabApp:
    def __init__(self, root, device=None, view=(1280, 720)):
        self.root = root
        root.title("Destruction lab")
        root.configure(bg=SV_BG)
        root.resizable(False, False)
        self.lab = DestructionLab(device, view[0], view[1])
        self.cam = LabCamera()
        self.paused = False
        self.hover = None
        self.held = None                     # (x, y) while the left button is down
        self.anchor = None                   # where the gravity well sits while held
        self.last_move = 0.0
        self.last_paint = 0.0
        self.orbit_from = None
        self.frames, self.fps_t = 0, time.perf_counter()
        self.awake = LAB_WAKE_FRAMES         # frames of physics left before an untouched scene is allowed to sleep
        self.stats_stale = True              # the overlay needs one refresh after the last step
        self.render_key = None               # what the last picture was drawn from; None forces a redraw
        self.steps_since = 0                 # stepped frames since the last stats readout
        self.was_active = False
        self.photo = None
        self.photo_z = None
        self.zoom = 1                        # screen pixels per rendered pixel (cheap low-resolution rendering)
        self.full = False
        self.panel_on = True
        self.res_win = next((k for k, v in LAB_RES_SIZES.items() if v == tuple(view)), "%d x %d" % tuple(view))
        self.res_full = LAB_FULL_MODES[0]
        self.ms_phys = self.ms_render = self.ms_read = None     # smoothed GPU timings for the overlay
        self.prof_lines = ()                                     # the last "Profile" button result (overlay lines)
        self.sub_sum = self.sub_n = 0                           # substeps run since the last readout (to normalise the physics time)
        self.gun_t = 0.0                     # wall-clock time of the machine gun's last round
        self.step_acc = 0.0                  # physics steps owed from earlier frames (slow motion)
        self.last_t = time.perf_counter()    # wall-clock time of the previous frame: the sim follows the clock, not the frame count
        self.sim_since = 0.0                 # simulated seconds run since the last readout (for the real-time figure)
        self.speed_now = 1.0
        self.rt = 1.0                        # simulated seconds per real second, from the last readout
        self.stats_req, self.stats_idle = False, False   # an asynchronous stats readout is on its way
        self.display_note = ""               # set when --gl was asked for but the Tk display is what you got
        self.perf = PerfMonitor()            # frame-time record behind the performance graphs
        self.sensors = GpuSensors(self.lab.device)
        self.perf_on = tk.IntVar(value=1)
        self._pf_t = None                    # start of the previous frame
        self.perf_photo = None
        self.perf_cache = {}
        self.view = 0                        # index into LAB_VIEWS
        self.legend_cache = {}
        self.fps_cap = LAB_FPS_CAP           # frame-rate limit (0 = unlimited); set from the Frame cap drop-down
        self.slice_from = self.slice_to = None   # the slicer's dragged line (screen pixels)
        self.keys = set()                    # movement keys held (WASD, E up, C down)
        self._mv_t = time.perf_counter()
        self._ui()
        self.place()
        self.tool_changed()
        root.bind("<v>", lambda e: self.cycle_view())
        root.bind("<space>", lambda e: self.toggle_pause())
        root.bind("<r>", lambda e: self.place())
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: self.toggle_fullscreen(False))
        root.bind("<Tab>", lambda e: self.toggle_panel())
        root.bind("<g>", lambda e: self.toggle_perf())
        root.bind("<q>", lambda e: self.tool_menu_toggle())
        root.bind("<e>", lambda e: self.cond_toggle())
        root.bind("<KeyPress>", lambda e: self.key_down(e.keysym))
        root.bind("<KeyRelease>", lambda e: self.key_up(e.keysym))
        root.after(30, self.tick)

    # ---- layout ---------------------------------------------------------------------------------------------
    def _ui(self):
        r = self.root
        W, H = self.lab.w, self.lab.h
        self.canvas = tk.Canvas(r, width=W, height=H, bg=SV_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nw")
        self.img = self.canvas.create_image(0, 0, anchor="nw")
        c = self.canvas
        self.ov_bg = c.create_rectangle(0, 0, 0, 0, fill=SV_BG, outline="", stipple="gray50")
        self.ov_text = c.create_text(14, 12, anchor="nw", fill=SV_FG, font=("Consolas", 10), text="")
        self.perf_item = c.create_image(14, 12, anchor="nw", state="hidden")          # the performance graphs (Tk display only)
        self.perf_txt = {}
        for key in (("header_l", "header_r") + tuple(f"t{i}" for i in range(6)) + tuple(f"v{i}" for i in range(6))
                    + tuple(f"s{i}" for i in range(6))):
            col = "#%02x%02x%02x" % LAB_SEG_COLOURS[int(key[1])] if key[0] == "s" else (SV_DIM if key[0] in "hv" else SV_FG)
            self.perf_txt[key] = c.create_text(0, 0, anchor="ne" if key in ("header_r",) or key.startswith("v") else "nw",
                                               fill=col, font=("Consolas", 9), text="", state="hidden")
        self.ov_help = c.create_text(14, H - 10, anchor="sw", fill=SV_DIM, font=("Segoe UI", 9),
                                     text="Left-click: use the tool     Right-drag: orbit     Wheel: zoom     Space: pause     R: new object     V: view mode     F11: fullscreen     Tab: hide panel")
        self.slice_line = c.create_line(0, 0, 0, 0, fill="#ffd84d", width=3, state="hidden", capstyle="round")
        self.beam_glow = c.create_line(0, 0, 0, 0, fill="#7a1a10", width=9, state="hidden", capstyle="round")
        self.beam = c.create_line(0, 0, 0, 0, fill="#ffd9c0", width=3, state="hidden", capstyle="round")
        self.well_dot = c.create_oval(0, 0, 0, 0, outline="#b987ff", width=3, state="hidden")
        self.wind_arrow = c.create_line(0, 0, 0, 0, fill=SV_ACCENT, width=4, arrow="last", arrowshape=(12, 14, 5), state="hidden")
        self.wind_text = c.create_text(W - 16, 100, anchor="ne", fill=SV_ACCENT, font=("Segoe UI", 10, "bold"), text="", state="hidden")
        # legend for the false-colour views: a gradient bar (rectangles filled from the GPU ramp) with end labels
        self.leg_bg = c.create_rectangle(0, 0, 0, 0, fill=SV_BG, outline="", stipple="gray50", state="hidden")
        self.leg_title = c.create_text(0, 0, anchor="se", fill=SV_FG, font=("Segoe UI", 9, "bold"), text="", state="hidden")
        self.leg_rects = [c.create_rectangle(0, 0, 0, 0, outline="", state="hidden") for _ in range(24)]
        self.leg_lo = c.create_text(0, 0, anchor="nw", fill=SV_DIM, font=("Segoe UI", 9), text="", state="hidden")
        self.leg_hi = c.create_text(0, 0, anchor="ne", fill=SV_DIM, font=("Segoe UI", 9), text="", state="hidden")
        c.bind("<Button-1>", lambda e: self.on_press(self._ev(e)))
        c.bind("<ButtonRelease-1>", lambda e: self.on_release(self._ev(e)))
        c.bind("<B1-Motion>", lambda e: self.on_drag(self._ev(e)))
        c.bind("<Motion>", lambda e: self.on_move(self._ev(e)))
        c.bind("<Leave>", lambda e: setattr(self, "hover", None))
        c.bind("<ButtonPress-3>", lambda e: setattr(self, "orbit_from", (e.x, e.y)))
        c.bind("<B3-Motion>", self.on_orbit)
        c.bind("<MouseWheel>", lambda e: self.cam.zoom(1 if e.delta > 0 else -1))
        c.bind("<Button-4>", lambda e: self.cam.zoom(1))
        c.bind("<Button-5>", lambda e: self.cam.zoom(-1))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Lab.TCombobox", fieldbackground=SV_CARD, background=SV_CARD, foreground=SV_FG, arrowcolor=SV_FG,
                        bordercolor=SV_LINE, lightcolor=SV_CARD, darkcolor=SV_CARD, selectbackground=SV_CARD,
                        selectforeground=SV_FG, padding=3)
        style.map("Lab.TCombobox", fieldbackground=[("readonly", SV_CARD)], foreground=[("readonly", SV_FG)],
                  selectbackground=[("readonly", SV_CARD)], selectforeground=[("readonly", SV_FG)],
                  background=[("active", SV_SEL), ("readonly", SV_CARD)])
        style.configure("Lab.Vertical.TScrollbar", background=SV_CARD, troughcolor=SV_SIDE, bordercolor=SV_SIDE,
                        arrowcolor=SV_DIM, lightcolor=SV_CARD, darkcolor=SV_CARD, gripcount=0)
        style.map("Lab.Vertical.TScrollbar", background=[("active", SV_SEL), ("pressed", SV_SEL)])
        style.configure("Lab.Horizontal.TScale", background=SV_ACCENT, troughcolor=SV_CARD, bordercolor=SV_SIDE,
                        lightcolor=SV_ACCENT, darkcolor=SV_ACCENT, gripcount=0)
        style.map("Lab.Horizontal.TScale", background=[("disabled", SV_LINE), ("active", "#8dbaff")],
                  lightcolor=[("disabled", SV_LINE)], darkcolor=[("disabled", SV_LINE)])
        r.option_add("*TCombobox*Listbox.background", SV_CARD)
        r.option_add("*TCombobox*Listbox.foreground", SV_FG)
        r.option_add("*TCombobox*Listbox.selectBackground", SV_SEL)
        r.option_add("*TCombobox*Listbox.selectForeground", SV_FG)
        # the control panel scrolls: `outer` is what the window lays out (self.side), `side` (inside a canvas) holds the widgets
        outer = self.side = tk.Frame(r, bg=SV_SIDE)
        outer.grid(row=0, column=1, sticky="ns")
        outer.grid_rowconfigure(0, weight=1)
        sc = self.side_canvas = tk.Canvas(outer, bg=SV_SIDE, highlightthickness=0, bd=0, width=285, height=300)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=sc.yview, style="Lab.Vertical.TScrollbar")
        sc.configure(yscrollcommand=vsb.set)
        sc.grid(row=0, column=0, sticky="ns")
        vsb.grid(row=0, column=1, sticky="ns")
        side = tk.Frame(sc, bg=SV_SIDE)
        sc.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", self._side_configure)
        outer.bind("<Enter>", lambda e: self._wheel_bind(True))
        outer.bind("<Leave>", lambda e: self._wheel_bind(False))
        side.grid_columnconfigure(0, weight=1)
        tk.Frame(side, bg=SV_SIDE, width=270, height=1).grid(row=99, column=0)      # fixes the panel's width
        tk.Label(side, text="Destruction lab", bg=SV_SIDE, fg=SV_FG, font=("Segoe UI", 14, "bold"), anchor="w",
                 padx=14, pady=8).grid(row=0, column=0, sticky="ew")
        # PLACE OBJECTS is a drop-down (it sits just above DISPLAY): the heading is a button that shows or hides the settings under it
        box = tk.Frame(side, bg=SV_SIDE, padx=14)
        box.grid(row=19, column=0, sticky="ew")
        ohead = tk.Button(side, text="\u25BE  PLACE OBJECTS", anchor="w", bg=SV_SIDE, fg=SV_ACCENT, activebackground=SV_SIDE,
                          activeforeground=SV_FG, relief="flat", bd=0, font=("Segoe UI", 9, "bold"), padx=14, cursor="hand2",
                          highlightthickness=0)

        def toggle_objects(_=None):
            if box.winfo_manager():
                box.grid_remove()
                ohead.configure(text="\u25B8  PLACE OBJECTS")
            else:
                box.grid()
                ohead.configure(text="\u25BE  PLACE OBJECTS")
            self._side_configure()
        ohead.configure(command=toggle_objects)
        ohead.grid(row=18, column=0, sticky="ew", pady=(8, 2))
        box.grid_columnconfigure(1, weight=1)
        self.shape_v, self.mat_v, self.size_v = tk.StringVar(value="Block"), tk.StringVar(value="Stone"), tk.StringVar(value="Medium")
        for n, (lab, var, vals) in enumerate((("Shape", self.shape_v, LAB_SHAPES), ("Material", self.mat_v, tuple(LAB_MATERIALS)))):
            tk.Label(box, text=lab, bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=n, column=0, sticky="w", pady=2)
            cb = ttk.Combobox(box, textvariable=var, values=vals, state="readonly", width=12, style="Lab.TCombobox")
            cb.grid(row=n, column=1, sticky="e", pady=2)
            cb.bind("<<ComboboxSelected>>", lambda e: self.place())
        # Detail: a log slider over the number of chunks (1,500 .. 500,000).  size_v stays "Medium" (kept for older code that reads it).
        self.detail_v = tk.DoubleVar(value=math.log(4096 / LAB_DETAIL_MIN) / math.log(LAB_DETAIL_MAX / LAB_DETAIL_MIN))
        self._detail_job = None
        tk.Label(box, text="Detail", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=2, column=0, sticky="w", pady=2)
        self.detail_val = tk.Label(box, text="", bg=SV_SIDE, fg=SV_FG, font=("Consolas", 10), anchor="e")
        self.detail_val.grid(row=2, column=1, sticky="e")
        self.detail_sc = ttk.Scale(box, from_=0.0, to=1.0, orient="horizontal", variable=self.detail_v, style="Lab.Horizontal.TScale",
                                   length=230, command=self._detail_moved)
        self.detail_sc.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.detail_warn = tk.Label(box, text="", bg=SV_SIDE, fg=SV_FG, font=("Segoe UI", 9, "bold"), anchor="w")
        self.detail_warn.grid(row=4, column=0, columnspan=2, sticky="w")
        self._detail_labels()
        pb = tk.Frame(side, bg=SV_SIDE, padx=14)
        pb.grid(row=3, column=0, sticky="ew", pady=(8, 2))
        pb.grid_columnconfigure(0, weight=3)
        pb.grid_columnconfigure(1, weight=2)
        self._button(pb, "Place new   (R)", self.place).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.graph_b = self._button(pb, "Graphs: on  (G)", self.toggle_perf)
        self.graph_b.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.tool_v = tk.StringVar(value="hammer")
        # the tools, their sliders and the damage multiplier live in a pop-up opened with Q, not in the side panel
        self.radius_v, self.speed_v = tk.DoubleVar(value=3.0), tk.DoubleVar(value=1.65)
        self.mult_v = tk.DoubleVar(value=1.0)
        tw = self.tool_win = tk.Toplevel(r)
        tw.withdraw()
        tw.overrideredirect(True)
        tw.configure(bg=SV_ACCENT)
        try:
            tw.attributes("-topmost", True)
        except tk.TclError:
            pass
        tb_ = tk.Frame(tw, bg=SV_SIDE, padx=0, pady=10)
        tb_.pack(padx=2, pady=2)
        tb_.grid_columnconfigure(0, weight=1)
        tk.Label(tb_, text="WEAPONS & TOOLS   (click or press a number;  Q or Esc closes)", bg=SV_SIDE, fg=SV_ACCENT,
                 font=("Segoe UI", 9, "bold"), anchor="w", padx=14).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        grid_ = tk.Frame(tb_, bg=SV_SIDE, padx=10)
        grid_.grid(row=1, column=0, sticky="ew")
        self.tool_btns = {}
        for n, (key, name, *_rest) in enumerate(LAB_TOOLS):
            bt = tk.Button(grid_, text=f"{n + 1}   {name}", command=lambda k=key: self._tool_menu_pick(k), anchor="w",
                           bg=SV_CARD, fg=SV_FG, activebackground=SV_SEL, activeforeground=SV_FG, relief="flat", bd=0,
                           font=("Segoe UI", 11), padx=12, pady=8, width=13, cursor="hand2", highlightthickness=0)
            bt.grid(row=n // 3, column=n % 3, padx=3, pady=3, sticky="ew")
            self.tool_btns[key] = bt
        self.hint = tk.Label(tb_, text="", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 9), anchor="w", justify="left",
                             wraplength=420, padx=14, height=2)
        self.hint.grid(row=2, column=0, sticky="ew", pady=(6, 4))
        self.radius_lab = self._slider(tb_, 3, "Size of the area (cm)", self.radius_v, 0.8, 8.0)
        self.speed_lab = self._slider(tb_, 4, "Force", self.speed_v, 0.0, 1.9)
        mb = tk.Frame(tb_, bg=SV_SIDE, padx=14)
        mb.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        tk.Label(mb, text="Damage", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        for m_ in LAB_MULTS:
            tk.Radiobutton(mb, text=("%gx" % m_), value=m_, variable=self.mult_v, command=self.mult_changed, bg=SV_SIDE,
                           fg=SV_FG, selectcolor=SV_SEL, activebackground=SV_SIDE, activeforeground=SV_FG, indicatoron=False,
                           relief="flat", bd=0, font=("Segoe UI", 10), padx=7, pady=3, cursor="hand2", highlightthickness=0,
                           width=4).pack(side="left", padx=2)
        self._button(tb_, "Close", self.tool_menu_toggle).grid(row=6, column=0, sticky="ew", padx=14, pady=(8, 0))
        tw.bind("<Escape>", lambda e: self._tool_menu_close())
        tw.bind("<q>", lambda e: self._tool_menu_close())
        tw.bind("<Key>", self._tool_menu_key)
        # the conditions (wind, earthquake) live in a pop-up opened with E, not in the side panel
        self.wind_on, self.wind_v, self.dir_v = tk.IntVar(value=0), tk.DoubleVar(value=15.0), tk.DoubleVar(value=0.0)
        self.quake_on, self.quake_v = tk.IntVar(value=0), tk.DoubleVar(value=15.0)
        self.sim_v = tk.DoubleVar(value=0.0)                  # log10 of the speed (slider in the E menu)
        self.steps_v = tk.DoubleVar(value=float(LAB_STEPS_DEFAULT))
        cw = self.cond_win = tk.Toplevel(r)
        cw.withdraw()
        cw.overrideredirect(True)
        cw.configure(bg=SV_ACCENT)
        try:
            cw.attributes("-topmost", True)
        except tk.TclError:
            pass
        cb_ = tk.Frame(cw, bg=SV_SIDE, padx=0, pady=8)
        cb_.pack(padx=2, pady=2)
        cb_.grid_columnconfigure(0, weight=1)
        tk.Label(cb_, text="ENVIRONMENT   (E or Esc closes)", bg=SV_SIDE, fg=SV_ACCENT, font=("Segoe UI", 9, "bold"), anchor="w",
                 padx=14).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.wind_lab = self._toggle(cb_, 1, "Wind", self.wind_on, self.wind_v, 0.0, 50.0)
        self.dir_lab = self._slider(cb_, 2, "Wind direction (degrees)", self.dir_v, 0.0, 360.0)
        self.quake_lab = self._toggle(cb_, 3, "Earthquake", self.quake_on, self.quake_v, 0.0, 50.0)
        tk.Label(cb_, text="SIMULATION", bg=SV_SIDE, fg=SV_ACCENT, font=("Segoe UI", 9, "bold"), anchor="w",
                 padx=14).grid(row=4, column=0, sticky="ew", pady=(10, 4))
        self.sim_lab = self._slider(cb_, 5, "Simulation speed", self.sim_v, math.log10(LAB_MIN_SPEED), 0.0)
        self.steps_lab = self._slider(cb_, 6, "Physics steps per 1/60 s", self.steps_v, LAB_STEPS_MIN, LAB_STEPS_MAX)
        self._button(cb_, "Close", self.cond_toggle).grid(row=7, column=0, sticky="ew", padx=14, pady=(8, 0))
        cw.bind("<Escape>", lambda e: self.cond_toggle(False))
        cw.bind("<e>", lambda e: self.cond_toggle(False))
        bar = tk.Frame(side, bg=SV_SIDE, padx=14)
        bar.grid(row=14, column=0, sticky="ew", pady=(10, 0))
        self.pause_b = self._button(bar, "Pause   (Space)", self.toggle_pause)
        self.pause_b.pack(side="left", fill="x", expand=True)
        # simulation speed: a log slider from real time (1x) down to LAB_MIN_SPEED; below one physics step per frame the
        # loop simply skips frames (see tick), so it can go as slow as you like
        mnb = tk.Frame(side, bg=SV_SIDE, padx=14)
        mnb.grid(row=16, column=0, sticky="ew", pady=(8, 0))
        mnb.grid_columnconfigure(0, weight=1)
        mnb.grid_columnconfigure(1, weight=1)
        self._button(mnb, "Weapons   (Q)", self.tool_menu_toggle).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self._button(mnb, "Environment   (E)", self.cond_toggle).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.msg = tk.StringVar(value="")
        tk.Label(side, textvariable=self.msg, bg=SV_SIDE, fg=SV_ACCENT, font=("Segoe UI", 9), anchor="w", justify="left",
                 wraplength=240, padx=14).grid(row=17, column=0, sticky="ew", pady=(8, 2))
        # DISPLAY is a drop-down: the heading is a button that shows or hides the settings under it
        dbox = tk.Frame(side, bg=SV_SIDE, padx=14)
        dbox.grid(row=21, column=0, sticky="ew")
        dhead = tk.Button(side, text="\u25B8  DISPLAY", anchor="w", bg=SV_SIDE, fg=SV_ACCENT, activebackground=SV_SIDE,
                          activeforeground=SV_FG, relief="flat", bd=0, font=("Segoe UI", 9, "bold"), padx=14, cursor="hand2",
                          highlightthickness=0)

        def toggle_display(_=None):
            if dbox.winfo_manager():
                dbox.grid_remove()
                dhead.configure(text="\u25B8  DISPLAY")
            else:
                dbox.grid()
                dhead.configure(text="\u25BE  DISPLAY")
            self._side_configure()
        dhead.configure(command=toggle_display)
        dhead.grid(row=20, column=0, sticky="ew", pady=(8, 2))
        dbox.grid_columnconfigure(1, weight=1)
        tk.Label(dbox, text="Resolution", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=0, column=0, sticky="w", pady=2)
        self.res_v = tk.StringVar(value=self.res_win)
        self.res_cb = ttk.Combobox(dbox, textvariable=self.res_v, values=LAB_RESOLUTIONS, state="readonly", width=14, style="Lab.TCombobox")
        self.res_cb.grid(row=0, column=1, sticky="e", pady=2)
        self.res_cb.bind("<<ComboboxSelected>>", self.res_changed)
        tk.Label(dbox, text="View   (V)", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=1, column=0, sticky="w", pady=2)
        self.view_v = tk.StringVar(value=LAB_VIEWS[0])
        self.view_cb = ttk.Combobox(dbox, textvariable=self.view_v, values=LAB_VIEWS, state="readonly", width=14, style="Lab.TCombobox")
        self.view_cb.grid(row=1, column=1, sticky="e", pady=2)
        self.view_cb.bind("<<ComboboxSelected>>", self.view_changed)
        tk.Label(dbox, text="GPU phases", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=2, column=0, sticky="w", pady=2)
        self._button(dbox, "Profile", self.profile_phases).grid(row=2, column=1, sticky="e", pady=2)
        tk.Label(dbox, text="Anti-aliasing", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=3, column=0, sticky="w", pady=2)
        self.aa_v = tk.StringVar(value="FXAA")
        aa_cb = ttk.Combobox(dbox, textvariable=self.aa_v, values=tuple(LAB_AA_MODES), state="readonly", width=14, style="Lab.TCombobox")
        aa_cb.grid(row=3, column=1, sticky="e", pady=2)
        aa_cb.bind("<<ComboboxSelected>>", self.quality_changed)
        tk.Label(dbox, text="Sharpen", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=4, column=0, sticky="w", pady=2)
        self.sharp_v = tk.DoubleVar(value=0.3)
        ttk.Scale(dbox, from_=0.0, to=1.0, orient="horizontal", variable=self.sharp_v, style="Lab.Horizontal.TScale", length=130,
                  command=lambda _=None: self.quality_changed()).grid(row=4, column=1, sticky="e", pady=2)
        tk.Label(dbox, text="Perf graphs", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=5, column=0, sticky="w", pady=2)
        tk.Checkbutton(dbox, variable=self.perf_on, command=self.perf_toggled, bg=SV_SIDE, fg=SV_FG, selectcolor=SV_SEL,
                       activebackground=SV_SIDE, activeforeground=SV_FG, relief="flat", bd=0, highlightthickness=0,
                       cursor="hand2").grid(row=5, column=1, sticky="e", pady=2)
        tk.Label(dbox, text="Frame cap", bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w").grid(row=6, column=0, sticky="w", pady=2)
        self.cap_v = tk.StringVar(value=str(self.fps_cap))
        cap_cb = ttk.Combobox(dbox, textvariable=self.cap_v, values=("30", "60", "100", "120", "Unlimited"), state="readonly",
                              width=14, style="Lab.TCombobox")
        cap_cb.grid(row=6, column=1, sticky="e", pady=2)
        cap_cb.bind("<<ComboboxSelected>>", self.fps_changed)
        dbox.grid_remove()                                  # starts closed
        self.full_b =self._button(side, "Fullscreen   (F11)", self.toggle_fullscreen)
        self.full_b.grid(row=22, column=0, sticky="ew", padx=14, pady=(6, 10))
        for w_ in side.winfo_children() + [c for f in side.winfo_children() for c in f.winfo_children()]:
            if isinstance(w_, ttk.Combobox):                # a combobox would otherwise take the wheel and change its value
                w_.bind("<MouseWheel>", self._side_wheel)
                w_.bind("<Button-4>", self._side_wheel)
                w_.bind("<Button-5>", self._side_wheel)

    def _side_configure(self, e=None):
        """The panel's content changed size: update the scroll range, and (windowed) make the panel as tall as its content
        but no taller than the screen allows; anything beyond that scrolls."""
        sc, inner = self.side_canvas, self.side_canvas.winfo_children()[0]
        w, h = inner.winfo_reqwidth(), inner.winfo_reqheight()
        sc.configure(scrollregion=(0, 0, w, h), width=w)
        if not getattr(self, "full", False):
            sc.configure(height=max(200, min(h, self.root.winfo_screenheight() - 120)))

    def _side_wheel(self, e):
        d = getattr(e, "delta", 0)
        step = -1 if (d > 0 or getattr(e, "num", 0) == 4) else 1
        sc = self.side_canvas
        if sc.yview() != (0.0, 1.0):                         # only when there is something to scroll
            sc.yview_scroll(step * 3, "units")
        return "break"

    def _wheel_bind(self, on):
        """The wheel scrolls the panel while the pointer is over it (Windows / macOS: <MouseWheel>; X11: buttons 4 and 5)."""
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            if on:
                self.root.bind_all(seq, self._side_wheel)
            else:
                self.root.unbind_all(seq)

    @staticmethod
    def _button(parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=SV_CARD, fg=SV_FG, relief="flat", bd=0, activebackground=SV_SEL,
                         activeforeground=SV_FG, font=("Segoe UI", 10, "bold"), pady=6, cursor="hand2", highlightthickness=0)

    @staticmethod
    def _section(parent, row, text):
        tk.Label(parent, text=text, bg=SV_SIDE, fg=SV_ACCENT, font=("Segoe UI", 9, "bold"), anchor="w", padx=14,
                 pady=0).grid(row=row, column=0, sticky="ew", pady=(8, 2))

    def _slider(self, parent, row, text, var, lo, hi):
        f = tk.Frame(parent, bg=SV_SIDE, padx=14)
        f.grid(row=row, column=0, sticky="ew")
        f.grid_columnconfigure(0, weight=1)
        head = tk.Label(f, text=text, bg=SV_SIDE, fg=SV_DIM, font=("Segoe UI", 10), anchor="w")
        head.grid(row=0, column=0, sticky="w")
        val = tk.Label(f, text="", bg=SV_SIDE, fg=SV_FG, font=("Consolas", 10), anchor="e")
        val.grid(row=0, column=1, sticky="e")
        sc = ttk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var, style="Lab.Horizontal.TScale", length=230,
                       command=lambda _=None: self.update_labels())
        sc.grid(row=1, column=0, columnspan=2, sticky="ew")
        return head, val, sc

    def _toggle(self, parent, row, text, on_var, var, lo, hi):
        """A checkbox (the condition is on or off) with its strength slider underneath."""
        f = tk.Frame(parent, bg=SV_SIDE, padx=14)
        f.grid(row=row, column=0, sticky="ew")
        f.grid_columnconfigure(0, weight=1)
        tk.Checkbutton(f, text=text, variable=on_var, bg=SV_SIDE, fg=SV_FG, selectcolor=SV_CARD, activebackground=SV_SIDE,
                       activeforeground=SV_FG, font=("Segoe UI", 10), bd=0, highlightthickness=0, anchor="w",
                       command=self.update_labels).grid(row=0, column=0, sticky="w")
        val = tk.Label(f, text="", bg=SV_SIDE, fg=SV_FG, font=("Consolas", 10), anchor="e")
        val.grid(row=0, column=1, sticky="e")
        sc = ttk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var, style="Lab.Horizontal.TScale", length=230,
                       command=lambda _=None: self.update_labels())
        sc.grid(row=1, column=0, columnspan=2, sticky="ew")
        return None, val, sc

    # ---- tools ----------------------------------------------------------------------------------------------
    def tool(self):
        return next(t for t in LAB_TOOLS if t[0] == self.tool_v.get())

    def sim_speed(self):
        """Simulation speed from the slider: 1.0 = real time, snapped to exactly 1 near the top."""
        s = 10.0 ** self.sim_v.get()
        return 1.0 if s > 0.97 else max(LAB_MIN_SPEED, s)

    def strength(self):
        return 10.0 ** self.speed_v.get()          # hit speed / power in the tool's own units (log slider)

    def tool_changed(self):
        key, name, hint, radius, speed, colour = self.tool()
        self.hint.configure(text=f"{name}:  {hint}")
        if radius > 0:
            self.radius_v.set(radius)
        if speed > 0:
            self.speed_v.set(math.log10(speed))
        self.radius_lab[2].state(["!disabled"] if radius > 0 else ["disabled"])
        self.speed_lab[2].state(["!disabled"] if speed > 0 else ["disabled"])
        self.speed_lab[0].configure(text=LAB_FORCE_LABEL.get(key, "Amount per click" if speed <= 0 else "Force (m/s for a stone chunk)"))
        self.update_labels()

    def update_labels(self):
        key, _, _, radius, speed, _ = self.tool()
        self.radius_lab[1].configure(text=f"{self.radius_v.get() * 10:.0f} cm" if radius > 0 else "-")
        self.speed_lab[1].configure(text=f"{self.strength():.1f}" if speed > 0 else "-")
        self.wind_lab[1].configure(text=f"{self.wind_v.get():.0f} m/s")
        self.dir_lab[1].configure(text=f"{self.dir_v.get():.0f}")
        self.quake_lab[1].configure(text=f"{self.quake_v.get():.0f} m/s2")
        self.sim_lab[1].configure(text=f"{self.sim_speed():.3g}x")
        n = int(round(self.steps_v.get()))
        if n != self.lab.substeps:                              # the Steps slider moved
            self.lab.set_steps(n)
            self.wake()
        soft = self.lab.soft
        self.steps_lab[1].configure(text=f"{self.lab.substeps}" + ("" if soft >= 0.999 else f"  (soft {soft * 100:.0f}%)"))

    def _detail_labels(self):
        target = lab_detail_target(self.detail_v.get())
        lvl = lab_detail_level(target)
        self.detail_val.config(text=f"{target:,}")
        self.detail_warn.config(text={"WARNING": "WARNING: very heavy, expect slow motion", "CAUTIOUS": "CAUTIOUS: heavy, may run slower"}.get(lvl, ""),
                                fg="#f87171" if lvl == "WARNING" else SV_WARN)
        return target

    def _detail_moved(self, _=None):
        """The slider moved: update the readout now, rebuild shortly after it stops (large builds are not cheap)."""
        self._detail_labels()
        if self._detail_job is not None:
            self.root.after_cancel(self._detail_job)
        self._detail_job = self.root.after(350, self._detail_apply)

    def _detail_apply(self):
        self._detail_job = None
        self.place(keep_view=True)

    def place(self, keep_view=False):
        self.lab.hold_gravity = False                                          # gravity is on as soon as the object is placed
        edge = lab_edge_for(self.shape_v.get(), self._detail_labels())
        n = self.lab.build(self.shape_v.get(), self.mat_v.get(), edge)
        want = int(round(self.steps_v.get()))                                    # the slider's value carries over to a new object...
        if self.lab.stable_steps > want and self.lab.any_ductile:           # ...but metals need enough steps to stay stiff
            want = min(LAB_STEPS_MAX, self.lab.stable_steps)
            self.steps_v.set(float(want))
        self.lab.set_steps(want)
        self.update_labels()
        self.hover = self.anchor = None
        if not keep_view:
            nx, ny, nz = self.lab.dims
            self.cam.target = np.array([0.0, self.lab.height * 0.45, 0.0])
            self.cam.dist = max(2.0, 1.9 * self.lab.s * max(nx, ny, nz))
            self.msg.set(f"{self.mat_v.get()} {self.shape_v.get().lower()}: {n} chunks, {self.lab.total_bonds} bonds. It settles for a moment.")
        self.wake()

    # ---- keyboard: WASD camera, Q tool menu --------------------------------------------------------------------
    MOVE_KEYS = ("w", "a", "s", "d", "f", "c")

    def key_down(self, name):
        name = str(name).lower()
        if name in self.MOVE_KEYS:
            self.keys.add(name)

    def key_up(self, name):
        self.keys.discard(str(name).lower())

    def _move_camera(self):
        """WASD slides the camera's target across the floor (W = away from you, in the direction you look), F up, C down.
        The speed grows with the zoom distance, so it feels the same close up and far out."""
        now = time.perf_counter()
        dt = min(0.1, now - self._mv_t)
        self._mv_t = now
        if not self.keys:
            return
        cam = self.cam
        fwd = np.array([-math.sin(cam.yaw), 0.0, -math.cos(cam.yaw)])
        right = np.array([-fwd[2], 0.0, fwd[0]])
        v = np.zeros(3)
        if "w" in self.keys: v += fwd
        if "s" in self.keys: v -= fwd
        if "d" in self.keys: v += right
        if "a" in self.keys: v -= right
        if "f" in self.keys: v[1] += 1.0
        if "c" in self.keys: v[1] -= 1.0
        cam.target = np.clip(cam.target + v * cam.dist * 0.6 * dt, [-12.0, 0.0, -12.0], [12.0, 10.0, 12.0])

    def tool_menu_toggle(self):
        """Q: the tool pop-up (tools, size, force, damage).  Q, Esc or Close hides it."""
        tw = self.tool_win
        if tw.winfo_viewable():
            self._tool_menu_close()
            return
        self._tool_menu_refresh()
        tw.update_idletasks()
        r = self.root
        x = r.winfo_rootx() + max(0, (r.winfo_width() - tw.winfo_reqwidth()) // 2)
        y = r.winfo_rooty() + max(0, (r.winfo_height() - tw.winfo_reqheight()) // 2)
        tw.geometry(f"+{x}+{y}")
        tw.deiconify()
        tw.focus_force()

    def _tool_menu_refresh(self):
        for key, bt in self.tool_btns.items():
            bt.configure(bg=SV_SEL if key == self.tool_v.get() else SV_CARD)

    def _tool_menu_close(self):
        self.tool_win.withdraw()
        self.root.focus_set()

    def cond_toggle(self, on=None):
        """E: show or hide the conditions pop-up (wind, earthquake).  It stays open while the simulation runs."""
        cw = self.cond_win
        on = (not cw.winfo_viewable()) if on is None else bool(on)
        if on:
            cw.update_idletasks()
            r = self.root
            x = r.winfo_rootx() + max(0, r.winfo_width() - cw.winfo_reqwidth() - 320)
            y = r.winfo_rooty() + 60
            cw.geometry(f"+{x}+{y}")
            cw.deiconify()
            cw.focus_force()
        else:
            cw.withdraw()
            self.root.focus_set()
        self.update_labels()

    def _tool_menu_key(self, e):
        if e.char and e.char.isdigit():
            n = int(e.char) - 1
            if 0 <= n < len(LAB_TOOLS):
                self._tool_menu_pick(LAB_TOOLS[n][0], close=True)

    def _tool_menu_pick(self, key, close=False):
        """Clicking a tool selects it and leaves the menu open (so size and force can be set); a number key also closes it."""
        self.tool_v.set(key)
        self.tool_changed()
        self._tool_menu_refresh()
        if close:
            self._tool_menu_close()
        self.wake()

    def quality_changed(self, _=None):
        """Anti-aliasing mode or sharpening changed: apply it and redraw the picture."""
        aa, ss = LAB_AA_MODES.get(self.aa_v.get(), (1, 1))
        got = self.lab.set_quality(aa=aa, ss=ss, sharp=self.sharp_v.get())
        note = f"   (supersampling lowered to {got}x: the picture would be too large)" if got != ss else ""
        self.msg.set(f"Picture: {self.aa_v.get()}, sharpen {self.sharp_v.get() * 100:.0f}%{note}")
        self.render_key = None
        self.root.focus_set()
        self.wake()

    def grab_gain(self):
        """Spring stiffness of the grab tools from the Force slider (45 -> 180 per s^2)."""
        return max(10.0, self.strength() * 4.0)

    def mult(self):
        return float(self.mult_v.get())

    def mult_changed(self):
        self.msg.set(f"Damage x{self.mult():g}: laser heat, projectile mass, hit / blast / weaken strength")

    def wake(self):
        """Something happened that can change the scene: run the physics for a while and redraw."""
        self.awake = LAB_WAKE_FRAMES
        self.stats_stale = True
        self.render_key = None

    def apply(self, pick, painting=False):
        if pick is None:
            return
        self.wake()
        point, dirn = pick
        key = self.tool()[0]
        radius = self.radius_v.get()
        if key == "weaken":
            self.lab.weaken(point, radius * 0.1, (0.05 if painting else 0.3) * self.mult())
        elif key == "mend":
            self.lab.mend(point, radius * 0.1, 0.05 if painting else 0.3)
        else:
            m = self.mult()
            self.lab.hit(point, dirn, radius * 0.1, self.strength() * m, blast=(key == "blast"))
            if m > 1.0:                                       # chunk speeds are capped, so extra power cracks the bonds directly
                self.lab.weaken(point, radius * 0.1, LAB_EXTRA_DMG * (m - 1.0))
            self.msg.set(f"{self.tool()[1]}: {self.strength() * m:.1f} m/s over {radius * 10:.0f} cm" + (f"   (x{m:g})" if m != 1.0 else ""))

    # ---- mouse ----------------------------------------------------------------------------------------------
    def on_press(self, e):
        key = self.tool()[0]
        if key == "slice":
            self.slice_from = self.slice_to = (e.x, e.y)
            return
        self.held = (e.x, e.y)
        pk = self.lab.pick(self.cam, e.x, e.y)
        if key in ("grab", "tug"):
            if pk:
                n = self.lab.grab_start(self.cam, pk[0], self.grab_gain(), whole=(key == "grab"), radius=self.radius_v.get() * 0.1)
                self.msg.set(f"Holding {n} chunks")
                self.wake()
            return
        if key in ("cannon", "well", "laser", "gun"):
            self.wake()
        if key == "gun":
            self.gun_t = -1.0e9                              # the first round goes off at once
        if key == "cannon":
            eye, dirn = self.lab.ray(self.cam, e.x, e.y)
            self.lab.fire(self.cam, pk[0] if pk else eye + dirn * self.cam.dist, self.strength(), mass_mult=self.mult())
            self.msg.set(f"Cannonball fired at {self.strength():.0f} m/s" + (f"   (mass x{self.mult():g})" if self.mult() != 1.0 else ""))
        elif key == "well":
            eye, dirn = self.lab.ray(self.cam, e.x, e.y)
            self.anchor = pk[0] if pk else eye + dirn * self.cam.dist
        elif key != "laser":
            self.apply(pk)

    def on_release(self, e):
        if self.slice_from is not None:
            p0, p1 = self.slice_from, (e.x, e.y)
            self.slice_from = self.slice_to = None
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 6:
                self.lab.slice(self.cam, p0, p1)
                self.msg.set("Sliced.")
                self.wake()
            return
        self.held = self.anchor = None
        self.lab.well = None
        self.lab.grab_end()

    def on_drag(self, e):
        if self.slice_from is not None:
            self.slice_to = (e.x, e.y)
            return
        if self.held:
            self.held = (e.x, e.y)
        if self.tool()[0] in ("weaken", "mend"):
            now = time.perf_counter()
            if now - self.last_paint > 0.05:
                self.last_paint = now
                self.apply(self.lab.pick(self.cam, e.x, e.y), painting=True)
        self.on_move(e)

    def on_move(self, e):
        now = time.perf_counter()
        if now - self.last_move > 0.05:
            self.last_move = now
            pk = self.lab.pick(self.cam, e.x, e.y)
            self.hover = pk[0] if pk else None

    def on_orbit(self, e):
        if self.orbit_from:
            self.cam.orbit(e.x - self.orbit_from[0], e.y - self.orbit_from[1])
        self.orbit_from = (e.x, e.y)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_b.configure(text="Resume   (Space)" if self.paused else "Pause   (Space)")

    # ---- the frame loop -------------------------------------------------------------------------------------
    def tick(self):
        t0 = time.perf_counter()
        self._frame()
        dt = time.perf_counter() - t0
        iv = 1000.0 / self.fps_cap if self.fps_cap else 1.0
        self.root.after(max(1, int(iv - dt * 1000)), self.tick)

    def _frame(self):
        """One pass of the frame loop: inputs, physics steps, picture, overlays, stats.  Tk calls it through tick();
        the OpenGL window (LabAppGL) calls it from its own loop."""
        lab, key = self.lab, self.tool()[0]
        self._move_camera()
        t_pf = time.perf_counter()
        if self._pf_t is not None:
            self.perf.frame(t_pf - self._pf_t)                # real time since the last frame: the 'response time'
        self._pf_t = t_pf
        az = math.radians(self.dir_v.get())
        speed = self.wind_v.get() if self.wind_on.get() else 0.0
        lab.wind = (math.cos(az) * speed, 0.0, math.sin(az) * speed)
        lab.quake_amp = self.quake_v.get() if self.quake_on.get() else 0.0
        if lab.quake_amp > 0.0 or speed > 0.0:
            lab.touch()                                       # wind and earthquake count as interaction: gravity comes on
        # Speed: a normal frame runs lab.substeps physics steps (1/60 s).  At speed s we owe lab.substeps * s steps per
        # frame; the fraction is carried over, so below one step per frame some frames simply run no physics at all.
        self.speed_now = self.sim_speed()
        substeps = 0
        if not self.paused:
            now_w = time.perf_counter()
            wall = min(now_w - self.last_t, self.max_catchup())       # real seconds since the last frame, capped so one hitch can't snowball
            self.last_t = now_w
            self.step_acc += lab.substeps * 60.0 * wall * self.speed_now
            substeps = int(self.step_acc)
            self.step_acc -= substeps
        else:
            self.step_acc = 0.0
            self.last_t = time.perf_counter()
        beam = None
        if self.held and not self.paused:
            if key == "laser" and substeps > 0:
                pk = lab.pick(self.cam, *self.held)
                if pk:
                    eye = self.cam.basis()[0]
                    lab.laser(eye, pk[1], pk[0], self.radius_v.get() * 0.1, self.strength() * 0.3 * self.mult(), substeps * lab.dt)
                    beam = self.held
            elif key == "well" and self.anchor is not None:
                lab.well = (self.anchor, self.strength())
                lab.touch()
            elif key == "gun":
                now = time.perf_counter()                   # wall-clock seconds: the rhythm you feel, whatever the physics speed
                if now - self.gun_t >= LAB_GUN_PERIOD:
                    self.gun_t = now
                    pk = lab.pick(self.cam, *self.held)
                    eye, dirn = lab.ray(self.cam, *self.held)
                    lab.fire(self.cam, pk[0] if pk else eye + dirn * self.cam.dist, self.strength(), bullet=True, mass_mult=self.mult())
                    self.msg.set(f"Machine gun: {self.strength():.0f} m/s, one round per second")
        if self.held and key in ("grab", "tug") and lab.grab is not None:
            lab.grab_move(self.cam, *self.held, gain=self.grab_gain())
            self.awake = LAB_WAKE_FRAMES
        if self.held and key in ("laser", "well", "gun"):
            self.awake = LAB_WAKE_FRAMES                      # a held tool keeps changing things
        continuous = lab.quake_amp > 0.0 or any(abs(x) > 1.0e-6 for x in lab.wind)    # wind / earthquake never settle
        active = not self.paused and (self.awake > 0 or continuous)
        stepped = False
        if active and substeps > 0:
            if not self.was_active:                           # just woke up: don't count the idle gap in the fps figure
                self.fps_t, self.steps_since = time.perf_counter(), 0
            every = max(1, round(LAB_STATS_EVERY * min(1.0, self.speed_now)))      # slower speed: read the stats more often
            t_a = time.perf_counter()
            lab.step(substeps, want_load=(self.steps_since + 1 >= every or self.awake <= 1))
            self.perf.add("step", time.perf_counter() - t_a)
            stepped = True
            self.sub_sum += substeps
            self.sub_n += 1
            self.sim_since += substeps * lab.dt
            self.perf.sim_s += substeps * lab.dt
            self.frames += 1
            self.steps_since += 1
            self.stats_stale = True
            if self.awake > 0:
                self.awake -= substeps / lab.substeps      # counted in simulated time, so slow motion doesn't fall asleep early
        self.was_active = active
        key_, name, hint, rad, _, colour = self.tool()
        radius = rad and self.radius_v.get() * 0.1
        hv = None if self.hover is None else tuple(float(x) for x in self.hover)
        rkey = (self.cam.yaw, self.cam.pitch, self.cam.dist, tuple(float(x) for x in self.cam.target), hv, radius, colour, self.view)
        if stepped or rkey != self.render_key:               # nothing moved and the camera is still: keep the last picture
            t_a = time.perf_counter()
            self._draw_picture(radius, colour)
            self.perf.add("pic", time.perf_counter() - t_a)
            self.perf.draws += 1
            self.render_key = rkey
        self._overlays(beam)
        self._perf_update()
        t_a = time.perf_counter()
        if not self.stats_req:                                # start a readout (it is fetched, without waiting, a frame or more later)
            if stepped and self.steps_since >= every:
                self.stats_req, self.stats_idle = True, False
            elif not active and self.stats_stale:
                self.stats_req, self.stats_idle = True, True   # one last readout after the physics goes to sleep
            if self.stats_req:
                lab.stats_request()
        if self.stats_req:
            st = lab.stats_poll()
            if st is not None:
                self.stats_req = False
                self.show_stats(idle=self.stats_idle, st=st)
            elif not lab.stats_busy():                        # the request vanished (new object, resize): ask again later
                self.stats_req = False
        self.perf.add("stat", time.perf_counter() - t_a)

    def _draw_picture(self, radius, colour):
        """Render the scene and put it on screen.  Tk path: GPU -> host PPM bytes -> Tk image (LabAppGL overrides this)."""
        ppm = self.lab.render(self.cam, self.hover, radius, colour, self.view)
        self._show(ppm)

    def _overlays(self, beam):
        Z = self.zoom
        if self.slice_from is not None and self.slice_to is not None:
            self.canvas.coords(self.slice_line, self.slice_from[0] * Z, self.slice_from[1] * Z, self.slice_to[0] * Z, self.slice_to[1] * Z)
            self.canvas.itemconfigure(self.slice_line, state="normal")
        else:
            self.canvas.itemconfigure(self.slice_line, state="hidden")
        c, W, H = self.canvas, self.lab.w * Z, self.lab.h * Z
        if beam:
            for item in (self.beam_glow, self.beam):
                c.coords(item, W * 0.5, H, beam[0] * Z, beam[1] * Z)
                c.itemconfigure(item, state="normal")
        else:
            c.itemconfigure(self.beam_glow, state="hidden")
            c.itemconfigure(self.beam, state="hidden")
        scr = self.cam.project(self.anchor, self.lab.w, self.lab.h, self.lab.f) if self.anchor is not None and self.held else None
        if scr:
            r = 10 + 4 * math.sin(time.perf_counter() * 8)
            c.coords(self.well_dot, scr[0] * Z - r, scr[1] * Z - r, scr[0] * Z + r, scr[1] * Z + r)
            c.itemconfigure(self.well_dot, state="normal")
        else:
            c.itemconfigure(self.well_dot, state="hidden")
        if self.wind_on.get():
            _, fwd, right, up = self.cam.basis()
            az = math.radians(self.dir_v.get())
            v = np.array([math.cos(az), 0.0, math.sin(az)])
            dx, dy = float(v @ right), -float(v @ up)
            ln = math.hypot(dx, dy) or 1.0
            L = 18 + 40 * min(1.0, self.wind_v.get() / 50.0)
            cx0, cy0 = W - 70, 62
            c.coords(self.wind_arrow, cx0 - dx / ln * L / 2, cy0 - dy / ln * L / 2, cx0 + dx / ln * L / 2, cy0 + dy / ln * L / 2)
            c.itemconfigure(self.wind_arrow, state="normal")
            c.itemconfigure(self.wind_text, text=f"WIND {self.wind_v.get():.0f} m/s", state="normal")
        else:
            c.itemconfigure(self.wind_arrow, state="hidden")
            c.itemconfigure(self.wind_text, state="hidden")

    def _show(self, ppm):
        """Put the picture on screen, reusing one PhotoImage (no new image object every frame).  When the picture is
        rendered at a lower resolution than the window (zoom > 1) Tk blows it up with its own integer zoom."""
        try:
            data = ppm
            if self.photo is None:
                self.photo = tk.PhotoImage(data=data, format="PPM")
            else:
                self.photo.configure(data=data, format="PPM")
        except (tk.TclError, TypeError):
            data = bytes(ppm)                                  # some Tk builds want bytes, not a bytearray
            if self.photo is None:
                self.photo = tk.PhotoImage(data=data, format="PPM")
            else:
                self.photo.configure(data=data, format="PPM")
        shown = self.photo
        if self.zoom > 1:
            if self.photo_z is None:
                self.photo_z = tk.PhotoImage(width=self.lab.w * self.zoom, height=self.lab.h * self.zoom)
            self.photo_z.tk.call(self.photo_z, "copy", self.photo, "-zoom", self.zoom, self.zoom)
            shown = self.photo_z
        self.canvas.itemconfigure(self.img, image=shown)

    # ---- display: resolution, fullscreen --------------------------------------------------------------------
    def _ev(self, e):
        """Mouse events arrive in screen pixels; the lab works in rendered pixels."""
        if self.zoom > 1:
            e.x, e.y = e.x // self.zoom, e.y // self.zoom
        return e

    def res_changed(self, _=None):
        if self.full:
            self.res_full = self.res_v.get()
        else:
            self.res_win = self.res_v.get()
        self.root.focus_set()
        self.msg.set("Changing resolution ...")
        self.root.after(120, self._apply_view_later)      # not inside the drop-down's own click: its list is still open and grabbing the mouse

    def _apply_view_later(self):
        print("[lab] resolution change: start", flush=True)
        self.apply_view()
        print("[lab] resolution change: done", flush=True)
        self.msg.set("Resolution: " + self.res_v.get())

    def toggle_panel(self):
        if not self.full:
            return
        self.panel_on = not self.panel_on
        if self.panel_on:
            self.side.grid(row=0, column=1, sticky="ns")
        else:
            self.side.grid_remove()
        self.root.update_idletasks()
        self.apply_view()

    def toggle_fullscreen(self, on=None):
        on = (not self.full) if on is None else bool(on)
        if on == self.full:
            return
        self.full = on
        self.root.resizable(True, True) if on else None
        self.root.attributes("-fullscreen", on)
        if not on:
            self.root.resizable(False, False)
            self.panel_on = True
            self.side.grid(row=0, column=1, sticky="ns")
            self.root.geometry("")
        self.res_cb.configure(values=LAB_FULL_MODES if on else LAB_RESOLUTIONS)
        self.res_v.set(self.res_full if on else self.res_win)
        self.full_b.configure(text="Exit fullscreen   (Esc)" if on else "Fullscreen   (F11)")
        self.root.update_idletasks()
        self.apply_view()

    def view_changed(self, _=None):
        self.view = LAB_VIEWS.index(self.view_v.get())
        self.update_legend()
        self.render_key = None                                 # same simulation, different colours: redraw now
        self.root.focus_set()

    def cycle_view(self):
        self.view_v.set(LAB_VIEWS[(self.view + 1) % len(LAB_VIEWS)])
        self.view_changed()

    def update_legend(self):
        """Show, move or hide the colour key for the current view."""
        c, Z = self.canvas, self.zoom
        W, H = self.lab.w * Z, self.lab.h * Z
        items = [self.leg_bg, self.leg_title, self.leg_lo, self.leg_hi] + self.leg_rects
        info = LAB_VIEW_LEGEND.get(self.view)
        if info is None:
            for it in items:
                c.itemconfigure(it, state="hidden")
            return
        title, lo, hi, pal = info
        bar_w, x1, y1 = 144, W - 20, H - 34          # bar spans x1 - bar_w .. x1, y1 - 12 .. y1
        x0 = x1 - bar_w
        c.itemconfigure(self.leg_title, text=title, state="normal")
        c.coords(self.leg_title, x1, y1 - 18)
        if pal is None:                                # no gradient: two lines of text
            for r in self.leg_rects:
                c.itemconfigure(r, state="hidden")
            c.itemconfigure(self.leg_lo, text=lo, anchor="ne", state="normal")
            c.coords(self.leg_lo, x1, y1 - 14)
            c.itemconfigure(self.leg_hi, text=hi, anchor="ne", state="normal")
            c.coords(self.leg_hi, x1, y1)
            top = y1 - 32
            c.coords(self.leg_bg, x1 - 190, top, x1 + 10, y1 + 18)
        else:
            if pal not in self.legend_cache:
                self.legend_cache[pal] = self.lab.legend_colours(pal, len(self.leg_rects))
            n = len(self.leg_rects)
            for k, (r, rgb) in enumerate(zip(self.leg_rects, self.legend_cache[pal])):
                hexcol = "#%02x%02x%02x" % tuple(max(0, min(255, int(v * 255))) for v in rgb)
                c.coords(r, x0 + bar_w * k / n, y1 - 12, x0 + bar_w * (k + 1) / n + 1, y1)
                c.itemconfigure(r, fill=hexcol, state="normal")
            c.itemconfigure(self.leg_lo, text=lo, anchor="nw", state="normal")
            c.coords(self.leg_lo, x0, y1 + 3)
            c.itemconfigure(self.leg_hi, text=hi, anchor="ne", state="normal")
            c.coords(self.leg_hi, x1, y1 + 3)
            c.coords(self.leg_bg, x0 - 12, y1 - 34, x1 + 12, y1 + 22)
        c.itemconfigure(self.leg_bg, state="normal")
        for it in [self.leg_bg, self.leg_title] + self.leg_rects + [self.leg_lo, self.leg_hi]:
            c.tag_raise(it)                            # bg first, then the bar and text above it

    def apply_view(self):
        """Set the rendered size and the on-screen size from the chosen resolution."""
        if self.full:
            pw = self.side.winfo_reqwidth() if self.panel_on else 0
            aw, ah = max(320, self.root.winfo_screenwidth() - pw), max(240, self.root.winfo_screenheight())
            z = max(1, int(round(ah / float(lab_res_size(self.res_v.get())[1]))))      # Tk can only enlarge by whole numbers
            w, h = aw // z, ah // z
        else:
            z = 1
            w, h = lab_res_size(self.res_v.get())
        self.zoom = z
        self.lab.resize(w, h)
        W, H = w * z, h * z
        c = self.canvas
        c.configure(width=W, height=H)
        c.coords(self.ov_help, 14, H - 10)
        c.coords(self.wind_text, W - 16, 100)
        self.update_legend()
        self.photo = self.photo_z = None                      # sizes changed: make fresh images on the next picture
        c.itemconfigure(self.img, image="")
        self.wake()

    def _take_timings(self):
        """Fold the GPU timings measured since the last readout into smoothed figures.  Physics is normalised to a full
        frame (lab.substeps steps = 1/60 s), so it reads the same at any simulation speed."""
        lab = self.lab
        ph, rd = lab.t_phys.take(), lab.t_render.take()
        if ph and self.sub_n:
            full = (sum(ph) / len(ph)) * lab.substeps / max(self.sub_sum / self.sub_n, 1e-9)
            self.ms_phys = full if self.ms_phys is None else 0.5 * self.ms_phys + 0.5 * full
        if rd:
            ms = sum(rd) / len(rd)
            self.ms_render = ms if self.ms_render is None else 0.5 * self.ms_render + 0.5 * ms
        self.sub_sum = self.sub_n = 0

    def profile_phases(self):
        """The 'Profile' button: time each phase of a physics step with CUDA events and keep the result on the overlay."""
        st = self.lab.stats()
        res = self.lab.profile(8)
        steps = res.pop("steps")
        res.pop("mark_ms", None)
        total = sum(res.values())
        order = sorted(res.items(), key=lambda kv: -kv[1])
        pct = 100.0 * st["broken"] / max(st["total"], 1)
        self.prof_lines = (f"GPU phases at {pct:.0f}% broken, {st['loose']} loose, {st['molten']} molten ({steps} steps),",
                           "ms per 1/60 s; kernels one by one, with the timer's own cost taken off:",) + tuple(
            f"   {name:10s}{ms:8.2f} ms  {100.0 * ms / max(total, 1e-9):3.0f}%" for name, ms in order if ms > 0.0) + (
            f"   {'sum':10s}{total:8.2f} ms",)
        self.msg.set("Profiled: " + ", ".join(f"{n} {ms:.1f}" for n, ms in order[:3]) + " ms")
        self.show_stats()

    def _timing_lines(self):
        """Overlay lines: where the GPU time goes, so the next optimisation is chosen from numbers."""
        lab = self.lab
        extra = ("",) + self.prof_lines if self.prof_lines else ()
        if self.perf_on.get():                                 # the numbers live in the graphs above
            return ("",) + ((self.display_note,) if self.display_note else ()) + self.prof_lines
        if self.ms_phys is None or self.ms_render is None:
            return ("", "timing: measuring ...") + extra
        total = self.ms_phys + self.ms_render
        return ("",
                f"Physics  {self.ms_phys:7.2f} ms   per 1/60 s ({lab.substeps} steps{', graph' if lab.use_graph else ', NO graph'})",
                f"Render   {self.ms_render:7.2f} ms   {lab.w} x {lab.h}",
                f"Readout  {self.ms_read:7.2f} ms   every {max(1, round(LAB_STATS_EVERY * min(1.0, self.speed_now)))} frames"
                if self.ms_read is not None else "",
                f"Physics is {100.0 * self.ms_phys / max(total, 1e-9):.0f}% of GPU time  ->  at most {1000.0 / max(total, 1e-9):.0f} frames/s") + extra

    # ---- performance graphs ---------------------------------------------------------------------------------
    def perf_ph(self):
        """Height of one graph: smaller in a small window."""
        h = self.lab.h * self.zoom
        return 54 if h >= 640 else (34 if h >= 540 else 24)

    def stats_y(self):
        """Top of the simulation-stats text: below the graphs when they are on."""
        return 12 + (self.perf.layout(self.perf_ph())[2] + 10 if self.perf_on.get() else 0)

    def fps_changed(self, _=None, quiet=False):
        """The Frame cap drop-down: limit how many pictures a second are drawn (30 / 60 / 100 / 120 / unlimited).
        The simulation still follows the clock, so a lower cap draws less often but keeps the same speed."""
        t = self.cap_v.get()
        self.fps_cap = int(t) if t.isdigit() else 0
        gl = getattr(self, "glv", None)
        if gl is not None:
            gl.set_vsync(0 < self.fps_cap <= 60)          # vsync would hold 100 / 120 down to the screen's refresh rate
        self.step_acc, self.last_t = 0.0, time.perf_counter()
        if not quiet:
            self.msg.set("Frame cap: " + (f"{self.fps_cap} fps" if self.fps_cap else "unlimited"))
        self.root.focus_set()

    def max_catchup(self):
        """Most real time one frame will simulate: one frame's worth at the cap (never less than 1/60 s)."""
        return max(1.0 / 60.0, 1.0 / self.fps_cap if self.fps_cap else 0.0) * (1.0 + 1.0e-6)

    def toggle_perf(self):
        self.perf_on.set(0 if self.perf_on.get() else 1)
        self.perf_toggled()

    def perf_toggled(self):
        self.graph_b.configure(text="Graphs: on  (G)" if self.perf_on.get() else "Graphs: off  (G)")
        self.canvas.coords(self.ov_text, 14, self.stats_y())
        if self.perf_on.get():
            self.perf_cache = {}
            self._perf_refresh()
        else:
            self._perf_hide()
        self.stats_stale = True
        self.render_key = None
        self.show_stats(idle=not (self.awake > 0))

    def _perf_update(self):
        if self.perf_on.get() and self.perf.sample(time.perf_counter(), self.ms_phys, self.ms_render, self.sensors):
            self._perf_refresh()

    def _perf_hide(self):
        self.canvas.itemconfigure(self.perf_item, state="hidden")
        for cid in self.perf_txt.values():
            self.canvas.itemconfigure(cid, state="hidden")

    def _perf_refresh(self):
        ph = self.perf_ph()
        img, texts = self.perf.draw(ph)
        blk, head, H = self.perf.layout(ph)
        if "header_r" in texts and self.perf.cur is not None:
            texts["header_r"] = f"sim {self.rt:.2f}x"
        c, xl, xr = self.canvas, 14 + 8, 14 + LAB_PERF_W - 8
        for key, cid in self.perf_txt.items():
            txt = texts.get(key, "")
            if key.startswith("header"):
                y = 12 + 3
            elif key[0] == "s":
                y = 12 + head + 5 * blk + 18 + ph + 3
            else:
                y = 12 + head + int(key[1]) * blk + 2
            x = xr if key == "header_r" or key[0] == "v" else (xl + int(key[1]) * 66 if key[0] == "s" else xl)
            sig = (txt, y, x)
            if self.perf_cache.get(key) == sig:
                continue
            self.perf_cache[key] = sig
            c.coords(cid, x, y)
            c.itemconfigure(cid, text=txt, state="normal" if txt else "hidden")
        c.coords(self.ov_text, 14, self.stats_y())
        self._perf_show(img)

    def _perf_show(self, img):
        ppm = b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + img.tobytes()
        if self.perf_photo is None:
            self.perf_photo = tk.PhotoImage(data=ppm, format="PPM")
        else:
            self.perf_photo.configure(data=ppm, format="PPM")
        self.canvas.itemconfigure(self.perf_item, image=self.perf_photo, state="normal")

    def show_stats(self, idle=False, st=None):
        """Update the overlay.  st: a readout that has already arrived (stats_poll); without it the numbers are read now,
        waiting for the GPU."""
        now = time.perf_counter()
        fps = self.steps_since / max(now - self.fps_t, 1e-6)
        self.rt = self.sim_since / max(now - self.fps_t, 1e-6)          # simulated seconds per real second
        self.sim_since = 0.0
        self.fps_t, self.steps_since, self.stats_stale = now, 0, False
        if st is None:
            self.stats_req = False
            st = self.lab.stats()
        read_ms = (time.perf_counter() - now) * 1000.0         # the readout itself (the picture's sync has already drained the GPU)
        self.ms_read = read_ms if self.ms_read is None else 0.5 * self.ms_read + 0.5 * read_ms
        self._take_timings()
        if LAB_VIEWS[self.view] == "Pieces":
            self.render_key = None                             # the piece labels may just have changed: redraw once
        if st["ke"] > LAB_WAKE_KE or st["heat"] > LAB_WAKE_HEAT:
            self.awake = max(self.awake, LAB_WAKE_FRAMES)      # still moving, or still hot: keep simulating
            idle = False
        pct = 100.0 * st["broken"] / max(st["total"], 1)
        lines = (f"Kinetic energy   {st['ke']:10.1f} J", f"Bonds broken     {st['broken']:6d} of {st['total']}  ({pct:.1f}%)",
                 f"Hardest bond     {st['load']:6.0f}% of its limit",
                 f"Pieces {st['pieces']}   Loose {st['loose']}   Burned {st['burned']}" + (f"   Molten {st['molten']}" if st["molten"] else ""),
                 f"Largest piece    {st['biggest']:6d} chunks",
                 "asleep (nothing is moving)" if idle else
                 (f"{fps:.0f} frames/s   sim {self.rt:.2f}x real time" + ("" if self.rt >= 0.95 * self.speed_now else "  (can't keep up)")
                  if self.speed_now >= 1.0 else f"{fps:.0f} frames/s   sim {self.rt:.2f}x real time (asked {self.speed_now:.3g}x)")) + self._timing_lines()
        if self.perf_on.get():                                 # graphs on: the box only appears for a Profile result or a warning
            lines = tuple(l for l in self._timing_lines() if l)
        show = bool(lines)
        self.canvas.itemconfigure(self.ov_text, text="\n".join(lines), state="normal" if show else "hidden")
        self.canvas.itemconfigure(self.ov_bg, state="normal" if show else "hidden")
        self.canvas.coords(self.ov_text, 14, self.stats_y())
        if show:
            x0, y0, x1, y1 = self.canvas.bbox(self.ov_text)
            self.canvas.coords(self.ov_bg, x0 - 8, y0 - 6, x1 + 8, y1 + 6)


class GLView:
    """The Destruction lab's picture in an OpenGL window, fed straight from the GPU.

    The Tk path costs, every frame:  GPU -> host copy, a synchronisation, Tk parsing a PPM on one CPU core.  Here the
    renderer writes the finished picture into an OpenGL pixel buffer that CUDA has mapped (Warp's RegisteredGLBuffer),
    OpenGL copies that buffer into a texture on the GPU, and the texture is drawn.  The pixels never visit the CPU.

    The text, legend, laser beam, gravity-well ring and wind arrow are still laid out by LabApp on its (hidden) Tk
    canvas; this class reads those canvas items back and draws them with pyglet, so the layout logic exists once.

    Needs  pip install pyglet  (2.1 or newer).  For the zero-copy path the window must be on the same GPU as CUDA: on
    a laptop with two GPUs set python.exe to "High performance" (Windows: Settings > System > Display > Graphics).
    Otherwise Warp quietly falls back to copying through the CPU, and the overlay says so (`interop` is False).
    If you ever see torn or flickering pictures, set GLView.sync_before_unmap = True."""

    sync_before_unmap = False

    def __init__(self, app, w, h, zoom=1, host=None):
        import pyglet
        from pyglet import gl
        self.pyglet, self.gl = pyglet, gl
        self.app = app
        self.dev = app.lab.device
        self.w, self.h, self.zoom = w, h, zoom
        self.embedded, self.win = False, None
        pw, ph = max(64, int(round(w * zoom))), max(64, int(round(h * zoom)))
        if host is not None:                                  # try to live inside the Tk window (Windows)
            try:
                self.win = pyglet.window.Window(pw, ph, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
                                                visible=False, vsync=True)
                self._embed(host, pw, ph)
                self.embedded = True
            except Exception as e:
                print(f"Could not embed the OpenGL view in the control window ({type(e).__name__}: {e}); using two windows.",
                      file=sys.stderr)
                try:
                    self.win.close()
                except Exception:
                    pass
                self.win = None
        if self.win is None:
            self.win = pyglet.window.Window(pw, ph, caption="Destruction lab", resizable=False, vsync=True)
        try:
            self.win.set_mouse_cursor(self.win.get_system_mouse_cursor(self.win.CURSOR_CROSSHAIR))
        except Exception:
            pass
        gl.glClearColor(0.05, 0.06, 0.08, 1.0)
        self.pbo = self.reg = self.tex = None
        self.interop = False
        self._buffers(w, h)
        self.batch = pyglet.graphics.Batch()
        self.layers = [pyglet.graphics.Group(order=n) for n in range(3)]      # 0 backgrounds, 1 shapes, 2 text
        self.static, self.dynamic = [], []
        self.panel, self.panel_tex = False, None              # the performance graphs (a small RGB picture)
        self.win.push_handlers(self)

    # ---- the picture ----------------------------------------------------------------------------------------
    def _register(self):
        """Register the pixel buffer with CUDA.  (Its own method so a test can substitute a stand-in.)"""
        return wp.RegisteredGLBuffer(int(self.pbo.value), self.dev, wp.RegisteredGLBuffer.WRITE_DISCARD)

    def _buffers(self, w, h):
        """(Re)create the pixel buffer, its CUDA registration and the texture for a w x h picture."""
        gl = self.gl
        self.win.switch_to()
        self.reg = None                                       # unregisters the old buffer before it is deleted
        if self.pbo is not None:
            gl.glDeleteBuffers(1, self.pbo)
        if self.tex is not None:
            try:
                self.tex.delete()
            except Exception:
                pass
        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, w * h * 3, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.reg = self._register()
        self.interop = self.reg.resource is not None
        self.tex = self.pyglet.image.Texture.create(w, h, internalformat=gl.GL_RGB8, fmt=gl.GL_RGB,
                                                    min_filter=gl.GL_NEAREST, mag_filter=gl.GL_NEAREST)
        self.w, self.h = w, h

    def resize(self, w, h, zoom, size=None):
        self.zoom = zoom
        if (w, h) != (self.w, self.h):
            self._buffers(w, h)
        pw, ph = size or (max(64, int(round(w * zoom))), max(64, int(round(h * zoom))))
        if self.embedded:
            self.place_child(pw, ph)
        elif not self.win.fullscreen:
            self.win.set_size(pw, ph)

    def _embed(self, host, pw, ph):
        """Make the pyglet window a child of a Tk frame, so the picture and the controls are one window."""
        if sys.platform != "win32":
            raise RuntimeError("embedding is Windows-only")
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        u.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
        u.SetParent.restype = wintypes.HWND
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetWindowLongW.restype = ctypes.c_long
        u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        u.SetWindowLongW.restype = ctypes.c_long
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        u.SetFocus.argtypes = [wintypes.HWND]
        hwnd = self.win._hwnd
        parent = host.winfo_id()
        style = u.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
        style = (style & ~0x80000000 & ~0x00C00000) | 0x40000000 | 0x10000000 | 0x04000000     # not popup / caption; child, visible, clip siblings
        u.SetWindowLongW(hwnd, -16, style)                  # style first, then the parent (Windows' documented order)
        u.SetParent(hwnd, parent)
        u.SetWindowPos(hwnd, None, 0, 0, pw, ph, 0x0004 | 0x0020 | 0x0040)     # no z-order change, frame changed, show
        self._u, self._hwnd = u, hwnd

    def place_child(self, pw, ph):
        self._u.SetWindowPos(self._hwnd, None, 0, 0, int(pw), int(ph), 0x0004 | 0x0010)

    def set_vsync(self, on):
        try:
            self.win.set_vsync(bool(on))
        except Exception:
            pass

    def draw_picture(self, render_fn):
        """render_fn(target) must write the w*h*3 picture bytes (bottom row first) into the Warp array it is given.
        Then the buffer is handed back to OpenGL and copied into the texture (all on the GPU)."""
        gl = self.gl
        self.win.switch_to()
        target = self.reg.map(wp.uint8, (self.w * self.h * 3,))
        try:
            render_fn(target)
            if self.sync_before_unmap:
                wp.synchronize_device(self.dev)
        finally:
            self.reg.unmap()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBindTexture(self.tex.target, self.tex.id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)             # rows are w*3 bytes: not always a multiple of 4
        gl.glTexSubImage2D(self.tex.target, 0, 0, 0, self.w, self.h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 0)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

    def close(self):
        self.reg = None
        try:
            self.win.close()
        except Exception:
            pass

    # ---- overlays: the Tk canvas items, drawn with pyglet ---------------------------------------------------
    def bind_items(self):
        """Pick the canvas items to mirror.  Static ones (stats, help, legend) are re-read only when LabApp says they
        changed; dynamic ones (beam, well ring, wind arrow) every frame."""
        a, P = self.app, self.pyglet
        mono, sans = ["Consolas", "DejaVu Sans Mono", "Courier New"], ["Segoe UI", "DejaVu Sans", "Arial"]
        self.static = [self._item(a.ov_bg, "rect", 0), self._item(a.ov_text, "text", 2, mono, 10, False, True),
                       self._item(a.ov_help, "text", 2, sans, 9, False, False), self._item(a.leg_bg, "rect", 0)]
        self.static += [self._item(r, "rect", 1) for r in a.leg_rects]
        self.static += [self._item(a.leg_title, "text", 2, sans, 9, True, False), self._item(a.leg_lo, "text", 2, sans, 9, False, False),
                        self._item(a.leg_hi, "text", 2, sans, 9, False, False)]
        self.dynamic = [self._item(a.slice_line, "line", 1), self._item(a.beam_glow, "line", 1), self._item(a.beam, "line", 1),
                        self._item(a.well_dot, "oval", 1), self._item(a.wind_arrow, "arrow", 1),
                        self._item(a.wind_text, "text", 2, sans, 10, True, False)]
        self.static += [self._item(cid, "text", 2, mono, 9, False, False) for cid in a.perf_txt.values()]
        self.ov_bg_item, self.ov_text_item = self.static[0], self.static[1]

    def set_panel(self, img):
        """Upload the performance picture (an (h, w, 3) uint8 array, top row first) and show it at the top left."""
        P, gl = self.pyglet, self.gl
        self.win.switch_to()
        h, w = img.shape[:2]
        if self.panel_tex is None or (self.panel_tex.width, self.panel_tex.height) != (w, h):
            self.panel_tex = P.image.Texture.create(w, h, internalformat=gl.GL_RGB8, fmt=gl.GL_RGB,
                                                    min_filter=gl.GL_NEAREST, mag_filter=gl.GL_NEAREST)
        self.panel_tex.blit_into(P.image.ImageData(w, h, "RGB", img.tobytes(), pitch=-w * 3), 0, 0, 0)
        self.panel = True

    def _item(self, cid, kind, layer, fonts=None, size=9, bold=False, multiline=False):
        """One mirrored canvas item and the pyglet objects that draw it (created hidden)."""
        from pyglet import shapes, text
        grp = self.layers[layer]
        if kind == "rect":
            objs = [shapes.Rectangle(0, 0, 1, 1, batch=self.batch, group=grp)]
        elif kind == "text":
            kw = dict(multiline=True, width=1200) if multiline else {}
            objs = [text.Label("", 0, 0, font_name=fonts, font_size=size, weight="bold" if bold else "normal",
                               anchor_x="left", anchor_y="top", batch=self.batch, group=grp, **kw)]
        else:
            n = {"line": 1, "arrow": 3, "oval": 24}[kind]
            objs = [shapes.Line(0, 0, 0, 0, batch=self.batch, group=grp) for _ in range(n)]
        for o in objs:
            o.visible = False
        return {"id": cid, "kind": kind, "objs": objs, "sig": None}

    @staticmethod
    def _anchor(a):
        """Tk anchor ('nw', 'se', ...) -> pyglet (anchor_x, anchor_y)."""
        ax = "right" if "e" in a else ("left" if "w" in a else "center")
        ay = "top" if "n" in a else ("bottom" if "s" in a else "center")
        return ax, ay

    def _rgb(self, s):
        if len(s) == 7 and s[0] == "#":
            return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
        r, g, b = self.app.root.winfo_rgb(s)                    # a colour name
        return r >> 8, g >> 8, b >> 8

    def _sync(self, it):
        """Bring one mirrored item up to date with its Tk canvas item (does nothing if the canvas item is unchanged)."""
        c, H = self.app.canvas, self.win.height
        cid, kind, objs = it["id"], it["kind"], it["objs"]
        hidden = c.itemcget(cid, "state") == "hidden"
        co = tuple(c.coords(cid))
        colour = c.itemcget(cid, "outline" if kind == "oval" else "fill")
        txt = c.itemcget(cid, "text") if kind == "text" else ""
        extra = c.itemcget(cid, "anchor") if kind == "text" else (
            c.itemcget(cid, "stipple") if kind == "rect" else c.itemcget(cid, "width"))
        sig = (hidden, co, colour, txt, extra, H)
        if sig == it["sig"]:
            return
        it["sig"] = sig
        if hidden or not colour and kind != "rect":
            for o in objs:
                o.visible = False
            return
        rgb = self._rgb(colour) if colour else (0, 0, 0)
        if kind == "rect":
            o = objs[0]
            x0, y0, x1, y1 = co
            o.x, o.y, o.width, o.height = x0, H - y1, max(x1 - x0, 0.0), max(y1 - y0, 0.0)
            o.color = rgb
            o.opacity = 140 if extra else 255                   # a stippled Tk rectangle is half see-through
            o.visible = True
        elif kind == "text":
            o = objs[0]
            o.text = txt
            o.x, o.y = co[0], H - co[1]
            o.anchor_x, o.anchor_y = self._anchor(extra)
            o.color = (*rgb, 255)
            o.visible = True
        elif kind == "oval":                                    # a ring made of short lines
            x0, y0, x1, y1 = co
            cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            n, th = len(objs), float(extra or 1)
            for k, o in enumerate(objs):
                a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
                o.x, o.y = cx + rx * math.cos(a0), H - (cy + ry * math.sin(a0))
                o.x2, o.y2 = cx + rx * math.cos(a1), H - (cy + ry * math.sin(a1))
                o.thickness, o.color, o.visible = th, rgb, True
        else:                                                   # line (or line with an arrow head)
            th = float(extra or 1)
            x0, y0, x1, y1 = co
            main = objs[0]
            main.x, main.y, main.x2, main.y2 = x0, H - y0, x1, H - y1
            main.thickness, main.color, main.visible = th, rgb, True
            if kind == "arrow":
                ang = math.atan2(y1 - y0, x1 - x0)
                for o, s in zip(objs[1:], (0.45, -0.45)):
                    bx, by = x1 - 12 * math.cos(ang + s), y1 - 12 * math.sin(ang + s)
                    o.x, o.y, o.x2, o.y2 = x1, H - y1, bx, H - by
                    o.thickness, o.color, o.visible = th, rgb, True

    def draw_frame(self, static_dirty):
        """Clear, draw the picture and the overlays.  (The caller flips.)"""
        for it in self.dynamic:
            self._sync(it)
        if static_dirty:
            for it in self.static:
                self._sync(it)
            lab, bg = self.ov_text_item["objs"][0], self.ov_bg_item["objs"][0]      # the stats panel's backdrop fits the text
            if lab.visible:
                bg.x, bg.y = lab.x - 8, lab.y - lab.content_height - 6
                bg.width, bg.height = lab.content_width + 16, lab.content_height + 12
                bg.color, bg.opacity, bg.visible = self._rgb("#0d1015"), 140, True
        self.win.switch_to()
        self.win.clear()
        self.tex.blit(0, 0, width=self.win.width, height=self.win.height)          # the picture always fills the window
        if self.panel and self.panel_tex is not None:
            self.panel_tex.blit(14, self.win.height - 12 - self.panel_tex.height)
        self.batch.draw()

    # ---- window events -> the same handlers the Tk canvas calls -----------------------------------------------
    def _ev(self, x, y):
        from types import SimpleNamespace
        return SimpleNamespace(x=int(x / self.zoom), y=int((self.win.height - y) / self.zoom))

    def on_mouse_press(self, x, y, button, modifiers):
        if self.embedded:
            self._u.SetFocus(self._hwnd)                      # a child window does not take the keyboard by itself
        m = self.pyglet.window.mouse
        if button == m.LEFT:
            self.app.on_press(self._ev(x, y))
        elif button == m.RIGHT:
            self.app.orbit_from = (int(x), self.win.height - int(y))

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self.pyglet.window.mouse.LEFT:
            self.app.on_release(self._ev(x, y))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        from types import SimpleNamespace
        m = self.pyglet.window.mouse
        if buttons & m.LEFT:
            self.app.on_drag(self._ev(x, y))
        if buttons & m.RIGHT:
            self.app.on_orbit(SimpleNamespace(x=int(x), y=self.win.height - int(y)))      # orbiting works in screen pixels

    def on_mouse_motion(self, x, y, dx, dy):
        self.app.on_move(self._ev(x, y))

    def on_mouse_leave(self, x, y):
        self.app.hover = None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y:
            self.app.cam.zoom(1 if scroll_y > 0 else -1)

    def on_key_press(self, symbol, modifiers):
        K, a = self.pyglet.window.key, self.app
        if symbol == K.V:
            a.cycle_view()
        elif symbol == K.SPACE:
            a.toggle_pause()
        elif symbol == K.R:
            a.place()
        elif symbol == K.F11:
            a.toggle_fullscreen()
        elif symbol == K.ESCAPE:
            a.toggle_fullscreen(False)                           # (pyglet would close the window on Escape)
        elif symbol == K.TAB:
            a.toggle_panel()
        elif symbol == K.G:
            a.toggle_perf()
        elif symbol == K.Q:
            a.tool_menu_toggle()
        elif symbol == K.E:
            a.cond_toggle()
        else:
            name = {K.W: "w", K.A: "a", K.S: "s", K.D: "d", K.F: "f", K.C: "c"}.get(symbol)
            if name:
                a.key_down(name)
        return self.pyglet.event.EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        K = self.pyglet.window.key
        name = {K.W: "w", K.A: "a", K.S: "s", K.D: "d", K.F: "f", K.C: "c"}.get(symbol)
        if name:
            self.app.key_up(name)

    def on_close(self):
        self.app.stop_gl()


class LabAppGL(LabApp):
    """The Destruction lab with the picture in an OpenGL window (GLView).  The controls stay in the Tk window; the
    frame loop is the same one LabApp uses (_frame), run from a loop that serves both windows."""

    def __init__(self, root, device=None, view=(1280, 720)):
        self.running, self.gl_dirty, self.glv = True, True, None
        super().__init__(root, device, view)
        root.update_idletasks()
        print("[lab] controls built; creating the OpenGL view ...", flush=True)
        self.glv = GLView(self, self.lab.w, self.lab.h, 1, host=self.host if self.EMBED else None)
        print(f"[lab] OpenGL view created (embedded={self.glv.embedded}, zero-copy={self.glv.interop})", flush=True)
        self.glv.bind_items()
        if self.glv.embedded:
            root.title("Destruction lab")
        else:
            self.host.grid_remove()
            root.title("Destruction lab - controls")
        root.protocol("WM_DELETE_WINDOW", self.stop_gl)
        print("[lab] sizing the view ...", flush=True)
        self.apply_view()
        self.fps_changed(quiet=True)
        print("[lab] ready, entering the frame loop", flush=True)

    def _ui(self):
        super()._ui()
        self.canvas.grid_remove()          # the picture lives in the GL window; the canvas only holds the overlay items
        self.host = tk.Frame(self.root, width=self.lab.w, height=self.lab.h, bg=SV_BG, bd=0, highlightthickness=0)
        self.host.grid(row=0, column=0, sticky="nw")      # the GL window is placed inside this frame (see GLView._embed)

    def tick(self):                        # LabApp.__init__ schedules this once; the GL loop calls _frame() itself
        pass

    EMBED = True                           # False (--no-embed): keep the OpenGL picture in its own window
    loop_on = False

    def stop_gl(self):
        self.running = False
        if not self.loop_on:                 # still starting up (or stuck): there is no loop to notice, so quit now
            os._exit(0)

    def _pump_tk(self, limit=300):
        """root.update(), but bounded: if something keeps queuing Tk events faster than they are handled, update() never
        returns and the whole program looks frozen.  Handle at most `limit` events per frame and say what is queuing."""
        import _tkinter
        tkc = self.root.tk
        self.root.update_idletasks()
        n = 0
        while n < limit and tkc.dooneevent(_tkinter.DONT_WAIT):
            n += 1
        if n >= limit:
            self._storms = getattr(self, "_storms", 0) + 1
            if self._storms in (1, 50, 500):
                try:
                    pend = tkc.call("after", "info")
                except Exception:
                    pend = "?"
                print(f"[lab] Tk event storm #{self._storms}: {limit} events in one frame; pending after-callbacks: {pend}", flush=True)
                for p in pend if isinstance(pend, (tuple, list)) else ():
                    try:
                        print("[lab]    ", p, "->", tkc.call("after", "info", p), flush=True)
                    except Exception:
                        pass

    def run_gl(self):
        glv, root, pyglet = self.glv, self.root, self.glv.pyglet
        self.loop_on = True
        n_frames = 0
        timer = None
        try:                                                    # finer sleep() so 100 / 120 fps caps are accurate
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            timer = ctypes.windll.winmm
        except Exception:
            pass
        try:
            while self.running:
                t0 = time.perf_counter()
                pyglet.clock.tick()
                glv.win.switch_to()
                glv.win.dispatch_events()
                if not self.running:
                    break
                try:
                    self._pump_tk()
                except tk.TclError:
                    break
                self.perf.add("tk", time.perf_counter() - t0)
                if n_frames == 0:
                    print("[lab] first frame: physics and picture (may compile kernels) ...", flush=True)
                self._frame()
                if n_frames == 0:
                    print("[lab] first frame: drawing to the window ...", flush=True)
                t1 = time.perf_counter()
                glv.draw_frame(self.gl_dirty)
                self.gl_dirty = False
                glv.win.flip()                                  # (with vsync this waits for the screen)
                if n_frames == 0:
                    print("[lab] first frame done", flush=True)
                if n_frames % 8 == 0:
                    import faulthandler
                    faulthandler.dump_traceback_later(10, repeat=True)     # a 10 s stall prints where the program is stuck
                n_frames += 1
                self.perf.add("flip", time.perf_counter() - t1)
                if self.fps_cap:
                    t2 = time.perf_counter()
                    self._pace(t0)
                    self.perf.add("wait", time.perf_counter() - t2)         # deliberate pacing, not work
        finally:
            if timer is not None:
                try:
                    timer.timeEndPeriod(1)
                except Exception:
                    pass
            glv.close()
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _pace(self, t0):
        """Hold the frame to 1 / fps_cap seconds: sleep most of the wait, spin the last millisecond or two."""
        end = t0 + 1.0 / self.fps_cap
        while True:
            rest = end - time.perf_counter()
            if rest <= 0.0:
                return
            if rest > 0.002:
                time.sleep(rest - 0.0015)

    def _draw_picture(self, radius, colour):
        lab, cam, hover, view = self.lab, self.cam, self.hover, self.view
        self.glv.draw_picture(lambda t: lab.render(cam, hover, radius, colour, view, target=t))

    def update_legend(self):
        super().update_legend()
        self.gl_dirty = True

    def show_stats(self, idle=False, st=None):
        super().show_stats(idle, st)
        self.gl_dirty = True

    def _perf_show(self, img):                  # the graphs go straight into an OpenGL texture, not a Tk image
        self.glv.set_panel(img)
        self.gl_dirty = True

    def _perf_hide(self):
        super()._perf_hide()
        self.glv.panel = False
        self.gl_dirty = True

    def _timing_lines(self):
        how = ("OpenGL, written straight from the GPU (no host copy)" if self.glv.interop else
               "OpenGL, COPY FALLBACK (the window is not on the CUDA GPU)")
        if self.perf_on.get() and self.glv.interop:
            return super()._timing_lines()
        return super()._timing_lines() + (f"Display  {how}",)

    def apply_view(self):
        """Rendered size from the chosen resolution; in fullscreen the GL window covers the whole screen."""
        glv = self.glv
        if glv.embedded:
            return self._apply_view_embedded()
        if self.full:
            sw, sh = glv.win.screen.width, glv.win.screen.height
            h = max(64, min(sh, lab_res_size(self.res_v.get())[1]))
            w = max(64, int(round(h * sw / float(sh))))                 # same shape as the screen
            z = sh / float(h)                                           # OpenGL stretches by any factor
        else:
            w, h = lab_res_size(self.res_v.get())
            sw, sh = glv.win.screen.width, glv.win.screen.height
            # a window can't be bigger than the screen's usable area (title bar, taskbar): shrink the display, not the picture
            pw_ = 310 if self.panel_on else 0                             # room for the separate controls window
            z = min(1.0, (sw - 16 - pw_) / float(w), (sh - 110) / float(h))
            z = max(z, min(1.0, 0.6 * (sw - 16) / float(w)))              # never collapse the picture to a thumbnail
        self.zoom = z
        self.lab.resize(w, h)
        glv.resize(w, h, z)
        if not self.full:
            self._fit_gl_window()
        self.canvas.coords(self.ov_help, 14, h * z - 10)
        self.canvas.coords(self.wind_text, w * z - 16, 100)
        self.update_legend()
        self.wake()

    def _fit_gl_window(self):
        """Keep the (separate) GL window fully on screen, to the right of the controls window when there is room."""
        try:
            win = self.glv.win
            sw, sh = win.screen.width, win.screen.height
            ww, wh = win.get_size()
            x, y = win.get_location()
            left = 310 if self.panel_on else 0                                    # just right of the controls window
            x = max(left, 0) if left + ww <= sw else sw - ww
            x = max(0, min(x, sw - ww))
            y = max(34, min(y, sh - wh - 48))                                      # title bar above, taskbar below
            win.set_location(int(x), int(y))
        except Exception:
            pass

    def _apply_view_embedded(self):
        """Picture size when the GL view sits inside the Tk window next to the controls."""
        glv = self.glv
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        pw = self.side.winfo_reqwidth() if self.panel_on else 0
        if self.full:
            aw, ah = max(320, sw - pw), sh
            h = max(64, min(ah, lab_res_size(self.res_v.get())[1]))
            w = max(64, int(round(h * aw / float(ah))))
            z = ah / float(h)
            hw_, hh_ = aw, ah
            self.side_canvas.configure(height=ah)
        else:
            w, h = lab_res_size(self.res_v.get())
            z = min(1.0, (sw - 16 - pw) / float(w), (sh - 110) / float(h))
            hw_, hh_ = max(64, int(round(w * z))), max(64, int(round(h * z)))
            self._side_configure()
        self.zoom = z
        print(f"[lab]   view {w} x {h}, shown at {hw_} x {hh_}: host ...", flush=True)
        self.host.configure(width=hw_, height=hh_)
        print("[lab]   renderer buffers ...", flush=True)
        self.lab.resize(w, h)
        print("[lab]   GL buffers ...", flush=True)
        glv.resize(w, h, z, (hw_, hh_))
        print("[lab]   GL buffers done", flush=True)
        self.canvas.coords(self.ov_help, 14, hh_ - 10)
        self.canvas.coords(self.wind_text, hw_ - 16, 100)
        self.update_legend()
        self.wake()

    def toggle_fullscreen(self, on=None):
        on = (not self.full) if on is None else bool(on)
        if on == self.full:
            return
        if self.glv.embedded:                      # the whole Tk window goes fullscreen, the picture fills what the panel leaves
            LabApp.toggle_fullscreen(self, on)
            return
        self.full = on
        self.glv.win.set_fullscreen(on)
        self.res_cb.configure(values=LAB_FULL_MODES if on else LAB_RESOLUTIONS)
        self.res_v.set(self.res_full if on else self.res_win)
        self.full_b.configure(text="Exit fullscreen   (Esc)" if on else "Fullscreen   (F11)")
        self.apply_view()

    def toggle_panel(self):
        """Tab: hide or show the controls (handy over a fullscreen picture)."""
        if self.glv.embedded:
            self.panel_on = not self.panel_on
            if self.panel_on:
                self.side.grid(row=0, column=1, sticky="ns")
            else:
                self.side.grid_remove()
            self.root.update_idletasks()
            self.apply_view()
            if not self.full:
                self.root.geometry("")
            return
        self.panel_on = not self.panel_on
        if self.panel_on:
            self.root.deiconify()
        else:
            self.root.withdraw()


def lab_main(gl=False, embed=True):
    if tk is None or ttk is None:
        sys.exit("tkinter is not available in this Python, so the lab window cannot open.")
    root = tk.Tk()
    why = None                                                  # why the OpenGL display could not be used (shown in the window)
    if gl:
        app = None
        try:
            if not hasattr(wp, "RegisteredGLBuffer"):
                raise RuntimeError("this Warp version has no RegisteredGLBuffer")
            import pyglet                                       # noqa: F401
            import faulthandler
            faulthandler.dump_traceback_later(25, repeat=True)      # if startup stalls, print where (to the console) every 25 s
            LabAppGL.EMBED = embed
            app = LabAppGL(root)
        except Exception as e:                                  # missing pyglet, no OpenGL, ...: use the Tk window
            why = f"{type(e).__name__}: {e}"
            print(f"OpenGL display unavailable ({type(e).__name__}: {e}); using the Tk display.  "
                  f"(For the GL display:  pip install pyglet)", file=sys.stderr)
            try:
                root.destroy()
            except tk.TclError:
                pass
            root = tk.Tk()
        if app is not None:
            app.run_gl()                                        # errors while running are not hidden by a fallback
            return
    app = LabApp(root)
    if why:
        app.display_note = f"Tk display: OpenGL unavailable ({why[:70]})  - try: pip install pyglet"
        app.msg.set("OpenGL could not start, so this is the slower Tk display.")
    root.mainloop()


if __name__ == "__main__":
    # python physics_engine.py                           open the Destruction lab through OpenGL (the default)
    # python physics_engine.py --lab [--tk] [--embed]    same;  --tk: the slower Tk display, --embed: GL picture inside the Tk window
    # python physics_engine.py --viewer [state.json]     open the State Viewer
    # python physics_engine.py --gguf model.gguf ...     run a language model (see PART 3)
    args = sys.argv[1:]
    if not args or args[0] in ("--lab", "--build"):      # --build is still accepted but no longer does anything
        # OpenGL is the default display; --gl and --no-embed are still accepted (lab_gl.bat passes them) and now change nothing
        lab_main(gl="--tk" not in args, embed="--embed" in args)
    elif args and (args[0] == "--gguf" or args[0].lower().endswith(".gguf")):
        gguf_main(args[1:] if args[0] == "--gguf" else args)
    else:
        if args and args[0] == "--viewer":
            sys.argv.pop(1)
        if tk is None or ttk is None:
            sys.exit("tkinter is not available in this Python, so the State Viewer window cannot open.")
        viewer_main()
