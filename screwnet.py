# screwnet_axis_keyboard_lm.py
# Run: manimgl screwnet_axis_keyboard_lm.py ScrewNetAxisKeyboardLM

from manimlib import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# Utils
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.array(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else (v / n)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def skew(w: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=float).reshape(3)
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0],
    ], dtype=float)

def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    a = normalize(axis)
    K = skew(a)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def perp_basis(l_unit: np.ndarray, u_ref: np.ndarray = np.array([1.0, 0.0, 0.0])):
    """
    Build (u0,v0) orthonormal basis perpendicular to l_unit.
    """
    l = normalize(l_unit)
    u = np.array(u_ref, dtype=float).reshape(3)
    u = u - np.dot(u, l) * l
    if np.linalg.norm(u) < 1e-8:
        tmp = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(tmp, l)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        u = tmp - np.dot(tmp, l) * l
    u0 = normalize(u)
    v0 = normalize(np.cross(l, u0))
    return u0, v0

def se3_from_screw_displacement(l_unit: np.ndarray, m_orth: np.ndarray, theta: float, d: float) -> np.ndarray:
    """
    sigma=(l,m,theta,d) -> T in SE(3)
    Use p* = l x m (assuming ||l||=1 and l·m=0).
    x' = R x + (p* - R p* + d l)
    """
    l = normalize(l_unit)
    p_star = np.cross(l, m_orth)
    R = rodrigues(l, theta)
    t = p_star - (R @ p_star) + d * l
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def pose_frame_mobject(pos: np.ndarray,
                       R: np.ndarray,
                       axis_len: float = 1.0,
                       stroke_width: float = 7,
                       opacity: float = 1.0) -> VGroup:
    pos = np.array(pos, dtype=float).reshape(3)
    R = np.array(R, dtype=float).reshape(3, 3)

    x_vec = Vector(axis_len * (R @ np.array([1.0, 0.0, 0.0]))).set_color(RED)
    y_vec = Vector(axis_len * (R @ np.array([0.0, 1.0, 0.0]))).set_color(GREEN)
    z_vec = Vector(axis_len * (R @ np.array([0.0, 0.0, 1.0]))).set_color(BLUE)

    for v in (x_vec, y_vec, z_vec):
        v.shift(pos)
        v.set_stroke(width=stroke_width)
        v.set_opacity(opacity)

    center = Dot(point=pos, radius=0.06).set_color(WHITE).set_opacity(opacity)
    return VGroup(x_vec, y_vec, z_vec, center)

def axis_arrow_mobject(l_unit: np.ndarray, p_on_axis: np.ndarray, length: float = 12.0, stroke_width: float = 12):
    l = normalize(l_unit)
    p = np.array(p_on_axis, dtype=float).reshape(3)
    start = p - 0.5 * length * l
    v = Vector(length * l).set_color(RED).set_stroke(width=stroke_width)
    v.shift(start)
    return v

# ============================================================
# Scene
# ============================================================
class ScrewNetAxisKeyboardLM(ThreeDScene):
    def setup_camera(self):
        if hasattr(self, "move_camera"):
            self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, run_time=0)

    def construct(self):
        self.setup_camera()
        self.add(ThreeDAxes())

        # ------------------------------------------------------------
        # Mode switch by keyboard (1~4)
        #   1: Locked / Rigid   (theta=0, d=0)
        #   2: Revolute         (theta=1, d=0)
        #   3: Prismatic        (theta=0, d=1)
        #   4: Screw            (theta=1, d=1)
        # ------------------------------------------------------------
        mode = ValueTracker(2.0)   # start as Revolute
        theta = ValueTracker(1.0)
        d = ValueTracker(0.0)
        self.add(mode, theta, d)

        def set_mode(k: int):
            mode.set_value(float(k))
            if k == 1:      # Locked / Rigid
                theta.set_value(0.0)
                d.set_value(0.0)
            elif k == 2:    # Revolute
                theta.set_value(1.0)
                d.set_value(0.0)
            elif k == 3:    # Prismatic
                theta.set_value(0.0)
                d.set_value(1.0)
            elif k == 4:    # Screw
                theta.set_value(1.0)
                d.set_value(1.0)

        # (선택) 시작 모드 강제 세팅
        set_mode(2)

        # ------------------------------------------------------------
        # Keyboard-controlled parameters to generate a VALID (l,m):
        #   l := unit direction from (yaw, pitch)
        #   p* := closest point to origin on axis, constrained p*·l = 0
        #   m := p* x l  (=> l·m = 0 automatically)
        # ------------------------------------------------------------
        
        yaw = ValueTracker(0.0)       # rotate around world Z
        pitch = ValueTracker(0.0)     # tilt up/down
        alpha = ValueTracker(0.0)     # p* coordinate along u0
        beta = ValueTracker(2.0)      # p* coordinate along v0
        self.add(yaw, pitch, alpha, beta)

        # Speeds / limits
        yaw_speed = 1.2       # rad/sec
        pitch_speed = 1.2     # rad/sec
        move_speed = 2.0      # units/sec for alpha/beta
        pitch_min = -1.45     # ~ -83 deg (avoid singular)
        pitch_max = 1.45

        alpha_min, alpha_max = -5.0, 5.0
        beta_min, beta_max = -5.0, 5.0

        # ------------------------------------------------------------
        # Key polling (pyglet)
        # ------------------------------------------------------------
        
        try:
            from pyglet.window import key as pyg_key
        except Exception:
            pyg_key = None

        controller = Mobject().set_opacity(0.0)

        def controller_updater(mob: Mobject, dt: float):
            if pyg_key is None:
                return
            win = getattr(self, "window", None)
            if win is None or (not hasattr(win, "is_key_pressed")):
                return

            def pressed(name: str) -> bool:
                k = getattr(pyg_key, name, None)
                return (k is not None) and win.is_key_pressed(k)

            # --- mode keys: 1~4 (top row) + NUM_1~NUM_4 (numpad) 지원 ---
            if pressed("_1") or pressed("NUM_1"):
                set_mode(1)
            elif pressed("_2") or pressed("NUM_2"):
                set_mode(2)
            elif pressed("_3") or pressed("NUM_3"):
                set_mode(3)
            elif pressed("_4") or pressed("NUM_4"):
                set_mode(4)

            # l orientation: J/L yaw, I/K pitch
            dy = 0.0
            dp = 0.0
            if pressed("L"):
                dy += yaw_speed * dt
            if pressed("J"):
                dy -= yaw_speed * dt
            if pressed("I"):
                dp += pitch_speed * dt
            if pressed("K"):
                dp -= pitch_speed * dt

            if dy != 0.0:
                yaw.set_value(yaw.get_value() + dy)
            if dp != 0.0:
                pitch.set_value(clamp(pitch.get_value() + dp, pitch_min, pitch_max))

            # axis position (through p*): WASD changes alpha/beta in the ⟂ plane
            da = 0.0
            db = 0.0
            if pressed("C"):
                da += move_speed * dt
            if pressed("Z"):
                da -= move_speed * dt
            if pressed("S"):
                db += move_speed * dt
            if pressed("X"):
                db -= move_speed * dt

            if da != 0.0:
                alpha.set_value(clamp(alpha.get_value() + da, alpha_min, alpha_max))
            if db != 0.0:
                beta.set_value(clamp(beta.get_value() + db, beta_min, beta_max))

        controller.add_updater(controller_updater)
        self.add(controller)

        # yaw, pitch, alpha, beta 만든 직후에 배치하세요.

        axis_cache = {"l": None, "u0": None, "v0": None, "p_star": None, "m": None}

        def update_axis_cache(mob=None, dt=0.0):
            y = yaw.get_value()
            p = pitch.get_value()

            cy, sy = np.cos(y), np.sin(y)
            cp, sp = np.cos(p), np.sin(p)

            l = np.array([cy * cp, sy * cp, sp], dtype=float)
            u0, v0 = perp_basis(l, u_ref=np.array([1.0, 0.0, 0.0]))

            a = alpha.get_value()
            b = beta.get_value()
            p_star = a * u0 + b * v0

            m = np.cross(p_star, l)

            axis_cache.update(l=l, u0=u0, v0=v0, p_star=p_star, m=m)

        def axis_state():
            return axis_cache["l"], axis_cache["u0"], axis_cache["v0"], axis_cache["p_star"], axis_cache["m"]

        # 초기값 1회 채우기 (always_redraw가 돌기 전에!)
        update_axis_cache()

        cache_driver = Mobject().set_opacity(0)
        cache_driver.add_updater(update_axis_cache)
        self.add(cache_driver)

        def axis_line_mobj():
            l, u0, v0, p_star, m = axis_state()
            return Line(p_star - 7.0 * l, p_star + 7.0 * l).set_color(GREY_B).set_stroke(width=3, opacity=0.6)

        axis_line = always_redraw(axis_line_mobj)
        self.add(axis_line)

        # ------------------------------------------------------------
        # Visualize axis (red arrow) from live (l,m)
        # ------------------------------------------------------------
 
        axis_arrow = always_redraw(lambda: axis_arrow_mobject(
            axis_state()[0],
            axis_state()[3],
            length=12.0,
            stroke_width=12
        ))

        pstar_dot = always_redraw(lambda: Dot(
            point=axis_state()[3],
            radius=0.06
        ).set_color(WHITE).set_opacity(0.9))

        self.add(axis_arrow, pstar_dot)

        # ------------------------------------------------------------
        # Triad at screw-axis "origin" p* (closest point to world origin)
        # Frame axes: x=u0, y=v0, z=l
        # ------------------------------------------------------------

        def axis_triad_mobj():
            l, u0, v0, p_star, m = axis_state()
            R_axis = np.column_stack([u0, v0, l])  # 3x3, right-handed orthonormal
            return pose_frame_mobject(p_star, R_axis, axis_len=0.9, stroke_width=7)

        axis_triad = always_redraw(axis_triad_mobj)
        self.add(axis_triad)


        # ------------------------------------------------------------
        # Triad at world origin (0,0,0) with world-aligned axes
        # ------------------------------------------------------------

        origin_triad = pose_frame_mobject(
            pos=np.array([0.0, 0.0, 0.0]),
            R=np.eye(3),
            axis_len=0.9,
            stroke_width=7,
            opacity=0.9
        )
        self.add(origin_triad)

        # ------------------------------------------------------------
        # Red ball following the mouse pointer (world position)
        # ------------------------------------------------------------
        
        red_ball = Sphere(radius=0.12, resolution=(18, 36))
        red_ball.set_color(RED)
        red_ball.set_opacity(0.9)

        red_ball.add_updater(lambda m: m.move_to(self.mouse_point.get_center()))
        self.add(red_ball)

        # ------------------------------------------------------------
        # Circle manifold (revolute only): slice of the cylinder at fixed u
        # C(v) = p* + u_slice*l + r(cos v u0 + sin v v0)
        # ------------------------------------------------------------
        
        r_cir = 2.0
        u_slice = ValueTracker(0.0)   # 원이 축을 따라 어디에 위치할지 (원하면 키로 움직이게 확장 가능)
        self.add(u_slice)

        circle_step = TAU / 120  # 240 샘플 정도

        def circle_manifold_mobj():
            l, u0, v0, p_star, m = axis_state()
            u = u_slice.get_value()

            circle = ParametricCurve(
                lambda v: p_star + u * l + r_cir * (np.cos(v) * u0 + np.sin(v) * v0),
                t_range=(0.0, TAU, circle_step),   # <- (min, max, step) 로 변경
            ).set_stroke(color=GREY_B, width=4)

            visible = (abs(d.get_value()) < 1e-9) and (abs(theta.get_value()) > 1e-9)
            circle.set_opacity(0.8 if visible else 0.0)

            return circle

        circle_manifold = always_redraw(circle_manifold_mobj)
        self.add(circle_manifold)



        # ------------------------------------------------------------
        # Unified guiding vector field for all modes (1~4)
        # ------------------------------------------------------------

        k_t = 2.0      # tangential gain
        k_r = 4.0      # radial attraction gain
        k_u = 3.0      # u-slice attraction (for revolute)
        k_sink = 2.5   # sink gain (for locked)

        omega = 1.0    # "flow speed" around/along
        arrow_scale = 0.35
        arrow_sw = 3

        # 공통 샘플: axis 좌표계에서 (u, v, dr)
        u_vals  = np.linspace(-3.0, 3.0, 4)
        v_vals  = np.linspace(0.0, TAU, 16, endpoint=False)
        dr_vals = [0.0, 0.6, 1.2]   # signed 대신 radius 느낌으로
        # dr_vals = [0.0, 0.8]                          # 3 -> 2

        field_params = [(u, v, dr) for u in u_vals for v in v_vals for dr in dr_vals]

        guiding_field = VGroup(*[
            Arrow(ORIGIN, RIGHT*1e-3, buff=0).set_stroke(width=arrow_sw)
            for _ in field_params
        ])


        def guiding_field_updater(group: VGroup, dt: float):
            l, u0, v0, p_star, m = axis_state()  # 아래 2)에서 캐시 추천
            k = int(round(mode.get_value()))

            # pitch for screw
            th = theta.get_value()
            dd = d.get_value()
            h = (dd / th) if abs(th) > 1e-9 else 0.0  # prismatic/locked에서는 의미 없음

            # 모드별 opacity (원하면 조절)
            if k == 1:
                op = 0.55
            elif k == 2:
                op = 0.75   # <- Revolute에서도 통합 필드 보이게
            elif k == 3:
                op = 0.75
            elif k == 4:
                op = 0.75
            else:
                op = 0.0

            u_slice_val = u_slice.get_value()

            for mob, (u, v, dr) in zip(group, field_params):
                rdir = np.cos(v) * u0 + np.sin(v) * v0
                tdir = np.cross(l, rdir)  # circle/cylinder tangent direction

                # ----------------------------
                # Mode 1: Locked (sink to p_star)
                # ----------------------------
                if k == 1:
                    pos  = p_star + u * l + dr * rdir
                    vvec = -k_sink * (pos - p_star)

                # ----------------------------
                # Mode 2: Revolute (circle at u=u_slice, rho=r_cir)
                # ----------------------------
                elif k == 2:
                    pos  = p_star + u * l + (r_cir + dr) * rdir

                    # tangent around circle
                    v_tan = omega * (r_cir * tdir)

                    # attract to radius r_cir (dr -> 0)
                    v_rad = -k_r * dr * rdir

                    # attract to slice u_slice (u -> u_slice)
                    v_slc = -k_u * (u - u_slice_val) * l

                    vvec = v_tan + v_rad + v_slc

                # ----------------------------
                # Mode 3: Prismatic (line: rho=0, flow along axis)
                # ----------------------------
                elif k == 3:
                    pos  = p_star + u * l + dr * rdir

                    # flow along axis
                    v_ax  = omega * l

                    # attract to axis (dr -> 0)
                    v_rad = -k_r * dr * rdir

                    vvec = v_ax + v_rad

                # ----------------------------
                # Mode 4: Screw (helix-ish: combine axial + tangential)
                # ----------------------------
                elif k == 4:
                    pos  = p_star + u * l + (r_cir + dr) * rdir

                    # helix tangent direction ~ r*tdir + h*l
                    v_tan = omega * (r_cir * tdir + h * l)

                    # attract to radius r_cir
                    v_rad = -k_r * dr * rdir

                    #   u ≈ h*v + c  (c=0 가정)
                    # v_phase = -(u - h*v) * l  형태로 u를 맞춰도 꽤 잘 붙습니다.
                    v_phase = -0.8 * (u - h * v) * l if abs(h) > 1e-9 else 0.0 * l
                    # v_phase = 0.0 * l

                    vvec = v_tan + v_rad + v_phase

                else:
                    pos  = p_star
                    vvec = 1e-6 * u0

                mag = np.linalg.norm(vvec)
                if mag < 1e-9:
                    vvec = 1e-6 * u0
                    mag = np.linalg.norm(vvec)

                vdraw = arrow_scale * (vvec / mag)

                mob.put_start_and_end_on(pos, pos + vdraw)
                mob.set_opacity(op)

        guiding_field.add_updater(guiding_field_updater)
        self.add(guiding_field)


        # ------------------------------------------------------------
        # Mouse-follow triad blended with vector field dynamics
        # ------------------------------------------------------------
        triad_anchor = VectorizedPoint(self.mouse_point.get_center())
        self.add(triad_anchor)

        # 블렌딩/필드 영향 파라미터 (원하는 느낌으로 튜닝)
        k_mouse = 6.0   # 마우스 쪽으로 끌리는 속도(클수록 마우스를 더 잘 따라감)
        k_field = 1.0   # 벡터장 영향 스케일(클수록 벡터장에 더 끌림)
        v_max = 8.0     # 최대 속도 제한(폭주 방지)

        def velocity_field_at(x_world: np.ndarray) -> np.ndarray:
            x = np.array(x_world, dtype=float).reshape(3)

            l, u0, v0, p_star, _ = axis_state()
            k = int(round(mode.get_value()))

            th = theta.get_value()
            dd = d.get_value()
            h = (dd / th) if abs(th) > 1e-9 else 0.0  # screw pitch factor

            # axis frame decomposition
            r = x - p_star
            u = np.dot(r, l)
            r_perp = r - u * l
            rho = np.linalg.norm(r_perp)

            if rho < 1e-6:
                rdir = u0  # 임의의 수직 방향
            else:
                rdir = r_perp / rho

            tdir = np.cross(l, rdir)  # 둘레 접선 방향

            # 모드별 벡터장
            if k == 1:
                # Locked: p_star로 수렴
                v = -k_sink * (x - p_star)

            elif k == 2:
                # Revolute: 원(반경 r_cir, u=u_slice)으로 유도 + 둘레 접선 흐름
                dr = rho - r_cir
                v_tan = omega * (r_cir * tdir)
                v_rad = -k_r * dr * rdir
                v_slc = -k_u * (u - u_slice.get_value()) * l
                v = v_tan + v_rad + v_slc

            elif k == 3:
                # Prismatic: 축으로 유도 + 축 방향 흐름
                v_ax = omega * l
                v_rad = -k_r * rho * rdir
                v = v_ax + v_rad

            elif k == 4:
                # Screw: 둘레+축 혼합 접선(헬릭스 느낌) + 반경 유도
                dr = rho - r_cir
                v_tan = omega * (r_cir * tdir + h * l)
                v_rad = -k_r * dr * rdir
                v = v_tan + v_rad
                # (원하면 위상항 추가 가능하지만, 먼저는 꺼두는 걸 추천)
                # v += -0.8 * (u - h * ???) * l

            else:
                v = np.zeros(3)

            # 속도 제한
            speed = np.linalg.norm(v)
            if speed > v_max:
                v = (v_max / speed) * v
            return v

        def triad_anchor_updater(mob: Mobject, dt: float):
            pos = mob.get_center()
            mouse = self.mouse_point.get_center()

            # 1) 벡터장에 따라 한 스텝 적분 (Euler)
            v = velocity_field_at(pos)
            pos = pos + (k_field * v) * dt

            # 2) 마우스 목표로 1차 블렌딩(안정적)
            a = 1.0 - np.exp(-k_mouse * dt)  # dt에 대해 안정적인 blending factor
            pos = (1.0 - a) * pos + a * mouse

            mob.move_to(pos)

        triad_anchor.add_updater(triad_anchor_updater)

        blended_mouse_triad = always_redraw(
            lambda: pose_frame_mobject(
                pos=triad_anchor.get_center(),
                R=np.eye(3),
                axis_len=0.7,
                stroke_width=7,
                opacity=0.9
            )
        )
        self.add(blended_mouse_triad)


        def manifold_target_point(x_world: np.ndarray) -> np.ndarray:
            """
            현재 모드에 대응하는 manifold(구속면) 위에서
            x_world(마우스 포인트)를 가장 자연스럽게 대응시키는 목표점(target)을 반환.
            """
            x = np.array(x_world, dtype=float).reshape(3)

            l, u0, v0, p_star, _ = axis_state()
            k = int(round(mode.get_value()))

            # axis frame decomposition
            r = x - p_star
            u = float(np.dot(r, l))
            r_perp = r - u * l

            # 각도 v 계산(수직평면에서 u0, v0 좌표로)
            a = float(np.dot(r_perp, u0))
            b = float(np.dot(r_perp, v0))
            v = np.arctan2(b, a)  # [-pi, pi]
            if v < 0:
                v += TAU          # [0, 2pi)

            # ----------------------------
            # Mode 1: Locked -> 점 p_star
            # ----------------------------
            if k == 1:
                return p_star

            # ----------------------------
            # Mode 2: Revolute -> 원 (u = u_slice, rho = r_cir)
            # ----------------------------
            if k == 2:
                u_t = u_slice.get_value()
                return p_star + u_t * l + r_cir * (np.cos(v) * u0 + np.sin(v) * v0)

            # ----------------------------
            # Mode 3: Prismatic -> 축 직선 (rho = 0)
            # ----------------------------
            if k == 3:
                return p_star + u * l

            # ----------------------------
            # Mode 4: Screw -> 헬릭스 (rho = r_cir, u ≈ h * v)
            #   v는 마우스가 가리키는 각을 쓰고,
            #   회전수를 unwrap해서 u(마우스의 축방향 성분)에 가장 가깝게 맞춤.
            # ----------------------------
            if k == 4:
                th = theta.get_value()
                dd = d.get_value()
                h = (dd / th) if abs(th) > 1e-9 else 0.0

                # h가 0이면 screw가 붕괴하므로(실질적으로 revolute/cylinder),
                # 안전하게 원으로 fallback
                if abs(h) < 1e-9:
                    return p_star + u * l + r_cir * (np.cos(v) * u0 + np.sin(v) * v0)

                # v를 unwrap 해서 u/h에 가장 가까운 turn 선택
                kturn = int(np.round((u / h - v) / TAU))
                v_unwrap = v + TAU * kturn
                u_t = h * v_unwrap

                return p_star + u_t * l + r_cir * (np.cos(v_unwrap) * u0 + np.sin(v_unwrap) * v0)

            # fallback
            return p_star


        # ghost_triad = always_redraw(
        #     lambda: pose_frame_mobject(
        #         pos=manifold_target_point(self.mouse_point.get_center()),  # 목표점
        #         R=np.eye(3),
        #         axis_len=0.7,
        #         stroke_width=7,
        #         opacity=0.25,  # 반투명
        #     )
        # )
        # self.add(ghost_triad)


        ghost_triad = always_redraw(
            lambda: (lambda l,u0,v0,p_star,_:
                pose_frame_mobject(
                    pos=manifold_target_point(self.mouse_point.get_center()),
                    R=np.column_stack([u0, v0, l]),  # 축-기저 정렬
                    axis_len=0.7,
                    stroke_width=7,
                    opacity=0.25,
                )
            )(*axis_state())
        )
        self.add(ghost_triad)

        ghost_line = always_redraw(
            lambda: DashedLine(
                triad_anchor.get_center(),
                manifold_target_point(self.mouse_point.get_center()),
                dash_length=0.15,
            ).set_stroke(width=2, opacity=0.35)
        )
        self.add(ghost_line)

        # ------------------------------------------------------------
        # Real-time matplotlib plots for poses and errors
        # ------------------------------------------------------------
        class PosePlotter:
            def __init__(self, window_size=50):
                self.window_size = window_size
                self.t = deque(maxlen=window_size)
                self.red = [deque(maxlen=window_size) for _ in range(3)]
                self.proj = [deque(maxlen=window_size) for _ in range(3)]
                self.blend = [deque(maxlen=window_size) for _ in range(3)]
                self.err_red_proj = [deque(maxlen=window_size) for _ in range(3)]
                self.err_blend_proj = [deque(maxlen=window_size) for _ in range(3)]

                plt.ion()
                self.fig, self.axes = plt.subplots(5, 1, sharex=True, figsize=(9, 11))
                self.fig.suptitle("Pose Tracking (Realtime)")

                titles = [
                    "Red Ball Pose (XYZ)",
                    "Projection Triad Pose (XYZ)",
                    "Blended Triad Pose (XYZ)",
                    "Error: Red Ball - Projection (XYZ)",
                    "Error: Blended - Projection (XYZ)",
                ]
                self.lines = []
                for ax, title in zip(self.axes, titles):
                    ax.set_title(title)
                    ax.set_ylabel("value")
                    ax.grid(True, alpha=0.3)
                    line_x, = ax.plot([], [], color="red", label="x")
                    line_y, = ax.plot([], [], color="green", label="y")
                    line_z, = ax.plot([], [], color="blue", label="z")
                    ax.legend(loc="upper right")
                    self.lines.append((line_x, line_y, line_z))
                self.axes[-1].set_xlabel("time (s)")

            def update(self, t_value, red_pos, proj_pos, blend_pos):
                self.t.append(t_value)
                for i in range(3):
                    self.red[i].append(red_pos[i])
                    self.proj[i].append(proj_pos[i])
                    self.blend[i].append(blend_pos[i])
                    self.err_red_proj[i].append(red_pos[i] - proj_pos[i])
                    self.err_blend_proj[i].append(blend_pos[i] - proj_pos[i])

                data_groups = [
                    self.red,
                    self.proj,
                    self.blend,
                    self.err_red_proj,
                    self.err_blend_proj,
                ]
                t_list = list(self.t)
                for (line_x, line_y, line_z), data in zip(self.lines, data_groups):
                    line_x.set_data(t_list, list(data[0]))
                    line_y.set_data(t_list, list(data[1]))
                    line_z.set_data(t_list, list(data[2]))

                for ax in self.axes:
                    ax.relim()
                    ax.autoscale_view()

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

        plotter = PosePlotter(window_size=25)
        plot_time = 0.0
        plot_accum = 0.0
        plot_interval = 0.1

        def plot_updater(_, dt):
            nonlocal plot_time, plot_accum
            plot_time += dt
            plot_accum += dt
            if plot_accum < plot_interval:
                return
            plot_accum = 0.0

            red_pos = self.mouse_point.get_center()
            proj_pos = manifold_target_point(self.mouse_point.get_center())
            blend_pos = triad_anchor.get_center()
            plotter.update(plot_time, red_pos, proj_pos, blend_pos)

        plot_driver = Mobject().set_opacity(0)
        plot_driver.add_updater(plot_updater)
        self.add(plot_driver)

        # ------------------------------------------------------------
        # UI: external UDP target receiver and smoother
        # ------------------------------------------------------------


        import socket, threading, json
        from collections import defaultdict

        # --- construct() 안, trackers 만든 이후 ---
        PORT = 5005
        targets = defaultdict(lambda: None)
        lock = threading.Lock()

        def udp_listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", PORT))
            while True:
                data, _ = sock.recvfrom(4096)
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                with lock:
                    for k, v in msg.items():
                        targets[k] = v

        threading.Thread(target=udp_listener, daemon=True).start()

        def apply_targets_updater(_, dt):
            # dt 기반 smoothing(선택): 너무 튀면 부드럽게 따라가게
            a = 1.0 - np.exp(-30.0 * dt)  # 10은 반응 속도(크면 더 빨리)
            with lock:
                t = dict(targets)

            # mode는 즉시 반영하는 편이 보통 편함
            if t.get("mode") is not None:
                set_mode(int(round(t["mode"])))

            # 연속 파라미터들
            def lerp(cur, tgt):
                return cur if tgt is None else (1 - a) * cur + a * float(tgt)

            yaw.set_value(   lerp(yaw.get_value(),   t.get("yaw")))
            pitch.set_value( lerp(pitch.get_value(), t.get("pitch")))
            alpha.set_value( lerp(alpha.get_value(), t.get("alpha")))
            beta.set_value(  lerp(beta.get_value(),  t.get("beta")))

            # 필요하면 추가(예: omega, r_cir, k_r 등도 동일하게)
            # omega = ...
            # r_cir = ...

        driver = Mobject().set_opacity(0)
        driver.add_updater(apply_targets_updater)
        self.add(driver)








        # ------------------------------------------------------------
        # HUD: show current l, m and controls
        # ------------------------------------------------------------
        hud = Text("...", font_size=24).to_corner(UL)
        hud.fix_in_frame()
        self.add(hud)

        acc = 0.0
        def hud_updater(mob, dt):
            nonlocal acc
            acc += dt
            if acc < 0.1:   # 10Hz
                return
            acc = 0.0
            l, u0, v0, p_star, m = axis_state()
            k = int(round(mode.get_value()))
            mode_name = {1:"Locked",2:"Revolute",3:"Prismatic",4:"Screw"}.get(k,"Unknown")
            new = Text(
                f"... mode={k} ({mode_name}) ...\n"
                f"l=[{l[0]:.2f},{l[1]:.2f},{l[2]:.2f}] ...",
                font_size=24
            ).to_corner(UL)
            new.fix_in_frame()
            mob.become(new)

        hud.add_updater(hud_updater)

        self.wait(30)
