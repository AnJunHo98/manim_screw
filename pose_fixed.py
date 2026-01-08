# pose_mouse_local_constraint_field.py
from manimlib import *
import numpy as np

# ============================
# Utilities
# ============================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else (v / n)

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def smoothstep01(t: float) -> float:
    # classic smoothstep on [0,1]
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def near_manifold_weight(e: float, e_near: float, e_far: float) -> float:
    """
    e <= e_near  -> w = 1 (field dominates)
    e >= e_far   -> w = 0 (mouse dominates)
    smooth in between
    """
    if e_far <= e_near:
        return 1.0 if e <= e_near else 0.0
    t = (e - e_near) / (e_far - e_near)  # 0..1
    return 1.0 - smoothstep01(t)

def pose_frame_mobject(pos: np.ndarray,
                       R: np.ndarray,
                       axis_len: float = 1.0,
                       stroke_width: float = 7,
                       opacity: float = 1.0) -> VGroup:
    """
    6D pose = (pos, R). Visualize as triad + center dot.
    x=RED, y=GREEN, z=BLUE
    """
    pos = np.array(pos, dtype=float)

    x_vec = Vector(axis_len * (R @ np.array([1.0, 0.0, 0.0]))).set_color(RED)
    y_vec = Vector(axis_len * (R @ np.array([0.0, 1.0, 0.0]))).set_color(GREEN)
    z_vec = Vector(axis_len * (R @ np.array([0.0, 0.0, 1.0]))).set_color(BLUE)

    for v in (x_vec, y_vec, z_vec):
        v.shift(pos)
        v.set_stroke(width=stroke_width)
        v.set_opacity(opacity)

    center = Dot(point=pos, radius=0.06).set_color(WHITE).set_opacity(opacity)
    return VGroup(x_vec, y_vec, z_vec, center)

# ============================
# Constraint + A-centered orbit field
# ============================
class CylinderPlaneConstraint:
    """
    Axis (p0, a). Constraint manifold:
      - cylinder: distance-to-axis rho = L
      - plane:    axis coordinate s = s_des

    Guidance field:
      F = F_r + F_s + F_t
        F_r: radial stabilizer   (rho -> L)
        F_s: axial stabilizer    (s -> s_des)
        F_t: tangential/orbit    (rotate around axis a)
    """
    def __init__(self,
                 p0: np.ndarray,
                 a_axis: np.ndarray,
                 u_fallback: np.ndarray,
                 L: float,
                 s_des: float,
                 k_r: float = 12.0,
                 k_s: float = 12.0,
                 k_t: float = 2.5,
                 omega: float = 1.0):
        self.p0 = np.array(p0, dtype=float)
        self.a = normalize(np.array(a_axis, dtype=float))
        self.u_fallback = normalize(np.array(u_fallback, dtype=float))
        self.L = float(L)
        self.s_des = float(s_des)
        self.k_r = float(k_r)
        self.k_s = float(k_s)
        self.k_t = float(k_t)
        self.omega = float(omega)

    def decompose(self, p: np.ndarray):
        """
        p -> (s, r_perp, rho, r_hat)
        """
        p = np.array(p, dtype=float)
        r = p - self.p0
        s = float(np.dot(self.a, r))
        r_perp = r - s * self.a
        rho = float(np.linalg.norm(r_perp))
        if rho < 1e-9:
            r_hat = self.u_fallback
        else:
            r_hat = r_perp / rho
        return s, r_perp, rho, r_hat

    def constraint_only_force(self, p: np.ndarray) -> np.ndarray:
        s, _, rho, r_hat = self.decompose(p)
        F_r = -self.k_r * (rho - self.L) * r_hat
        F_s = -self.k_s * (s - self.s_des) * self.a
        return F_r + F_s

    def tangential_force(self, p: np.ndarray) -> np.ndarray:
        """
        Orbit around axis a via right-hand-rule:
          F_t ~ a x r_perp
        """
        _, r_perp, rho, r_hat = self.decompose(p)
        if rho < 1e-9:
            t = np.cross(self.a, r_hat)
        else:
            t = np.cross(self.a, r_perp)
        return self.k_t * self.omega * t

    def guidance_force(self, p: np.ndarray) -> np.ndarray:
        return self.constraint_only_force(p) + self.tangential_force(p)

    def project_to_manifold(self, p: np.ndarray) -> np.ndarray:
        _, _, _, r_hat = self.decompose(p)
        center = self.p0 + self.s_des * self.a
        return center + self.L * r_hat

    def frame_from_point(self, p: np.ndarray) -> np.ndarray:
        _, r_perp, _, r_hat = self.decompose(p)
        z_axis = self.a
        x_axis = self.u_fallback if np.linalg.norm(r_perp) < 1e-6 else r_hat
        y_axis = normalize(np.cross(z_axis, x_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
        return np.column_stack([x_axis, y_axis, z_axis])

# ============================
# Scene
# ============================
class PoseMouseLocalConstraintField(ThreeDScene):
    def setup_camera(self):
        ENABLE_CAMERA_SPIN = False

        if hasattr(self, "move_camera"):
            self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, run_time=0)
        elif hasattr(self, "camera") and hasattr(self.camera, "frame") and hasattr(self.camera.frame, "set_euler_angles"):
            self.camera.frame.set_euler_angles(theta=-45 * DEGREES, phi=70 * DEGREES, gamma=0)

        if ENABLE_CAMERA_SPIN:
            frame = getattr(getattr(self, "camera", None), "frame", None)
            if frame is not None:
                def spin(m, dt):
                    if hasattr(m, "increment_theta"):
                        m.increment_theta(0.10 * dt)
                    else:
                        m.rotate(0.10 * dt, axis=OUT)
                frame.add_updater(spin)

    def construct(self):
        self.setup_camera()
        self.add(ThreeDAxes())

        # ----------------------------
        # Pose A + axis
        # ----------------------------
        poseA_pos = np.array([0.0, 0.0, 0.0])
        a_axis = OUT
        u_fallback = RIGHT

        RA = np.eye(3)
        self.add(pose_frame_mobject(poseA_pos, RA, axis_len=1.0, stroke_width=8))

        # ----------------------------
        # Constraint parameters
        # ----------------------------
        L = 3.0
        s_des = 0.0

        constraint = CylinderPlaneConstraint(
            p0=poseA_pos,
            a_axis=np.array(a_axis),
            u_fallback=np.array(u_fallback),
            L=L,
            s_des=s_des,
            k_r=12.0,
            k_s=12.0,
            k_t=2.5,     # 접선(회전) 강도
            omega=1.0    # 회전 방향(부호 바꾸면 반대)
        )

        # Visualize axis and ring (intersection at s_des)
        axis_line = Line(
            poseA_pos - 6.0 * np.array(a_axis),
            poseA_pos + 6.0 * np.array(a_axis),
        ).set_color(GREY_B).set_stroke(width=4)
        self.add(axis_line)

        circle_center = poseA_pos + s_des * np.array(a_axis)
        guide_ring = ParametricCurve(
            lambda tt: circle_center + L * (np.cos(tt) * np.array(RIGHT) + np.sin(tt) * np.array(UP)),
            t_range=(0, TAU, 0.02)
        ).set_color(GREY_C).set_stroke(width=3, opacity=0.85)
        self.add(guide_ring)

        # ----------------------------
        # Pose B: 기본은 마우스 추종 + (링에 가까워지면) 필드 영향 증가
        # ----------------------------
        mouse_scale = 1.0

        def mouse_world_point():
            mp = self.mouse_point
            p = mp.get_center() if hasattr(mp, "get_center") else np.array(mp, dtype=float)
            # 마우스는 plane(z=s_des) 위에서 움직이게 고정
            return np.array([mouse_scale * p[0], mouse_scale * p[1], s_des], dtype=float)

        # Pose B anchor (actual simulated point)
        poseB_anchor = Dot(point=poseA_pos + np.array([L, 0.0, s_des]), radius=0.01).set_opacity(0.0)
        self.add(poseB_anchor)

        # ----- Blend tuning -----
        k_mouse = 6.0          # 마우스 추종 "스프링" 게인
        field_gain = 0.18      # 벡터장 속도 스케일
        v_max = 10.0           # 안정성용 속도 클램프

        # "접선이 의미있는 영역" = 제약 다양체(링)에 가까운 영역으로 해석
        # e는 링까지의 (radial+axial) 오차로 정의
        e_near = 0.15          # 이 거리 이하면 field 영향 거의 최대
        e_far  = 1.20          # 이 거리 이상이면 field 영향 거의 0

        def update_poseB_anchor(m: Mobject, dt: float):
            p = np.array(m.get_center(), dtype=float)
            pm = mouse_world_point()

            # mouse-follow velocity
            v_mouse = k_mouse * (pm - p)

            # error-to-manifold (use radial+axial errors)
            s, _, rho, _ = constraint.decompose(p)
            e = float(np.sqrt((rho - L) ** 2 + (s - s_des) ** 2))

            # weight: near manifold -> 1, far -> 0
            w = near_manifold_weight(e, e_near=e_near, e_far=e_far)

            # A-centered guidance field (includes constraint + tangential/orbit)
            v_field = field_gain * constraint.guidance_force(p)

            # blend: always follow mouse, but near manifold the field term dominates more
            v = v_mouse + w * v_field

            vn = float(np.linalg.norm(v))
            if vn > v_max:
                v = (v_max / vn) * v

            m.shift(dt * v)

        poseB_anchor.add_updater(update_poseB_anchor)

        def poseB_point():
            return np.array(poseB_anchor.get_center(), dtype=float)

        poseB_frame = always_redraw(
            lambda: pose_frame_mobject(
                poseB_point(),
                constraint.frame_from_point(poseB_point()),
                axis_len=0.85,
                stroke_width=7
            )
        )
        self.add(poseB_frame)

        # ----------------------------
        # Projection + helper visuals
        # ----------------------------
        proj_dot = always_redraw(
            lambda: Dot(
                constraint.project_to_manifold(poseB_point()),
                radius=0.06
            ).set_color(GREY_A).set_opacity(0.9)
        )
        self.add(proj_dot)

        proj_pose = always_redraw(
            lambda: pose_frame_mobject(
                constraint.project_to_manifold(poseB_point()),
                constraint.frame_from_point(constraint.project_to_manifold(poseB_point())),
                axis_len=0.8,
                stroke_width=6,
                opacity=0.35
            )
        )
        self.add(proj_pose)

        to_proj_vec = always_redraw(
            lambda: Vector(
                0.7 * normalize(constraint.project_to_manifold(poseB_point()) - poseB_point())
            ).set_color(PURPLE).set_stroke(width=5).shift(poseB_point())
        )
        self.add(to_proj_vec)

        # Guidance at Pose B (full, not weighted)
        force_vec_at_B = always_redraw(
            lambda: Vector(
                0.9 * normalize(constraint.guidance_force(poseB_point()))
            ).set_color(ORANGE).set_stroke(width=6).shift(poseB_point())
        )
        self.add(force_vec_at_B)

        # ----------------------------
        # A-centered vector field visualization (guidance field)
        # ----------------------------
        local_field = VGroup()

        grid_xy = [-4.0, -2.0, 0.0, 2.0, 4.0]
        grid_z  = [s_des]      # plane 위에서의 회전장(접선장) 직관적으로 보기
        arrow_len = 0.55

        origin = np.array(poseA_pos, dtype=float)

        for dx in grid_xy:
            for dy in grid_xy:
                for dz in grid_z:
                    sample = origin + dx * np.array(RIGHT) + dy * np.array(UP) + dz * np.array(OUT)
                    f = constraint.guidance_force(sample)
                    d = arrow_len * normalize(f)
                    if np.linalg.norm(d) < 1e-6:
                        d = arrow_len * 0.001 * np.array(RIGHT)
                    vec = Vector(d).set_color(YELLOW).set_opacity(0.55).set_stroke(width=4).shift(sample)
                    local_field.add(vec)

        self.add(local_field)

        # ----------------------------
        # Label
        # ----------------------------
        label = Text(
            "Pose B: mouse-follow + (near ring) A-centered orbit field influence.\n"
            "Yellow: A-centered guidance field. Orange: guidance at B. Purple: B -> projection.\n"
            f"Blend: v = k_mouse*(pm-p) + w(e)*field_gain*F(p),  w(e): e<=e_near ->1, e>=e_far ->0",
            font_size=24
        ).to_corner(UL)
        label.fix_in_frame()
        self.add(label)

        self.wait(30)
