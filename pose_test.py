# pose_mouse_local_constraint_field.py
from manimlib import *
import numpy as np

# ============================
# Utilities
# ============================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else (v / n)

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
# Constraint field (constraint-only)
# ============================
class CylinderPlaneConstraint:
    """
    Axis (p0, a). Constraint manifold:
      - cylinder: distance-to-axis rho = L
      - plane:    axis coordinate s = s_des

    Guidance field here is "constraint-only":
      F = F_r + F_s
    (No angular/ideal-target term)
    """
    def __init__(self,
                 p0: np.ndarray,
                 a_axis: np.ndarray,
                 u_fallback: np.ndarray,
                 L: float,
                 s_des: float,
                 k_r: float = 12.0,
                 k_s: float = 12.0):
        self.p0 = np.array(p0, dtype=float)
        self.a = normalize(np.array(a_axis, dtype=float))          # axis unit
        self.u_fallback = normalize(np.array(u_fallback, dtype=float))  # for degenerate radial
        self.L = float(L)
        self.s_des = float(s_des)
        self.k_r = float(k_r)
        self.k_s = float(k_s)

    def decompose(self, p: np.ndarray):
        """
        p -> (s, r_perp, rho, r_hat)
        where:
          s = dot(a, p-p0)
          r_perp = (p-p0) - s*a
          rho = ||r_perp||
          r_hat = r_perp / rho
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
        """
        Guidance direction to satisfy constraints:
          - pull rho -> L (radial)
          - pull s   -> s_des (axial)
        """
        s, _, rho, r_hat = self.decompose(p)

        # radial toward cylinder surface
        F_r = -self.k_r * (rho - self.L) * r_hat

        # axial toward plane s = s_des
        F_s = -self.k_s * (s - self.s_des) * self.a

        return F_r + F_s

    def project_to_manifold(self, p: np.ndarray) -> np.ndarray:
        """
        Closest point on the intersection (approximately):
          - set s -> s_des
          - set rho -> L, keep same radial direction
        """
        s, _, rho, r_hat = self.decompose(p)
        center = self.p0 + self.s_des * self.a
        return center + self.L * r_hat

    def frame_from_point(self, p: np.ndarray) -> np.ndarray:
        """
        Make an orientation that looks "joint-like":
          z-axis = axis a
          x-axis = radial direction from axis
          y-axis = z x x
        """
        _, r_perp, rho, r_hat = self.decompose(p)
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
        # 카메라 회전은 원하면 켤 수 있게 플래그로 둠
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

        # 마우스 조작을 확실히 하기 위해 axis는 월드 Z로 둠 (OUT)
        a_axis = OUT  # np.array([0,0,1]) equivalent
        u_fallback = RIGHT  # radial degeneracy fallback

        # Pose A는 고정 프레임으로만 표시 (원하면 회전시키면 됨)
        RA = np.eye(3)
        self.add(pose_frame_mobject(poseA_pos, RA, axis_len=1.0, stroke_width=8))

        # ----------------------------
        # Constraint parameters
        # ----------------------------
        L = 3.0
        s_des = 0.0  # Pose B를 이 평면(z=s_des)에 두고 싶으면 0.0이 자연스러움

        constraint = CylinderPlaneConstraint(
            p0=poseA_pos,
            a_axis=np.array(a_axis),
            u_fallback=np.array(u_fallback),
            L=L,
            s_des=s_des,
            k_r=12.0,
            k_s=12.0
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
        # Pose B follows mouse (kinematic)
        # ----------------------------
        mouse_scale = 1.0

        def mouse_world_point():
            mp = self.mouse_point
            p = mp.get_center() if hasattr(mp, "get_center") else np.array(mp, dtype=float)
            # mouse_point는 보통 z=0 평면에서의 좌표이므로,
            # 여기서는 z를 s_des로 고정해 "평면 위에서" 움직이게 함
            return np.array([mouse_scale * p[0], mouse_scale * p[1], s_des], dtype=float)

        poseB_frame = always_redraw(
            lambda: pose_frame_mobject(
                mouse_world_point(),
                constraint.frame_from_point(mouse_world_point()),
                axis_len=0.85,
                stroke_width=7
            )
        )
        self.add(poseB_frame)

        # ----------------------------
        # Constraint projection (B-dependent target, NOT ideal target)
        # ----------------------------
        proj_dot = always_redraw(
            lambda: Dot(
                constraint.project_to_manifold(mouse_world_point()),
                radius=0.06
            ).set_color(GREY_A).set_opacity(0.9)
        )
        self.add(proj_dot)

        proj_pose = always_redraw(
            lambda: pose_frame_mobject(
                constraint.project_to_manifold(mouse_world_point()),
                constraint.frame_from_point(constraint.project_to_manifold(mouse_world_point())),
                axis_len=0.8,
                stroke_width=6,
                opacity=0.35
            )
        )
        self.add(proj_pose)

        # Vector from Pose B -> projection (optional but useful)
        to_proj_vec = always_redraw(
            lambda: Vector(
                0.7 * normalize(constraint.project_to_manifold(mouse_world_point()) - mouse_world_point())
            ).set_color(PURPLE).set_stroke(width=5).shift(mouse_world_point())
        )
        self.add(to_proj_vec)

        # Constraint-only guidance at Pose B
        force_vec_at_B = always_redraw(
            lambda: Vector(
                0.9 * normalize(constraint.constraint_only_force(mouse_world_point()))
            ).set_color(ORANGE).set_stroke(width=6).shift(mouse_world_point())
        )
        self.add(force_vec_at_B)

        # ----------------------------
        # Local vector field around Pose B (this is what you asked)
        # ----------------------------
        local_field = VGroup()

        grid_xy = [-1.2, -0.6, 0.0, 0.6, 1.2]
        grid_z  = [-0.8, 0.0, 0.8]  # z 방향도 조금 샘플링해서 plane 제약 방향도 보이게
        arrow_len = 0.55

        for dx in grid_xy:
            for dy in grid_xy:
                for dz in grid_z:
                    vec = Vector(RIGHT).set_color(YELLOW).set_opacity(0.55).set_stroke(width=4)

                    def make_updater(dx=dx, dy=dy, dz=dz):
                        def _upd(m: Mobject, dt):
                            pB = mouse_world_point()
                            sample = pB + dx * np.array(RIGHT) + dy * np.array(UP) + dz * np.array(OUT)

                            f = constraint.constraint_only_force(sample)
                            d = arrow_len * normalize(f)

                            if np.linalg.norm(d) < 1e-6:
                                d = arrow_len * 0.001 * np.array(RIGHT)

                            m.become(
                                Vector(d).set_color(YELLOW).set_opacity(0.55).set_stroke(width=4).shift(sample)
                            )
                        return _upd

                    vec.add_updater(make_updater())
                    local_field.add(vec)

        self.add(local_field)

        # ----------------------------
        # Screen-fixed label
        # ----------------------------
        label = Text(
            "Pose B follows mouse.\n"
            "Yellow: local guidance field around Pose B (constraint-only).\n"
            "Orange: guidance at Pose B. Purple: B -> projected point on constraint manifold.",
            font_size=26
        ).to_corner(UL)
        label.fix_in_frame()
        self.add(label)

        self.wait(30)
