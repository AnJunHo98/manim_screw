# pose_mouse_local_constraint_field.py
from manimlib import *
import numpy as np

import numpy as np

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else (v / n)

def wrap_pi(angle: float) -> float:
    # [-pi, pi)로 래핑
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def smoothstep01(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def near_manifold_weight(e: float, e_near: float, e_far: float) -> float:
    """
    e <= e_near  -> w = 1 (A-기준 벡터장 영향 최대)
    e >= e_far   -> w = 0 (마우스 추종이 지배)
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

class ScrewJointConstraint:
    """
    Screw joint constraint about axis (p0, a).
      - radius constraint: rho = L
      - helical constraint: s = s0 + pitch * phi

    Special handling:
      - If |pitch| is very large, treat as prismatic:
          rho = L and phi = phi0 (no rotation), s is free.
    """
    def __init__(self,
                 p0: np.ndarray,
                 a_axis: np.ndarray,
                 u_ref: np.ndarray,          # angle reference direction (perp to axis ideally)
                 L: float,
                 s0: float,
                 pitch: float,               # length per rad (h). 0 -> revolute
                 phi0: float = 0.0,          # used in prismatic mode (keep angle fixed)
                 pitch_prismatic_thresh: float = 1e3,
                 k_r: float = 12.0,
                 k_phase: float = 12.0,      # helical phase correction gain
                 k_phi: float = 12.0,        # prismatic angle-hold gain
                 k_drive: float = 0.0,       # along-manifold assist (optional)
                 omega: float = 1.0):        # drive direction sign/speed
        self.p0 = np.array(p0, dtype=float)
        self.a = normalize(np.array(a_axis, dtype=float))
        self.L = float(L)
        self.s0 = float(s0)
        self.pitch = float(pitch)
        self.phi0 = float(phi0)
        self.pitch_prismatic_thresh = float(pitch_prismatic_thresh)

        self.k_r = float(k_r)
        self.k_phase = float(k_phase)
        self.k_phi = float(k_phi)
        self.k_drive = float(k_drive)
        self.omega = float(omega)

        # Build stable reference basis (u0, v0) on plane perpendicular to axis
        u = np.array(u_ref, dtype=float)
        u = u - np.dot(u, self.a) * self.a
        if np.linalg.norm(u) < 1e-8:
            # fallback if u_ref parallel to axis
            # pick any vector not parallel to a
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, self.a)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            u = tmp - np.dot(tmp, self.a) * self.a
        self.u0 = normalize(u)
        self.v0 = normalize(np.cross(self.a, self.u0))  # right-handed

    def is_prismatic(self) -> bool:
        return abs(self.pitch) >= self.pitch_prismatic_thresh

    def decompose(self, p: np.ndarray):
        """
        Returns:
          s, r_perp, rho, r_hat, phi, t_hat
        where t_hat is unit tangent around axis (a x r_hat).
        """
        p = np.array(p, dtype=float)
        r = p - self.p0
        s = float(np.dot(self.a, r))
        r_perp = r - s * self.a
        rho = float(np.linalg.norm(r_perp))

        if rho < 1e-9:
            # degenerate: pick reference radial
            r_hat = self.u0
        else:
            r_hat = r_perp / rho

        # angle phi from (u0, v0) basis
        x = float(np.dot(r_perp, self.u0))
        y = float(np.dot(r_perp, self.v0))
        phi = float(np.arctan2(y, x)) if rho >= 1e-9 else 0.0

        t_hat = normalize(np.cross(self.a, r_hat))  # tangent direction
        return s, r_perp, rho, r_hat, phi, t_hat

    def constraint_only_force(self, p: np.ndarray) -> np.ndarray:
        """
        Constraint stabilization only (no along-manifold drive):
          - rho -> L
          - if screw/revolute: s - (s0 + pitch*phi) -> 0
          - if prismatic: phi -> phi0 (no rotation), s free
        """
        s, _, rho, r_hat, phi, t_hat = self.decompose(p)

        # 1) radial to cylinder
        F_r = -self.k_r * (rho - self.L) * r_hat

        # 2) phase constraint
        if self.is_prismatic():
            # keep angle fixed; do NOT constrain s
            e_phi = wrap_pi(phi - self.phi0)
            F_phase = -self.k_phi * e_phi * t_hat
            return F_r + F_phase

        # screw/revolute: e = s - (s0 + pitch*phi)
        e = (s - (self.s0 + self.pitch * phi))

        # Choose minimal-norm (u along axis, v along tangent) to reduce e:
        # e_dot = u - pitch*(v/rho)  -> set = -k_phase*e
        rr = max(rho, 1e-6)
        alpha = self.pitch / rr
        denom = 1.0 + alpha * alpha

        u = -(self.k_phase * e) / denom
        v = +(self.k_phase * e) * alpha / denom

        F_phase = u * self.a + v * t_hat
        return F_r + F_phase

    def drive_along_manifold(self, p: np.ndarray) -> np.ndarray:
        """
        Optional: move along the 1-DOF joint direction to 'feel' revolute/prismatic.
        - revolute (pitch=0): tangent direction
        - screw: tangent + (pitch/rho)*axis
        - prismatic: axis direction
        """
        if self.k_drive == 0.0:
            return np.zeros(3)

        s, _, rho, r_hat, phi, t_hat = self.decompose(p)

        if self.is_prismatic():
            d = self.a
        else:
            rr = max(rho, 1e-6)
            d = normalize(t_hat + (self.pitch / rr) * self.a)

        return self.k_drive * self.omega * d

    def guidance_force(self, p: np.ndarray) -> np.ndarray:
        return self.constraint_only_force(p) + self.drive_along_manifold(p)

    def project_to_manifold(self, p: np.ndarray) -> np.ndarray:
        """
        Simple projection:
          - set rho -> L
          - set phase:
              screw: keep phi, set s = s0 + pitch*phi
              prismatic: keep s, set phi = phi0
        """
        s, _, rho, r_hat, phi, t_hat = self.decompose(p)

        if self.is_prismatic():
            # keep s, but force angle = phi0
            # build r_hat from phi0
            r_hat0 = normalize(np.cos(self.phi0) * self.u0 + np.sin(self.phi0) * self.v0)
            return self.p0 + s * self.a + self.L * r_hat0

        # screw/revolute
        s_new = self.s0 + self.pitch * phi
        return self.p0 + s_new * self.a + self.L * r_hat

    def frame_from_point(self, p: np.ndarray) -> np.ndarray:
        """
        Visualization frame:
          z = axis a, x = radial, y = z x x
        """
        _, r_perp, rho, r_hat, _, _ = self.decompose(p)
        z_axis = self.a
        x_axis = r_hat if np.linalg.norm(r_perp) >= 1e-6 else self.u0
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
        # ============================================================
        # User-tunable parameters (edit here)
        # ============================================================

        # Constraint geometry (제약 기하 파라미터)
        L = 3.0              # 원통 제약의 반지름(축으로부터 목표 거리 rho = L)
        s_des = 0.0          # 평면 제약의 목표 축 좌표(축 방향 위치) s = s_des

        # 3D mouse control (마우스 3D 확장: z를 키로 조절)
        mouse_z_init = s_des      # 시작 z
        mouse_z_min  = -6.0       # z 하한
        mouse_z_max  =  6.0       # z 상한
        mouse_z_speed = 3.0       # (units/sec) Q/E 누르고 있을 때 z 변화 속도
        mouse_z_key_up = "E"      # z 증가 키
        mouse_z_key_dn = "Q"      # z 감소 키

        # Screw parameters
        s0 = 0.0               # phi=0일 때 축방향 기준 위치
        pitch = 0            # 0이면 revolute, 0.5~2.0이면 screw(헬릭스), 매우 크면 prismatic에 근접
        phi0 = 0.0             # prismatic 모드에서 유지할 각도(회전 금지)

        pitch_prismatic_thresh = 1e3  # |pitch|가 이 이상이면 prismatic로 스위치

        # Constraint gains (제약 복원 게인)
        k_r = 12.0            # 원통 반경(rho-L) 오차 복원 강도 (필수)
        k_phase = 12.0        # screw/revolute에서 위상 제약 e = s-(s0+pitch*phi) 복원 강도
        k_phi = 12.0          # prismatic 모드에서 phi -> phi0 (각도 고정) 복원 강도

        # Along-manifold drive (원하면 joint 1-DOF 방향으로 "흘러가게" 하는 구동)
        k_drive = 0.0        # 0이면 끔. 0.5~2.0 정도로 켜볼 수 있음
        omega_drive = 1.0     # 구동 방향/속도 부호(+, -)
        
        # Pose B motion blend (mouse-follow vs A-field) (Pose B 이동 블렌딩 파라미터)
        k_mouse = 6.0        # 마우스 추종 강도(마우스 쪽으로 끌어당기는 스프링/PD 느낌)
        field_gain = 0.8     # A-기준 벡터장(guidance_force)을 속도로 반영하는 스케일
        v_max = 10.0         # 안정성을 위한 최대 속도 제한(클램프)

        # Distance-to-manifold weighting (near -> field dominates) (링/제약 다양체 근접도 기반 가중치)
        e_near = 0.15        # 제약 오차 e가 이 값 이하이면 벡터장 영향이 거의 최대(w≈1)
        e_far  = 1.20        # 제약 오차 e가 이 값 이상이면 마우스 추종이 지배적(w≈0)

        # A-centered global field visualization (TEAL) (Pose A 기준 전역 벡터장 시각화 설정)
        grid_xy_A = [-4.0, -2.0, 0.0, 2.0, 4.0]  # A 주변에서 벡터장을 샘플링할 x,y 격자 좌표들
        arrow_len_A = 0.55                       # A-기준 벡터장 화살표(벡터) 표시 길이 스케일
        field_A_opacity = 0.45                   # A-기준 벡터장 화살표 투명도
        field_A_stroke = 4                       # A-기준 벡터장 화살표 선 두께

        # B-centered local field visualization (YELLOW) (your original style)
        # (Pose B 기준 로컬 벡터장 시각화 설정: 기존 constraint_only_force 방식)
        grid_xy_B = [-1.2, -0.6, 0.0, 0.6, 1.2]  # B 주변 로컬 벡터장을 샘플링할 x,y 오프셋들
        grid_z_B  = [-0.8, 0.0, 0.8]             # B 주변 로컬 벡터장을 샘플링할 z 오프셋들
        arrow_len_B = 0.75                       # B-기준 로컬 벡터장 화살표 표시 길이 스케일
        field_B_opacity = 0.1                   # B-기준 로컬 벡터장 화살표 투명도
        field_B_stroke = 2                       # B-기준 로컬 벡터장 화살표 선 두께

        # Mouse marker (RED ball) (마우스 위치 표시용 빨간 공 설정)
        mouse_ball_radius = 0.09                 # 마우스 위치를 표시하는 빨간 공의 반지름
        mouse_ball_opacity = 0.95                # 마우스 위치 빨간 공의 투명도


        # ============================================================
        # Scene setup
        # ============================================================
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
        # Constraint object
        # ----------------------------
        constraint = ScrewJointConstraint(
            p0=poseA_pos,
            a_axis=np.array(a_axis),
            u_ref=np.array(RIGHT),
            L=L,
            s0=s0,
            pitch=pitch,
            phi0=phi0,
            pitch_prismatic_thresh=pitch_prismatic_thresh,
            k_r=k_r,
            k_phase=k_phase,
            k_phi=k_phi,
            k_drive=k_drive,
            omega=omega_drive
        )

        # Axis + ring visualize
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
        # Mouse point (world)
        # ----------------------------
        mouse_scale = 1.0

        def mouse_world_point():
            mp = self.mouse_point
            p = mp.get_center() if hasattr(mp, "get_center") else np.array(mp, dtype=float)
            # return np.array([mouse_scale * p[0], mouse_scale * p[1], s_des], dtype=float)
            return np.array([mouse_scale * p[0], mouse_scale * p[1], mouse_z.get_value()], dtype=float)

        # --- 3D mouse Z tracker ---
        mouse_z = ValueTracker(mouse_z_init)
        self.add(mouse_z)

        # 키 입력으로 z 조절 (manimgl은 pyglet 기반)
        try:
            from pyglet.window import key as pyg_key
        except Exception:
            pyg_key = None

        z_controller = Mobject().set_opacity(0.0)

        def z_key_updater(m: Mobject, dt: float):
            if pyg_key is None:
                return
            win = getattr(self, "window", None)
            if win is None or (not hasattr(win, "is_key_pressed")):
                return

            # 원하는 키로 매핑
            key_up = getattr(pyg_key, mouse_z_key_up, None)
            key_dn = getattr(pyg_key, mouse_z_key_dn, None)

            dz = 0.0
            if key_up is not None and win.is_key_pressed(key_up):
                dz += mouse_z_speed * dt
            if key_dn is not None and win.is_key_pressed(key_dn):
                dz -= mouse_z_speed * dt

            if dz != 0.0:
                mouse_z.increment_value(dz)
                mouse_z.set_value(clamp(mouse_z.get_value(), mouse_z_min, mouse_z_max))

        z_controller.add_updater(z_key_updater)
        self.add(z_controller)


        # --- NEW: red ball showing current mouse position ---
        mouse_ball = always_redraw(
            lambda: Sphere(
                radius=mouse_ball_radius,
                resolution=(18, 18)
            ).set_color(RED).set_opacity(mouse_ball_opacity).move_to(mouse_world_point())
        )
        
        self.add(mouse_ball)

        # ----------------------------
        # Pose B: mouse-follow + (near ring) A-centered orbit influence
        # ----------------------------
        poseB_anchor = Dot(point=poseA_pos + np.array([L, 0.0, s_des]), radius=0.01).set_opacity(0.0)
        self.add(poseB_anchor)

        def update_poseB_anchor(m: Mobject, dt: float):
            p = np.array(m.get_center(), dtype=float)
            pm = mouse_world_point()

            v_mouse = k_mouse * (pm - p)

            s, r_perp, rho, r_hat, phi, t_hat = constraint.decompose(p)
            
            if constraint.is_prismatic():
                e = float(np.sqrt((rho - L) ** 2 + (wrap_pi(phi - phi0)) ** 2))
            else:
                phase_err = float(s - (s0 + pitch * phi))
                e = float(np.sqrt((rho - L) ** 2 + phase_err ** 2))
            
            # e = float(np.sqrt((rho - L) ** 2 + (s - s_des) ** 2))
            w = near_manifold_weight(e, e_near=e_near, e_far=e_far)

            v_field = field_gain * constraint.guidance_force(p)

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

        force_vec_at_B = always_redraw(
            lambda: Vector(
                0.9 * normalize(constraint.guidance_force(poseB_point()))
            ).set_color(ORANGE).set_stroke(width=6).shift(poseB_point())
        )
        self.add(force_vec_at_B)

        # ----------------------------
        # (A 기준) 고정 벡터장 (TEAL)
        # ----------------------------
        field_A = VGroup()
        origin = np.array(poseA_pos, dtype=float)

        for dx in grid_xy_A:
            for dy in grid_xy_A:
                sample = origin + dx * np.array(RIGHT) + dy * np.array(UP) + s_des * np.array(OUT)
                f = constraint.guidance_force(sample)
                d = arrow_len_A * normalize(f)
                if np.linalg.norm(d) < 1e-6:
                    d = arrow_len_A * 0.001 * np.array(RIGHT)
                vec = Vector(d).set_color(TEAL).set_opacity(field_A_opacity).set_stroke(width=field_A_stroke).shift(sample)
                field_A.add(vec)

        self.add(field_A)

        # ----------------------------
        # (B 기준) 로컬 벡터장: 기존 코드 방식 (constraint_only_force) (YELLOW)
        # ----------------------------
        local_field_B = VGroup()

        for dx in grid_xy_B:
            for dy in grid_xy_B:
                for dz in grid_z_B:
                    vec = Vector(RIGHT).set_color(YELLOW).set_opacity(field_B_opacity).set_stroke(width=field_B_stroke)

                    def make_updater(dx=dx, dy=dy, dz=dz):
                        def _upd(m: Mobject, dt):
                            pB = poseB_point()
                            sample = pB + dx * np.array(RIGHT) + dy * np.array(UP) + dz * np.array(OUT)

                            f = constraint.constraint_only_force(sample)
                            d = arrow_len_B * normalize(f)

                            if np.linalg.norm(d) < 1e-6:
                                d = arrow_len_B * 0.001 * np.array(RIGHT)

                            m.become(
                                Vector(d)
                                .set_color(YELLOW)
                                .set_opacity(field_B_opacity)
                                .set_stroke(width=field_B_stroke)
                                .shift(sample)
                            )
                        return _upd

                    vec.add_updater(make_updater())
                    local_field_B.add(vec)

        self.add(local_field_B)

        # ----------------------------
        # Label
        # ----------------------------
        label = Text(
            "RED sphere: current mouse position.\n"
            "Pose B: mouse-follow + (near ring) A-centered orbit influence.\n"
            "TEAL: A-centered guidance field. YELLOW: B-centered local field (constraint-only).\n"
            "ORANGE: guidance at B. PURPLE: B -> projected point.",
            font_size=24
        ).to_corner(UL)
        label.fix_in_frame()
        self.add(label)

        self.wait(30)
