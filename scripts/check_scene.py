"""D07 장면 구성표 확인 코드 (scenes/monitor_arm.xml, 수직 다관절 6축 RRRRRR).

실행 예:
    uv run python scripts/check_scene.py
"""
from pathlib import Path
import math
import numpy as np
import mujoco

SCENE = Path(__file__).resolve().parent.parent / "scenes" / "monitor_arm.xml"

# R-01, R-02, R-03, R-10 기준
DIST_MIN_M, DIST_MAX_M = 0.50, 0.70
TOP_TOL_MM = 15.0
ANGLE_MAX_DEG = 10.0
CLEAR_MIN_MM = 50.0


def monitor_extents(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[float, float, float]:
    """모니터 패널(61x3x36 cm) 8개 꼭짓점의 월드 좌표로 최저 z, 최고 z, 최후방 y를 계산한다."""
    gid = m.geom("monitor_panel").id
    pos = d.geom_xpos[gid]
    rot = d.geom_xmat[gid].reshape(3, 3)
    hx, hy, hz = m.geom_size[gid]
    corners = np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    world_pts = pos + corners @ rot.T
    return float(world_pts[:, 2].min()), float(world_pts[:, 2].max()), float(world_pts[:, 1].max())


def arm_max_y(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """모니터 패널과 팔꿈치·상부 링크 중 뒷벽 쪽 최후방 y를 계산한다."""
    _, _, mon_max_y = monitor_extents(m, d)
    elbow_y = float(d.xpos[m.body("arm_forearm").id][1]) + 0.032
    return max(mon_max_y, elbow_y)


def monitor_world_roll_deg(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """화면 로컬 +Z축이 월드 수평면에서 얼마나 회전했는지(가로 0도, 세로 90도) 계산한다."""
    site_mat = d.site_xmat[m.site("monitor_center").id].reshape(3, 3)
    local_z = site_mat[:, 2]
    # 화면 정면이 -Y를 향할 때 로컬 +Z의 X·Z 성분으로 롤 각도 계산
    deg = abs(math.degrees(math.atan2(float(local_z[0]), float(local_z[2]))))
    return 0.0 if deg < 0.5 else deg


def target_ref_z(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    screen_center: np.ndarray,
    eye: np.ndarray,
    is_portrait: bool,
) -> tuple[float, float, str]:
    """R-02 기준 높이와 실제 측정 높이를 반환한다."""
    _, max_z, _ = monitor_extents(m, d)
    if is_portrait:
        read_line_z = max_z - (0.61 / 3.0)
        return float(eye[2]), read_line_z, "상단1/3"

    head_id = m.body("user_head").id
    head_xmat = d.xmat[head_id].reshape(3, 3)
    gaze_dir = head_xmat[:, 1]
    if gaze_dir[2] > 0.05:
        dy = float(screen_center[1] - eye[1])
        target_z = float(eye[2] + dy * (gaze_dir[2] / gaze_dir[1]))
    else:
        target_z = float(eye[2])

    top_z = float(d.site_xpos[m.site("monitor_top").id][2])
    return target_z, top_z, "상단"


def check_pose(m: mujoco.MjModel, key_name: str) -> bool:
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    for _ in range(500):  # 1.0 s 물리 진행
        mujoco.mj_step(m, d)

    eye = d.site_xpos[m.site("eye_center").id].copy()
    center = d.site_xpos[m.site("monitor_center").id].copy()
    desk_z = float(d.site_xpos[m.site("desk_surface").id][2])
    wall_y = float(d.site_xpos[m.site("wall_front").id][1])
    roll_deg = monitor_world_roll_deg(m, d)
    is_portrait = roll_deg > 45.0

    # R-01: 눈-화면 중심 거리
    dist_m = float(np.linalg.norm(eye - center))

    # R-02: 높이 오차
    target_z, meas_z, ref_label = target_ref_z(m, d, center, eye, is_portrait)
    height_err_mm = abs(meas_z - target_z) * 1000.0

    # R-03: 화면 정면 법선(-Y_site)과 화면 중심->눈 벡터 사이 각도
    site_mat = d.site_xmat[m.site("monitor_center").id].reshape(3, 3)
    screen_normal = -site_mat[:, 1]
    to_eye = eye - center
    to_eye /= np.linalg.norm(to_eye)
    cos_ang = float(np.clip(np.dot(screen_normal, to_eye), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_ang))

    # R-10: 모니터 8개 꼭짓점·팔꿈치 기준 책상·뒷벽 여유 및 전체 접촉
    min_z, _, _ = monitor_extents(m, d)
    desk_clear_mm = (min_z - desk_z) * 1000.0
    wall_clear_mm = (wall_y - arm_max_y(m, d)) * 1000.0

    ok = (
        DIST_MIN_M <= dist_m <= DIST_MAX_M
        and height_err_mm <= TOP_TOL_MM
        and angle_deg <= ANGLE_MAX_DEG
        and desk_clear_mm >= CLEAR_MIN_MM
        and wall_clear_mm >= CLEAR_MIN_MM
        and d.ncon == 0
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{key_name:12s}] {status} | 거리 {dist_m:.3f} m (R-01) | "
        f"{ref_label} {meas_z:.3f}/목표 {target_z:.3f} m (오차 {height_err_mm:.1f} mm, R-02) | "
        f"각도차 {angle_deg:.1f}도 (R-03) | 피벗 {roll_deg:.0f}도 | "
        f"책상여유 {desk_clear_mm:.0f} mm · 벽여유 {wall_clear_mm:.0f} mm · 접촉 {d.ncon}건 (R-10)"
    )
    return ok


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    moving = [
        "arm_turret",
        "arm_upper",
        "arm_forearm",
        "wrist_yaw_body",
        "wrist_pitch_body",
        "monitor_mount",
        "webcam_unit",
    ]
    moving_mass = round(sum(float(m.body(n).mass[0]) for n in moving), 3)
    monitor_mass = round(float(m.body("monitor_mount").mass[0]), 3)

    print(f"모델: {SCENE.name} (MuJoCo {mujoco.__version__})")
    print(
        f"관절 {m.njnt}개 {[m.joint(i).name for i in range(m.njnt)]}, "
        f"액추에이터 {m.nu}개 {[m.actuator(i).name for i in range(m.nu)]}, "
        f"카메라 {m.ncam}개 {[m.camera(i).name for i in range(m.ncam)]}"
    )
    print(f"구동부 질량 합 {moving_mass} kg (모니터+VESA {monitor_mass} kg)")

    assert m.njnt == 6 and m.nu == 6, "6축(관절 6, 액추에이터 6) 구성이 아님"

    all_ok = True
    for key in ("seated", "lean_forward", "reclined", "standing", "portrait"):
        all_ok &= check_pose(m, key)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
