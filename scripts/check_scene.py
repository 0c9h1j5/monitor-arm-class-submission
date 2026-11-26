"""D07 장면 구성표 확인 코드 (scenes/monitor_arm.xml).

실행 예:
    uv run python scripts/check_scene.py
"""
from pathlib import Path
import math
import numpy as np
import mujoco

SCENE = Path(__file__).resolve().parent.parent / "scenes" / "monitor_arm.xml"

# R-01, R-02, R-03, R-09 기준
DIST_MIN_M, DIST_MAX_M = 0.50, 0.70
TOP_TOL_MM = 15.0
ANGLE_MAX_DEG = 10.0
CLEAR_MIN_MM = 50.0


def head_gaze_target_top_z(m: mujoco.MjModel, d: mujoco.MjData, screen_center: np.ndarray, eye: np.ndarray) -> float:
    """머리 정면 시선 기준 목표 화면 상단 높이.

    똑바로 앉기·앞으로 숙이기·서기처럼 머리가 수평이거나 약간 숙인 자세는 수평 눈높이(eye_z)를 쓴다.
    기대 누워 머리가 위를 향한 자세(+x 회전 > 0)는 머리 정면 시선선이 화면 거리에 닿는 높이를 쓴다.
    """
    head_id = m.body("user_head").id
    head_xmat = d.xmat[head_id].reshape(3, 3)
    gaze_dir = head_xmat[:, 1]  # 머리 로컬 +Y가 얼굴 정면
    if gaze_dir[2] > 0.05:  # 위를 향해 기대 누운 자세
        dy = float(screen_center[1] - eye[1])
        return float(eye[2] + dy * (gaze_dir[2] / gaze_dir[1]))
    return float(eye[2])


def check_pose(m: mujoco.MjModel, key_name: str) -> bool:
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    for _ in range(500):  # 1.0 s 물리 진행
        mujoco.mj_step(m, d)

    eye = d.site_xpos[m.site("eye_center").id].copy()
    center = d.site_xpos[m.site("monitor_center").id].copy()
    top = d.site_xpos[m.site("monitor_top").id].copy()
    bot = d.site_xpos[m.site("monitor_bottom").id].copy()
    back = d.site_xpos[m.site("monitor_back").id].copy()
    desk_z = float(d.site_xpos[m.site("desk_surface").id][2])
    wall_y = float(d.site_xpos[m.site("wall_front").id][1])

    # R-01: 눈-화면 중심 거리
    dist_m = float(np.linalg.norm(eye - center))

    # R-02: 화면 상단 높이 오차
    target_top_z = head_gaze_target_top_z(m, d, center, eye)
    top_err_mm = abs(float(top[2]) - target_top_z) * 1000.0

    # R-03: 화면 정면 법선(-Y_site)과 화면 중심->눈 벡터 사이 각도
    site_mat = d.site_xmat[m.site("monitor_center").id].reshape(3, 3)
    screen_normal = -site_mat[:, 1]
    to_eye = eye - center
    to_eye /= np.linalg.norm(to_eye)
    cos_ang = float(np.clip(np.dot(screen_normal, to_eye), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_ang))

    # R-09: 책상·뒷벽 여유 및 접촉
    desk_clear_mm = (float(bot[2]) - desk_z) * 1000.0
    wall_clear_mm = (wall_y - float(back[1])) * 1000.0

    ok = (
        DIST_MIN_M <= dist_m <= DIST_MAX_M
        and top_err_mm <= TOP_TOL_MM
        and angle_deg <= ANGLE_MAX_DEG
        and desk_clear_mm >= CLEAR_MIN_MM
        and wall_clear_mm >= CLEAR_MIN_MM
        and d.ncon == 0
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{key_name:12s}] {status} | 거리 {dist_m:.3f} m (R-01) | "
        f"상단 {top[2]:.3f}/목표 {target_top_z:.3f} m (오차 {top_err_mm:.1f} mm, R-02) | "
        f"각도차 {angle_deg:.1f}도 (R-03) | "
        f"책상여유 {desk_clear_mm:.0f} mm · 벽여유 {wall_clear_mm:.0f} mm · 접촉 {d.ncon}건 (R-09)"
    )
    return ok


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    moving = ["arm_carriage", "arm_slider", "monitor_yaw_body", "monitor_mount", "webcam_unit"]
    moving_mass = round(sum(float(m.body(n).mass[0]) for n in moving), 3)
    monitor_mass = round(float(m.body("monitor_mount").mass[0]), 3)

    print(f"모델: {SCENE.name} (MuJoCo {mujoco.__version__})")
    print(
        f"관절 {m.njnt}개 {[m.joint(i).name for i in range(m.njnt)]}, "
        f"액추에이터 {m.nu}개 {[m.actuator(i).name for i in range(m.nu)]}, "
        f"카메라 {m.ncam}개 {[m.camera(i).name for i in range(m.ncam)]}"
    )
    print(f"구동부 질량 합 {moving_mass} kg (모니터+VESA {monitor_mass} kg)")

    all_ok = True
    for key in ("seated", "lean_forward", "reclined", "standing"):
        all_ok &= check_pose(m, key)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
