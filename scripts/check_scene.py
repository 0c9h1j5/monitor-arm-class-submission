"""D07 장면 구성표 확인 코드 (scenes/monitor_arm.xml).

실행 예:
    uv run python scripts/check_scene.py
"""
from pathlib import Path
import mujoco

SCENE = Path(__file__).resolve().parent.parent / "scenes" / "monitor_arm.xml"


def check_pose(m: mujoco.MjModel, key_name: str) -> None:
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    # 1초 동안 물리 진행 후 안정성·접촉·높이 오차 확인
    for _ in range(500):
        mujoco.mj_step(m, d)

    eye_z = float(d.site_xpos[m.site("eye_center").id][2])
    top_z = float(d.site_xpos[m.site("monitor_top").id][2])
    bot_z = float(d.site_xpos[m.site("monitor_bottom").id][2])
    desk_z = float(d.site_xpos[m.site("desk_surface").id][2])
    err_mm = abs(top_z - eye_z) * 1000.0
    clear_mm = (bot_z - desk_z) * 1000.0

    print(
        f"[{key_name}] eye_z={eye_z:.3f} m, monitor_top_z={top_z:.3f} m "
        f"(오차 {err_mm:.1f} mm), 책상 여유 {clear_mm:.0f} mm, 접촉 {d.ncon}건"
    )


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    moving = ["arm_carriage", "monitor_mount", "webcam_unit"]
    moving_mass = round(sum(float(m.body(n).mass[0]) for n in moving), 3)
    monitor_mass = round(float(m.body("monitor_mount").mass[0]), 3)

    print(f"모델: {SCENE.name} (MuJoCo {mujoco.__version__})")
    print(
        f"관절 {m.njnt}개 {[m.joint(i).name for i in range(m.njnt)]}, "
        f"액추에이터 {m.nu}개 {[m.actuator(i).name for i in range(m.nu)]}, "
        f"카메라 {m.ncam}개 {[m.camera(i).name for i in range(m.ncam)]}"
    )
    print(f"구동부 질량 합 {moving_mass} kg (모니터+VESA {monitor_mass} kg)")
    check_pose(m, "seated")
    check_pose(m, "standing")


if __name__ == "__main__":
    main()
