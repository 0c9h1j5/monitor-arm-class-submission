"""D07 장면 v2(6축 SCARA) 5자세 스크린샷 렌더 스크립트.

실행 예:
    uv run --with pillow python scripts/render_scene.py
"""
from pathlib import Path
import math
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "scenes" / "monitor_arm.xml"
OUT = ROOT / "docs" / "img"
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def load_font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def free_cam(lookat, dist: float, az: float, el: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    return cam


def draw_tag(dr: ImageDraw.ImageDraw, xy, text: str, fnt) -> None:
    bb = dr.textbbox(xy, text, font=fnt)
    dr.rounded_rectangle(
        (bb[0] - 8, bb[1] - 5, bb[2] + 8, bb[3] + 5),
        radius=6,
        fill=(20, 24, 30, 185),
    )
    dr.text(xy, text, font=fnt, fill=(255, 255, 255))


def render_pose(m: mujoco.MjModel, key_name: str, label: str, out_name: str) -> None:
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)

    eye = d.site_xpos[m.site("eye_center").id].copy()
    center = d.site_xpos[m.site("monitor_center").id].copy()
    dist_cm = float(np.linalg.norm(eye - center)) * 100.0
    pitch_deg = math.degrees(float(d.qpos[m.jnt_qposadr[m.joint("wrist_pitch").id]]))
    roll_deg = math.degrees(float(d.qpos[m.jnt_qposadr[m.joint("wrist_roll").id]]))
    if abs(roll_deg) < 0.5:
        roll_deg = 0.0
    is_portrait = abs(roll_deg) > 45.0

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    # 왼쪽: 3D 측면 대각선 시점 (SCARA 2링크 + 전신·의자·책상·모니터가 보이게)
    w_main, h_main = 1040, 720
    r_main = mujoco.Renderer(m, h_main, w_main)
    r_main.update_scene(d, free_cam([0.05, -0.20, 0.90], 2.58, 160, -14), opt)
    im_main = Image.fromarray(r_main.render())
    r_main.close()

    # 오른쪽: 모니터 웹캠 시점 (세로형일 때는 90도 영상 보정 적용)
    w_cam, h_cam = 560, 720
    if is_portrait:
        r_cam = mujoco.Renderer(m, w_cam, h_cam)
        r_cam.update_scene(d, "webcam", opt)
        arr = np.rot90(r_cam.render(), k=1)
        im_cam = Image.fromarray(arr)
        r_cam.close()
        cam_label = "웹캠 시점 (세로형 90° 보정, fovy 75°)"
    else:
        r_cam = mujoco.Renderer(m, h_cam, w_cam)
        r_cam.update_scene(d, "webcam", opt)
        im_cam = Image.fromarray(r_cam.render())
        r_cam.close()
        cam_label = "웹캠 시점 (webcam, fovy 75°)"

    canvas = Image.new("RGB", (w_main + w_cam, h_main), (255, 255, 255))
    canvas.paste(im_main, (0, 0))
    canvas.paste(im_cam, (w_main, 0))

    dr = ImageDraw.Draw(canvas, "RGBA")
    dr.line([(w_main, 0), (w_main, h_main)], fill=(210, 214, 220, 255), width=3)
    f_title = load_font(20)
    f_sub = load_font(17)

    draw_tag(
        dr,
        (18, 16),
        f"{label} — 거리 {dist_cm:.0f} cm · 기울기 {pitch_deg:+.0f}° · 피벗 {roll_deg:.0f}°",
        f_title,
    )
    draw_tag(
        dr,
        (18, h_main - 40),
        "빨간 점: 눈 중심 · 노란 점: 화면 중심 · 초록 점: 화면 상단 · 파란 점: 웹캠",
        f_sub,
    )
    draw_tag(dr, (w_main + 18, 16), cam_label, f_title)

    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / out_name)


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    poses = [
        ("seated", "똑바로 앉기 (seated)", "scene_seated.png"),
        ("lean_forward", "앞으로 숙이기 (lean_forward)", "scene_lean_forward.png"),
        ("reclined", "의자에 기대 눕기 (reclined)", "scene_reclined.png"),
        ("standing", "서기 (standing)", "scene_standing.png"),
        ("portrait", "세로형 피벗 (portrait, 90°)", "scene_portrait.png"),
    ]
    for key, label, fname in poses:
        render_pose(m, key, label, fname)
    print("saved:", *[OUT / f for _, _, f in poses])


if __name__ == "__main__":
    main()
