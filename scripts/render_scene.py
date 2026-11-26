"""D07 장면 v0 앉기·서기 스크린샷 렌더 스크립트.

실행 예:
    uv run --with pillow python scripts/render_scene.py
"""
from pathlib import Path
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

    eye_z = float(d.site_xpos[m.site("eye_center").id][2])
    top_z = float(d.site_xpos[m.site("monitor_top").id][2])

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    # 왼쪽: 3D 측면 대각선 시점 (책상·암·모니터·사람 머리·눈 표시)
    w_main, h_main = 1000, 720
    r_main = mujoco.Renderer(m, h_main, w_main)
    r_main.update_scene(d, free_cam([0.0, -0.12, 1.02], 2.25, 168, -12), opt)
    im_main = Image.fromarray(r_main.render())
    r_main.close()

    # 오른쪽: 모니터 상단 웹캠 시점 (눈이 화각 안에 들어오는지 확인)
    w_cam, h_cam = 600, 720
    r_cam = mujoco.Renderer(m, h_cam, w_cam)
    r_cam.update_scene(d, "webcam", opt)
    im_cam = Image.fromarray(r_cam.render())
    r_cam.close()

    canvas = Image.new("RGB", (w_main + w_cam, h_main), (255, 255, 255))
    canvas.paste(im_main, (0, 0))
    canvas.paste(im_cam, (w_main, 0))

    dr = ImageDraw.Draw(canvas, "RGBA")
    dr.line([(w_main, 0), (w_main, h_main)], fill=(210, 214, 220, 255), width=3)
    f_title = load_font(22)
    f_sub = load_font(18)

    draw_tag(
        dr,
        (18, 16),
        f"{label} — 눈높이 {eye_z:.2f} m / 화면 상단 {top_z:.2f} m",
        f_title,
    )
    draw_tag(
        dr,
        (18, h_main - 42),
        "빨간 점: 눈 중심(eye_center) · 초록 점: 화면 상단(monitor_top) · 파란 점: 웹캠",
        f_sub,
    )
    draw_tag(dr, (w_main + 18, 16), "웹캠 시점 (webcam, fovy 75°)", f_title)

    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / out_name)


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    render_pose(m, "seated", "앉은 자세 (seated)", "scene_seated.png")
    render_pose(m, "standing", "선 자세 (standing)", "scene_standing.png")
    print("saved:", OUT / "scene_seated.png", OUT / "scene_standing.png")


if __name__ == "__main__":
    main()
