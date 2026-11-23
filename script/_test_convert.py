"""COLMAP→Blender 转换器的 CPU 测试：python script/_test_convert.py"""
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import plyfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "script"))
import convert_colmap_to_blender  # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"{name}: OK")
    else:
        print(f"{name}: FAIL {detail}")
        failures.append(name)


def write_text_fixture(root, image_writer, cameras=None):
    """构造 text 格式 COLMAP 夹具：2 台相机、1 张图实际存在。"""
    sparse = os.path.join(root, "sparse")
    images = os.path.join(root, "images")
    os.makedirs(sparse)
    os.makedirs(images)

    if cameras is None:
        cameras = [
            "1 PINHOLE 8 6 10 10 4 3",
            "2 PINHOLE 8 6 10 10 4 3",
        ]
    with open(os.path.join(sparse, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write("\n".join(cameras) + "\n")

    q = math.sqrt(0.5)
    with open(os.path.join(sparse, "images.txt"), "w") as f:
        f.write("# Image list\n")
        # 相机 1：单位旋转，tvec=[0,0,5]
        f.write("1 1 0 0 0 0 0 5 1 img1.png\n")
        f.write("4 3 -1 5 5 -1\n")
        # 相机 2：绕 Z 轴 90°，图片不存在 → 应被跳过
        f.write(f"2 {q} 0 0 {q} 1 0 5 1 missing.png\n")
        f.write("4 3 -1\n")

    with open(os.path.join(sparse, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for i in range(5):
            f.write(f"{i + 1} {0.1 * i} 0 0 {255 - i} 0 0 0.5 1 0 2 0\n")

    image_writer(os.path.join(images, "img1.png"))
    return sparse, images


def write_gray16(path):
    img = np.zeros((6, 8), dtype=np.uint16)
    img[1:3, 2:5] = 60000
    cv2.imwrite(path, img)


def write_bgra(path):
    img = np.zeros((6, 8, 4), dtype=np.uint8)
    img[..., 2] = 255          # R
    img[..., 3] = 0            # alpha=0 也必须保留 RGB
    cv2.imwrite(path, img)


def test_main_fixture():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sparse, images = write_text_fixture(tmp, write_gray16)
        out = os.path.join(tmp, "out")
        info = convert_colmap_to_blender.convert_colmap_to_blender(sparse, images, out)

        check("frames skip missing image", info["frames"] == 1, str(info))
        check("points", info["points"] == 5, str(info))
        check("model", info["camera_models"] == ["PINHOLE"], str(info))

        with open(os.path.join(out, "transforms_train.json")) as f:
            contents = json.load(f)
        check("camera_angle_x",
              abs(contents["camera_angle_x"] - 2 * math.atan(8 / (2 * 10))) < 1e-9,
              str(contents.get("camera_angle_x")))
        frames = contents["frames"]
        check("one frame", len(frames) == 1)
        check("file_path", frames[0]["file_path"] == "train/img1", frames[0]["file_path"])

        M = np.array(frames[0]["transform_matrix"])
        check("json 4x4", M.shape == (4, 4) and np.allclose(M[3], [0, 0, 0, 1]))

        # 复刻 Blender 读取器：先取反第 1、2 列，再求逆
        c2w = M.copy()
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        check("reader pose roundtrip",
              np.allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
              and np.allclose(w2c[:3, 3], [0, 0, 5], atol=1e-6),
              f"R={w2c[:3, :3]} t={w2c[:3, 3]}")

        img = cv2.imread(os.path.join(out, "train", "img1.png"), cv2.IMREAD_UNCHANGED)
        check("8-bit 3ch",
              img is not None and img.dtype == np.uint8 and img.ndim == 3
              and img.shape[2] == 3,
              f"{None if img is None else (img.dtype, img.shape)}")

        ply = plyfile.PlyData.read(os.path.join(out, "points3d.ply"))
        check("ply count", len(ply["vertex"]) == 5)
        check("no sparse/masks",
              not os.path.exists(os.path.join(out, "sparse"))
              and not os.path.exists(os.path.join(out, "masks")))

        test_reader_guard(out)


def test_rgba_drops_alpha():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sparse, images = write_text_fixture(tmp, write_bgra)
        out = os.path.join(tmp, "out")
        convert_colmap_to_blender.convert_colmap_to_blender(sparse, images, out)
        img = cv2.imread(os.path.join(out, "train", "img1.png"), cv2.IMREAD_UNCHANGED)
        check("rgba 3ch", img.ndim == 3 and img.shape[2] == 3, str(img.shape))
        check("rgba keeps color", img[..., 2].mean() > 250, str(img[..., 2].mean()))


def test_non_pinhole_text():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sparse, images = write_text_fixture(
            tmp, write_gray16, cameras=["1 SIMPLE_RADIAL 8 6 10 4 3 0.0"])
        out = os.path.join(tmp, "out")
        try:
            convert_colmap_to_blender.convert_colmap_to_blender(sparse, images, out)
        except ValueError as e:
            msg = str(e)
            check("non-pinhole text error", "PINHOLE" in msg and "undistort" in msg, msg)
        else:
            check("non-pinhole text error", False, "没有抛 ValueError")


def test_cli_utf8_output():
    import subprocess
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sparse, images = write_text_fixture(
            tmp, write_gray16, cameras=["1 SIMPLE_RADIAL 8 6 10 4 3 0.0"])
        env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "script", "convert_colmap_to_blender.py"),
             "--sparse", sparse, "--images", images, "--out", os.path.join(tmp, "out")],
            capture_output=True, env=env)
        try:
            text = r.stdout.decode("utf-8")
        except UnicodeDecodeError as e:
            check("cli utf8", False, str(e))
            return
        check("cli utf8", r.returncode == 1 and "[ERROR]" in text and "PINHOLE" in text,
              text[-200:])


def test_reader_guard(out):
    try:
        from scene.dataset_readers import readNerfSyntheticInfo
    except ImportError:
        print("readNerfSyntheticInfo: SKIP (本地缺 simple_knn/pyexr)")
        return
    cams = readNerfSyntheticInfo(out, False, False).train_cameras
    check("reader cams", len(cams) == 1 and cams[0].image.shape[-1] == 3, str(len(cams)))


def main():
    test_main_fixture()
    test_rgba_drops_alpha()
    test_non_pinhole_text()
    test_cli_utf8_output()
    print(f"\nFAILURES: {len(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
