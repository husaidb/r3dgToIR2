"""把 COLMAP 输出（sparse）转换成 Relightable3DGaussian 训练用的 Blender 格式。

用法：
    python script/convert_colmap_to_blender.py \
        --sparse datasets/<scene>/raw/sparse/0 \
        --images datasets/<scene>/raw/images \
        --out work/<scene>/data

产物：transforms_train.json、train/<stem>.png、points3d.ply。
只依赖 numpy/cv2/plyfile，不导入 torch，可纯 CPU 运行。
"""
import argparse
import importlib.util
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# 直接按文件加载：import scene.colmap_loader 会执行 scene/__init__ → torch/simple_knn
# （colmap_loader 本身只依赖 numpy）。
_spec = importlib.util.spec_from_file_location(
    "_colmap_loader", os.path.join(REPO, "scene", "colmap_loader.py"))
colmap_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(colmap_loader)

qvec2rotmat = colmap_loader.qvec2rotmat
read_extrinsics_binary = colmap_loader.read_extrinsics_binary
read_extrinsics_text = colmap_loader.read_extrinsics_text
read_intrinsics_binary = colmap_loader.read_intrinsics_binary
read_intrinsics_text = colmap_loader.read_intrinsics_text
read_points3D_binary = colmap_loader.read_points3D_binary
read_points3D_text = colmap_loader.read_points3D_text

from utils.ply_utils import storePly  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _load_sparse(sparse_dir):
    def pick(name):
        bin_path = os.path.join(sparse_dir, name + ".bin")
        txt_path = os.path.join(sparse_dir, name + ".txt")
        if os.path.exists(bin_path):
            return bin_path
        if os.path.exists(txt_path):
            return txt_path
        raise ValueError(f"{sparse_dir} 下缺少 {name}.bin 或 {name}.txt")

    cams_path = pick("cameras")
    imgs_path = pick("images")
    pts_path = pick("points3D")

    if cams_path.endswith(".bin"):
        cameras = read_intrinsics_binary(cams_path)
    else:
        try:
            cameras = read_intrinsics_text(cams_path)
        except AssertionError:
            with open(cams_path) as f:
                model = next((ln.split()[1] for ln in f
                              if ln.strip() and not ln.startswith("#")), "?")
            raise ValueError(
                f"text 格式只支持 PINHOLE 相机（当前是 {model}）。"
                "请用 binary 输出，或先运行 colmap image_undistorter 再转换。")

    images = (read_extrinsics_binary(imgs_path) if imgs_path.endswith(".bin")
              else read_extrinsics_text(imgs_path))
    if pts_path.endswith(".bin"):
        xyz, rgb, _ = read_points3D_binary(pts_path)
    else:
        xyz, rgb, _ = read_points3D_text(pts_path)
    if xyz is None or len(xyz) == 0:
        raise ValueError(f"{pts_path} 中没有 3D 点，无法初始化点云")
    return cameras, images, xyz, rgb


_WARNED_MODELS = set()


def _camera_params(cam):
    """返回 (fx, fy, cx, cy)；忽略畸变参数。"""
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if cam.model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if cam.model == "SIMPLE_RADIAL":
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
    elif cam.model == "RADIAL":
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
    elif cam.model == "OPENCV":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(f"不支持的 COLMAP 相机模型 {cam.model}，请先 undistort")
    if cam.model not in _WARNED_MODELS:
        _WARNED_MODELS.add(cam.model)
        print(f"[WARN] 忽略 {cam.model} 的畸变参数，请确认镜头畸变较小或已 undistort")
    return fx, fy, cx, cy


def _convert_image(src, dst):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片 {src}")
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255.0).round().clip(0, 255).astype(np.uint8)
        print(f"[WARN] {os.path.basename(src)} 是 16-bit，已线性压到 8-bit")
    elif img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        print(f"[WARN] {os.path.basename(src)} 是浮点图，已按 [0,1] 压到 8-bit")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(dst, img)


def convert_colmap_to_blender(sparse_dir, images_dir, out_dir):
    cameras, images, xyz, rgb = _load_sparse(sparse_dir)
    train_dir = os.path.join(out_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    entries, stems = [], set()
    for image in images.values():
        basename = os.path.basename(image.name)
        stem = os.path.splitext(basename)[0]
        src = os.path.join(images_dir, basename)
        if not os.path.exists(src):
            print(f"[WARN] 图片缺失，跳过 {image.name}")
            continue
        if stem in stems:
            raise ValueError(f"图片 basename 冲突：{stem}（请保证文件名唯一）")
        stems.add(stem)
        _convert_image(src, os.path.join(train_dir, stem + ".png"))
        entries.append((image.name, stem, image))

    if not entries:
        raise ValueError("没有任何已注册相机能找到对应图片")

    entries.sort(key=lambda e: e[0])

    frames = []
    camera_angle_x = None
    for _, stem, image in entries:
        cam = cameras[image.camera_id]
        fx, _, _, _ = _camera_params(cam)
        if camera_angle_x is None:
            camera_angle_x = 2.0 * math.atan(cam.width / (2.0 * fx))
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1          # 抵消 Blender 读取器的 c2w[:3,1:3] *= -1
        frames.append({
            "file_path": "train/" + stem,
            "transform_matrix": c2w.tolist(),
        })

    with open(os.path.join(out_dir, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": frames}, f, indent=2)

    storePly(os.path.join(out_dir, "points3d.ply"), xyz, rgb)
    if len(xyz) < 1000:
        print(f"[WARN] 点云只有 {len(xyz)} 个点，重建可能不稳定")

    return {
        "frames": len(frames),
        "points": int(len(xyz)),
        "camera_models": sorted({cameras[img.camera_id].model for _, _, img in entries}),
    }


def main():
    parser = argparse.ArgumentParser(description="COLMAP sparse → Blender 格式")
    parser.add_argument("--sparse", required=True, help="含 cameras/images/points3D 的目录")
    parser.add_argument("--images", required=True, help="原始图片目录")
    parser.add_argument("--out", required=True, help="输出目录（数据根）")
    args = parser.parse_args()
    try:
        info = convert_colmap_to_blender(args.sparse, args.images, args.out)
    except ValueError as e:
        print("[ERROR]", e)
        sys.exit(1)
    print(f"[INFO] frames={info['frames']} points={info['points']} "
          f"models={info['camera_models']}")


if __name__ == "__main__":
    main()
