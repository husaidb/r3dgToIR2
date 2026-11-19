import os
import json
import argparse

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render


MASK_EXTS = (".png", ".jpg", ".jpeg")


def find_mask(masks_dir, image_name):
    for ext in MASK_EXTS:
        path = os.path.join(masks_dir, image_name + ext)
        if os.path.exists(path):
            return path
    return None


def load_foreground_mask(path, height, width):
    """返回 [H,W] float32，前景=1，背景=0；支持灰度、二值、RGBA。"""
    m = imageio.imread(path)
    if m.ndim == 3:
        if m.shape[2] == 4:
            m = m[..., 3]
        else:
            m = cv2.cvtColor(m[..., :3], cv2.COLOR_RGB2GRAY)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    if m.shape[:2] != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def project_points(xyz, cam):
    """返回像素坐标 u,v 与相机坐标 z。与 neilf.py 的约定一致。"""
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    z = (xyz_h @ cam.world_view_transform)[:, 2]
    uv = xyz_h @ cam.proj_matrix.T          # proj_matrix = K @ E
    u = uv[:, 0] / uv[:, 3].clamp_min(1e-8)
    v = uv[:, 1] / uv[:, 3].clamp_min(1e-8)
    return u, v, z


def main():
    parser = argparse.ArgumentParser(description="L2: object-mask metalness prior")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="阶段一 render 的 chkpnt*.pth（阶段二必须用同一个）")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="每视角前景 mask 目录，文件名 = image_name + 扩展名")
    parser.add_argument("--metallic", type=float, default=None,
                        help="直接指定物体 metalness，优先于 --scene_meta")
    parser.add_argument("--scene_meta", type=str, default="",
                        help="scene_meta.json，读取其中的 metallic / metal_prob")
    parser.add_argument("--out_dir", type=str, default="",
                        help="输出目录，默认 <source_path>/material_prior")
    parser.add_argument("--margin_abs", type=float, default=0.01,
                        help="深度容差绝对值（场景单位）")
    parser.add_argument("--margin_rel", type=float, default=0.05,
                        help="深度容差相对值（相对像素深度）")
    parser.add_argument("--min_opacity", type=float, default=0.1,
                        help="渲染不透明度低于该值的像素不参与可见性判定")
    parser.add_argument("--erode", type=int, default=0,
                        help="投影前对 2D mask 腐蚀的半径（像素），0 表示不腐蚀")
    parser.add_argument("--vote_thresh", type=float, default=0.5,
                        help="输出二值 mask 时使用的阈值，仅用于审计")
    parser.add_argument("--fill_bbox", action="store_true",
                        help="用已观测物体点的 3D bbox 填充未观测点（补物体背面）")
    args = parser.parse_args()

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)

    # ---------- 物体级 m ----------
    if args.metallic is not None:
        m_obj, m_source = float(args.metallic), "cli"
    elif args.scene_meta:
        with open(args.scene_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m_obj = float(meta.get("metallic", meta.get("metal_prob", 0.0)))
        m_source = "scene_meta"
        conf = float(meta.get("confidence", 1.0))
        if conf < 0.5:
            print(f"[WARN] low VLM confidence {conf:.2f}; please review before trusting m={m_obj:.2f}")
    else:
        raise ValueError("需要 --metallic 或 --scene_meta 提供物体级 metalness")

    # ---------- 加载阶段一几何 ----------
    gaussians = GaussianModel(dataset.sh_degree, render_type="render")
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)

    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    cams = scene.getTrainCameras()
    print(f"[INFO] {N} gaussians, {len(cams)} train cameras")

    votes = torch.zeros(N, device="cuda")
    weights = torch.zeros(N, device="cuda")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    skipped = 0

    # ---------- 逐视角投影投票 ----------
    for cam in tqdm(cams, desc="project & vote"):
        mask_path = find_mask(args.masks_dir, cam.image_name)
        if mask_path is None:
            skipped += 1
            continue

        H, W = cam.image_height, cam.image_width
        fg_np = load_foreground_mask(mask_path, H, W)
        if args.erode > 0:
            k = 2 * args.erode + 1
            kernel = np.ones((k, k), np.uint8)
            fg_np = cv2.erode(fg_np, kernel)
        fg = torch.from_numpy(fg_np).cuda()

        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, 1.0, None)
            depth = pkg["depth"][0]        # [H,W]
            opacity = pkg["opacity"][0]    # [H,W]
            u, v, z = project_points(xyz, cam)

        valid = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u.round().long().clamp(0, W - 1)
        vi = v.round().long().clamp(0, H - 1)
        d_px = depth[vi, ui]
        o_px = opacity[vi, ui]

        # 单边深度测试：z <= 可见表面深度 + 容差。
        # 包含物体正面和背面，排除物体后方的背景；物体前方的少量浮空点会被计入。
        tol = args.margin_abs + args.margin_rel * d_px
        visible = valid & (d_px > 0) & (o_px > args.min_opacity) & (z <= d_px + tol)

        votes[visible] += fg[vi, ui][visible]
        weights[visible] += 1.0

    if skipped:
        print(f"[WARN] {skipped} views have no mask and were skipped")

    object_prob = torch.where(weights > 0, votes / weights.clamp_min(1.0),
                              torch.zeros_like(votes))

    ratio = float((object_prob > args.vote_thresh).float().mean().item())
    if ratio > 0.5:
        print(f"[WARN] object ratio {ratio:.1%} > 50%: mask likely includes the table/background")
    elif ratio < 0.001:
        print(f"[WARN] object ratio {ratio:.3%} < 0.1%: check mask filenames / camera association")

    # ---------- 可选：补未观测到的物体点 ----------
    if args.fill_bbox:
        observed_obj = object_prob > 0.5
        if observed_obj.sum() > 100:
            pts = xyz[observed_obj]
            lo = torch.quantile(pts, 0.01, dim=0)
            hi = torch.quantile(pts, 0.99, dim=0)
            pad = 0.05 * (hi - lo)
            inside = ((xyz >= lo - pad) & (xyz <= hi + pad)).all(dim=-1)
            fill = (weights == 0) & inside
            object_prob = torch.maximum(object_prob, fill.float())
            print(f"[INFO] bbox fill: {int(fill.sum())} unobserved points set to object")

    metal_prob = object_prob * m_obj

    # ---------- 保存与统计 ----------
    out_dir = args.out_dir or os.path.join(dataset.source_path, "material_prior")
    os.makedirs(out_dir, exist_ok=True)
    obj_path = os.path.join(out_dir, "object_prob.npy")
    met_path = os.path.join(out_dir, "metal_prob.npy")
    np.save(obj_path, object_prob.cpu().numpy().astype(np.float32))
    np.save(met_path, metal_prob.cpu().numpy().astype(np.float32))

    stats = {
        "num_gaussians": int(N),
        "num_views": len(cams),
        "num_views_skipped": skipped,
        "observed_ratio": float((weights > 0).float().mean().item()),
        "object_ratio_at_thresh": float((object_prob > args.vote_thresh).float().mean().item()),
        "metallic": m_obj,
        "metallic_source": m_source,
        "margin_abs": args.margin_abs,
        "margin_rel": args.margin_rel,
        "erode": args.erode,
        "vote_thresh": args.vote_thresh,
    }
    with open(os.path.join(out_dir, "metal_prior_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[DONE] saved {obj_path} and {met_path}")


if __name__ == "__main__":
    main()
