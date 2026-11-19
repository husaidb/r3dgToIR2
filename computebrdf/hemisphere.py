import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaussian_renderer.neilf import GGX_specular
from scene.gaussian_model import GaussianModel, sample_incident_rays


@torch.no_grad()
def compute_reflectance(gaussians, sample_num=64, chunk_size=50000):
    base_color_all = gaussians.get_base_color
    roughness_all = gaussians.get_roughness
    normals_all = gaussians.get_normal

    rho_all, eps_all = [], []
    for offset in range(0, normals_all.shape[0], chunk_size):
        s = slice(offset, offset + chunk_size)
        base_color = base_color_all[s]
        roughness = roughness_all[s]
        normals = normals_all[s]
        # 视角沿法线，取逐点本征值；若需某视角下的值，把 viewdirs 换成该视角的单位方向
        viewdirs = normals

        incident_dirs, incident_areas = sample_incident_rays(normals, False, sample_num)
        f_d = base_color[:, None] / np.pi
        f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)
        n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)

        rho = ((f_d + f_s) * incident_areas * n_d_i).mean(dim=1).clamp(0, 1)
        rho_all.append(rho)
        eps_all.append(1.0 - rho)

    return torch.cat(rho_all).cpu().numpy(), torch.cat(eps_all).cpu().numpy()


def make_gaussians(checkpoint_path=None, ply_path=None):
    gaussians = GaussianModel(sh_degree=3, render_type="neilf")
    if checkpoint_path is not None:
        gaussians.create_from_ckpt(checkpoint_path, restore_optimizer=False)
    else:
        gaussians.load_ply(ply_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--out", type=str, default="reflectance.txt")
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    if args.checkpoint is None and args.ply is None:
        raise ValueError("checkpoint 或 ply 至少提供一个")

    gaussians = make_gaussians(args.checkpoint, args.ply)
    rho, eps = compute_reflectance(gaussians, args.sample_num, args.chunk_size)

    idx = np.arange(rho.shape[0])[:, None]
    data = np.concatenate([idx, rho, eps], axis=1)
    np.savetxt(args.out, data, fmt=["%d"] + ["%.6f"] * 6,
               header="index rho_r rho_g rho_b eps_r eps_g eps_b")
    print(f"wrote {data.shape[0]} points to {args.out}")
    print(f"mean rho: {rho.mean(axis=0)}")
    print(f"mean eps: {eps.mean(axis=0)}")
