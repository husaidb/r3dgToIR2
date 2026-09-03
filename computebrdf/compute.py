import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaussian_renderer.neilf import GGX_specular
from scene.gaussian_model import GaussianModel


def sample_brdf_to_txt(
    base_color,
    roughness,
    out_path="brdf_samples.txt",
    num_dirs=100,
    scan_mode="incoming",
    theta_i=0.0,
    phi_i=0.0,
    theta_o=0.0,
    phi_o=np.pi,
    max_theta=np.deg2rad(80.0),
    signed_scan=False,
):
    device = torch.device("cpu")

    base_color = torch.tensor(base_color, dtype=torch.float32, device=device).view(1, 3)
    roughness = float(roughness)
    roughness_tensor = torch.full((1, 3), roughness, dtype=torch.float32, device=device)

    def _dir_in_xz_plane(theta):
        return np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float32)

    if signed_scan:
        scan_thetas = np.linspace(-max_theta, max_theta, num_dirs)
    else:
        scan_thetas = np.linspace(0.0, max_theta, num_dirs)

    fresnel = 0.04

    if scan_mode == "incoming":
        N = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        V = torch.tensor(_dir_in_xz_plane(theta_o), dtype=torch.float32, device=device).view(1, 3)

        dirs = []
        for th in scan_thetas:
            dirs.append(_dir_in_xz_plane(th))
        dirs = torch.tensor(np.stack(dirs, axis=0), dtype=torch.float32, device=device)
        incident_dirs = dirs.unsqueeze(0)

        f_d = base_color.unsqueeze(1) / np.pi
        f_s = GGX_specular(
            normal=N,
            pts2c=V,
            pts2l=incident_dirs,
            roughness=roughness_tensor,
            fresnel=fresnel,
        )
        brdf = f_d + f_s
        brdf_np = brdf.squeeze(0).cpu().numpy()
    elif scan_mode == "outgoing":
        N = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(num_dirs, 1)

        v_dirs = []
        for th in scan_thetas:
            v_dirs.append(_dir_in_xz_plane(th))
        V = torch.tensor(np.stack(v_dirs, axis=0), dtype=torch.float32, device=device)

        L_single = torch.tensor(_dir_in_xz_plane(theta_i), dtype=torch.float32, device=device).view(1, 3)
        incident_dirs = L_single.view(1, 1, 3).repeat(num_dirs, 1, 1)

        roughness_rep = torch.full((num_dirs, 3), roughness, dtype=torch.float32, device=device)
        f_d = base_color.repeat(num_dirs, 1).unsqueeze(1) / np.pi
        f_s = GGX_specular(
            normal=N,
            pts2c=V,
            pts2l=incident_dirs,
            roughness=roughness_rep,
            fresnel=fresnel,
        )
        brdf = f_d + f_s
        brdf_np = brdf.squeeze(1).cpu().numpy()
    else:
        raise ValueError(f"Unknown scan_mode: {scan_mode}")

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_dirs):
            if scan_mode == "incoming":
                theta_i_i = float(scan_thetas[i])
                phi_i_i = float(phi_i)
                theta_o_i = float(theta_o)
                phi_o_i = float(phi_o)
            else:
                theta_i_i = float(theta_i)
                phi_i_i = float(phi_i)
                theta_o_i = float(scan_thetas[i])
                phi_o_i = float(phi_o)

            brdf_i = brdf_np[i]

            line = (
                f"{theta_i_i:.5f}, {phi_i_i:.5f}, "
                f"{theta_o_i:.5f}, {phi_o_i:.5f}, "
                f"{brdf_i[0]:.3f}, {brdf_i[1]:.3f}, {brdf_i[2]:.3f}\n"
            )
            f.write(line)

    print(f"BRDF samples written to: {out_path}")


def sample_brdf_for_gaussian(
    checkpoint_path=None,
    ply_path=None,
    index=0,
    out_path="brdf_samples_ggx.txt",
    num_dirs=100,
    scan_mode="incoming",
    theta_i=0.0,
    phi_i=0.0,
    theta_o=0.0,
    phi_o=np.pi,
    max_theta=np.deg2rad(80.0),
    signed_scan=False,
):
    if checkpoint_path is None and ply_path is None:
        raise ValueError("checkpoint_path 或 ply_path 至少提供一个")

    gaussians = GaussianModel(sh_degree=3, render_type="neilf")

    if checkpoint_path is not None:
        gaussians.create_from_ckpt(checkpoint_path, restore_optimizer=False)
    else:
        gaussians.load_ply(ply_path)

    base_color_all = gaussians.get_base_color.detach().cpu().numpy()
    roughness_all = gaussians.get_roughness.detach().cpu().numpy()

    if index < 0 or index >= base_color_all.shape[0]:
        raise IndexError(f"index 超出范围: 0 <= index < {base_color_all.shape[0]}")

    base_color = base_color_all[index].tolist()
    roughness = float(roughness_all[index, 0])

    sample_brdf_to_txt(
        base_color=base_color,
        roughness=roughness,
        out_path=out_path,
        num_dirs=num_dirs,
        scan_mode=scan_mode,
        theta_i=theta_i,
        phi_i=phi_i,
        theta_o=theta_o,
        phi_o=phi_o,
        max_theta=max_theta,
        signed_scan=signed_scan,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default="brdf_samples_ggx.txt")
    parser.add_argument("--num_dirs", type=int, default=100)
    parser.add_argument("--base_color", type=float, nargs=3, default=None)
    parser.add_argument("--roughness", type=float, default=None)
    parser.add_argument("--scan_mode", type=str, default="incoming", choices=["incoming", "outgoing"])
    parser.add_argument("--theta_i", type=float, default=0.0)
    parser.add_argument("--phi_i", type=float, default=0.0)
    parser.add_argument("--theta_o", type=float, default=0.0)
    parser.add_argument("--phi_o", type=float, default=float(np.pi))
    parser.add_argument("--max_theta", type=float, default=float(np.deg2rad(80.0)))
    parser.add_argument("--signed_scan", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is not None or args.ply is not None:
        sample_brdf_for_gaussian(
            checkpoint_path=args.checkpoint,
            ply_path=args.ply,
            index=args.index,
            out_path=args.out,
            num_dirs=args.num_dirs,
            scan_mode=args.scan_mode,
            theta_i=args.theta_i,
            phi_i=args.phi_i,
            theta_o=args.theta_o,
            phi_o=args.phi_o,
            max_theta=args.max_theta,
            signed_scan=args.signed_scan,
        )
    else:
        if args.base_color is None or args.roughness is None:
            raise ValueError("未提供模型路径时，必须指定 base_color 和 roughness")
        sample_brdf_to_txt(
            base_color=args.base_color,
            roughness=args.roughness,
            out_path=args.out,
            num_dirs=args.num_dirs,
            scan_mode=args.scan_mode,
            theta_i=args.theta_i,
            phi_i=args.phi_i,
            theta_o=args.theta_o,
            phi_o=args.phi_o,
            max_theta=args.max_theta,
            signed_scan=args.signed_scan,
        )
