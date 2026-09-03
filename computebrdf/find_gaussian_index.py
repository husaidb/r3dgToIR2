import argparse

import numpy as np
import torch

from scene.gaussian_model import GaussianModel


def find_index_by_xyz(ply_path, x, y, z):
    gaussians = GaussianModel(sh_degree=3, render_type="neilf")
    gaussians.load_ply(ply_path)

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    target = np.array([x, y, z], dtype=np.float32)
    diff = xyz - target[None, :]
    dist2 = np.sum(diff * diff, axis=-1)
    idx = int(np.argmin(dist2))
    closest = xyz[idx]

    print(f"input_xyz: {target[0]:.6f} {target[1]:.6f} {target[2]:.6f}")
    print(f"closest_xyz: {closest[0]:.6f} {closest[1]:.6f} {closest[2]:.6f}")
    print(f"index: {idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    args = parser.parse_args()

    find_index_by_xyz(args.ply, args.x, args.y, args.z)

