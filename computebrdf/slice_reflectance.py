import argparse

import numpy as np


def integrate_slice(theta, brdf, max_theta=None):
    keep = theta >= 0
    if max_theta is not None:
        keep &= theta <= max_theta
    theta, brdf = theta[keep], brdf[keep]

    order = np.argsort(theta)
    theta, brdf = theta[order], brdf[order]

    # 假设扫描平面代表方位角平均：rho = ∫f cosθ dω ≈ 2π ∫f cosθ sinθ dθ
    integrand = brdf * np.cos(theta)[:, None] * np.sin(theta)[:, None]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rho = trapz(integrand, theta, axis=0)
    rho = np.clip(2 * np.pi * rho, 0, 1)
    return rho, 1 - rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brdf", type=str, required=True)
    parser.add_argument("--max_theta", type=float, default=None, help="单位：弧度，默认用文件里的最大角度")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data = np.loadtxt(args.brdf, delimiter=",")
    theta_i, theta_o = data[:, 0], data[:, 2]
    if np.ptp(theta_i) > np.ptp(theta_o):
        theta = theta_i
    else:
        theta = theta_o

    rho, eps = integrate_slice(theta, data[:, 4:7], args.max_theta)

    print(f"theta range used: [0, {theta[theta >= 0].max():.4f}] rad")
    print(f"rho: {rho}")
    print(f"eps: {eps}")

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("rho_r rho_g rho_b eps_r eps_g eps_b\n")
            f.write(" ".join(f"{v:.6f}" for v in np.concatenate([rho, eps])) + "\n")
        print(f"written to {args.out}")
