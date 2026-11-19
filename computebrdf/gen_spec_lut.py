"""Offline Karis A/B LUT for the simplified Disney BRDF used in gaussian_renderer/neilf.py.

    rho_spec(r, N.V, F0) = F0 * A(r, N.V) + B(r, N.V)

White-furnace integral: incident radiance is constant 1, so no environment map is
involved. A/B are obtained by integrating  int f_s(l,v) (n.l) dl  over the hemisphere
with GGX half-vector sampling.

Must stay consistent with GGX_specular() in gaussian_renderer/neilf.py:
    D  = alpha2 / (pi * (NoH^2 (alpha2 - 1) + 1)^2),   alpha = r^2, alpha2 = alpha^2
    G  = G1(NoV) * G1(NoL),  G1(x) = x / (x (1-k) + k),  k = (alpha2 + 2r + 1) / 8
    F  = F0 + (1 - F0) * 2 ** ((-5.55473 * V.H - 6.98316) * V.H)
(the code writes 1/nom1 and 1/nom2, i.e. the reciprocal form of G1)

The denominator clamp in GGX_specular is not reproduced: it is a division guard
(1e-12), far below the true minimum of the denominator (~5e-8 at r=0.09), so it
does not alter the BRDF for any reachable roughness.

Output spec_lut.npz:
    A, B       [res_ndv, res_roughness]  (first index = N.V, second = roughness)
    roughness  [res_roughness]
    ndv        [res_ndv]

usage:
    python computebrdf/gen_spec_lut.py
    python computebrdf/gen_spec_lut.py --res 64 --out computebrdf/spec_lut.npz
"""

import argparse
import os

import numpy as np


def stratified_half_vectors(roughness, nu, nphi):
    """Deterministic GGX half vectors: uniform in (CDF, azimuth).

    E[f] over these points is the same estimator as random half-vector importance
    sampling, but low-discrepancy, so it is stable at grazing N.V where the random
    estimator has enormous variance.
    """
    a2 = (roughness * roughness) ** 2
    U, PH = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nphi) + 0.5) * 2.0 * np.pi / nphi,
                        indexing="ij")
    cos_h = np.sqrt((1.0 - U) / (1.0 + (a2 - 1.0) * U))
    sin_h = np.sqrt(np.clip(1.0 - cos_h ** 2, 0.0, 1.0))
    return np.stack([sin_h * np.cos(PH), sin_h * np.sin(PH), cos_h], -1).reshape(-1, 3)


def integrate_rho(viewdirs, halfs, k):
    """int f_s (n.l) dl for F0=0 and F0=1.

    viewdirs: [M,3]   halfs: [S,3]   returns rho0[M], rho1[M]
    estimator f_s * NoL / pdf_l = F * G * VoH / (NoV * NoH)
    """
    S = halfs.shape[0]
    ndv = np.maximum(viewdirs[:, 2], 1e-6)
    VoH = halfs @ viewdirs.T                                  # [S,M]
    L = 2.0 * VoH[..., None] * halfs[:, None, :] - viewdirs[None, :, :]
    NoL = L[..., 2]
    NoH = np.maximum(halfs[:, 2], 1e-6)[:, None]
    w = np.power(2.0, (-5.55473 * VoH - 6.98316) * VoH)       # (1-V.H)^5 approx
    valid = (VoH > 0.0) & (NoL > 0.0)
    NoLc = np.maximum(NoL, 1e-6)
    gl = NoLc / (NoLc * (1.0 - k) + k)                        # G1(NoL)
    est = np.where(valid, gl * VoH / (NoH * (ndv[None, :] * (1.0 - k) + k)), 0.0)
    return (est * w).sum(0) / S, est.sum(0) / S


def quadrature_rho(r, f0, ndv, nt=3000, np_=1500):
    """Independent check: integrate over the half-vector hemisphere on a fixed grid.

    Reliable for r >= ~0.2; below that the D lobe is narrower than the grid.
    """
    V = np.array([np.sqrt(max(0.0, 1.0 - ndv * ndv)), 0.0, ndv])
    a2 = (r * r) ** 2
    k = (r * r + 2.0 * r + 1.0) / 8.0
    th = (np.arange(nt) + 0.5) * np.pi / 2 / nt
    ph = (np.arange(np_) + 0.5) * 2.0 * np.pi / np_
    T, P = np.meshgrid(th, ph, indexing="ij")
    H = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    VoH = np.clip(H @ V, 0.0, 1.0)
    L = 2.0 * VoH[..., None] * H - V
    NoL = L[..., 2]
    NoH = np.cos(T)
    D = a2 / (np.pi * (NoH ** 2 * (a2 - 1.0) + 1.0) ** 2)
    F = f0 + (1.0 - f0) * np.power(2.0, (-5.55473 * VoH - 6.98316) * VoH)
    NoLc = np.clip(NoL, 1e-9, 1.0)
    g1l = NoLc / (NoLc * (1.0 - k) + k)
    nom1 = ndv * (1.0 - k) + k                             # G1(NoV) = NoV / nom1
    val = np.where((NoL > 0.0) & (VoH > 0.0), D * F * g1l * VoH / nom1 * np.sin(T), 0.0)
    return val.sum() * (np.pi / 2 / nt) * (2.0 * np.pi / np_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--nu", type=int, default=256, help="stratified samples along GGX CDF")
    ap.add_argument("--nphi", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16, help="N.V values per batch")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_lut.npz"))
    args = ap.parse_args()

    u = np.linspace(0.0, 1.0, args.res)
    roughness = 0.02 + 0.98 * u ** 2
    ndv = 0.02 + 0.98 * u
    theta = np.arccos(ndv)
    viewdirs = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)

    A = np.zeros((args.res, args.res))
    B = np.zeros((args.res, args.res))
    for j, r in enumerate(roughness):
        alpha = r * r
        k = (alpha + 2.0 * r + 1.0) / 8.0
        halfs = stratified_half_vectors(r, args.nu, args.nphi)
        for i0 in range(0, args.res, args.chunk):
            vb = viewdirs[i0:i0 + args.chunk]
            rho0, rho1 = integrate_rho(vb, halfs, k)
            B[i0:i0 + args.chunk, j] = rho0
            A[i0:i0 + args.chunk, j] = rho1 - rho0

    np.savez(args.out, A=A, B=B, roughness=roughness, ndv=ndv)
    print("saved:", args.out)
    print("A range [%.4f, %.4f]   B range [%.4f, %.4f]" % (A.min(), A.max(), B.min(), B.max()))

    def lookup(rv, vv):
        iv = np.interp(rv, roughness, np.arange(args.res))
        iu = np.interp(vv, ndv, np.arange(args.res))
        i, j = min(int(iu), args.res - 2), min(int(iv), args.res - 2)
        fu, fv = iu - i, iv - j

        def bil(M):
            return (M[i, j] * (1 - fu) * (1 - fv) + M[i + 1, j] * fu * (1 - fv)
                    + M[i, j + 1] * (1 - fu) * fv + M[i + 1, j + 1] * fu * fv)
        return bil(A), bil(B)

    print("\nvalidate vs independent quadrature (r >= 0.2) and fine stratification:")
    print(" r     N.V   A_lut    A_ref    B_lut    B_ref    A+B_lut  A+B_ref")
    err = 0.0
    for r, v in [(0.2, 1.0), (0.5, 0.866), (0.5, 0.5), (0.8, 0.26), (1.0, 0.1), (0.3, 1.0)]:
        al, bl = lookup(r, v)
        if r >= 0.2:
            a_r = quadrature_rho(r, 1.0, v) - quadrature_rho(r, 0.0, v)
            b_r = quadrature_rho(r, 0.0, v)
        else:
            alpha = r * r
            k = (alpha + 2.0 * r + 1.0) / 8.0
            vb = np.array([[np.sqrt(max(0.0, 1 - v * v)), 0.0, v]])
            r0, r1 = integrate_rho(vb, stratified_half_vectors(r, 1024, 1024), k)
            a_r, b_r = r1[0] - r0[0], r0[0]
        err = max(err, abs(al - a_r), abs(bl - b_r))
        print(" %.2f  %.3f  %.4f  %.4f  %.4f  %.4f   %.4f   %.4f"
              % (r, v, al, a_r, bl, b_r, al + bl, a_r + b_r))
    print("max |LUT - reference| = %.4f" % err)

    print("\nrho_spec = F0*A + B and eps = 1 - rho")
    print(" r    N.V    dielectric F0=0.04: rho   eps  |  metal F0=0.65: rho   eps")
    for r, v in [(0.09, 1.0), (0.3, 1.0), (0.5, 1.0), (0.8, 1.0), (1.0, 1.0), (0.5, 0.7)]:
        a, b = lookup(r, v)
        rd, rm = 0.04 * a + b, 0.65 * a + b
        print(" %.2f  %.3f      %.4f  %.3f      %.4f  %.3f" % (r, v, rd, 1 - rd, rm, 1 - rm))


if __name__ == "__main__":
    main()
