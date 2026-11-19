"""Regression test for the metallic BRDF / emissivity code in gaussian_renderer/neilf.py.

Extracts the LUT loader + lookup_spec_ab and GGX_specular straight from neilf.py
(importing the module needs the CUDA extension simple_knn), so the shipped code is
what is tested. rendering_equation itself is not extractable in isolation, so its
metallic formulas are re-evaluated here with the same real LUT.

    py -3.12 computebrdf/_test_l2.py        # (falls back to conda env 3dgs if torch ABI broken)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEILF = os.path.join(ROOT, "gaussian_renderer", "neilf.py")

src = open(NEILF, encoding="utf-8").read()
start = src.index("_SPEC_LUT_CACHE = {}")
end = src.index("def rendering_equation")
ns = {"os": os, "np": np, "torch": torch, "F": F, "__file__": NEILF}
exec(compile(src[start:end], NEILF, "exec"), ns)
lookup = ns["lookup_spec_ab"]

gsrc = src[src.index("def GGX_specular"):]
exec(compile(gsrc, NEILF, "exec"), ns)
GGX = ns["GGX_specular"]

fail = 0

# --- Test 1: scalar vs tensor fresnel equivalent -----------------------------
N, L = 64, 8
normal = torch.randn(N, 3)
viewdirs = torch.randn(N, 3)
incident_dirs = torch.randn(N, L, 3)
roughness = torch.rand(N, 3) * 0.9 + 0.09

out_s = GGX(normal, viewdirs, incident_dirs, roughness, fresnel=0.04)
out_t = GGX(normal, viewdirs, incident_dirs, roughness, fresnel=torch.full((N, 3), 0.04))
ok = torch.allclose(out_s, out_t, atol=1e-6)
fail += not ok
print("Test1 scalar/tensor fresnel: shape=%s maxdiff=%.2e  %s"
      % (tuple(out_t.shape), (out_s - out_t).abs().max().item(), "OK" if ok else "FAIL"))

# --- Test 2: per-row F0 tensor vs row-by-row scalar calls --------------------
F0 = torch.rand(N, 3)
full = GGX(normal, viewdirs, incident_dirs, roughness, fresnel=F0)
for i in range(4):
    row = GGX(normal[i:i + 1], viewdirs[i:i + 1], incident_dirs[i:i + 1],
              roughness[i:i + 1], fresnel=F0[i:i + 1])
    ok = torch.allclose(full[i:i + 1], row, atol=1e-6)
    fail += not ok
    print("Test2 row %d: maxdiff=%.2e  %s"
          % (i, (full[i:i + 1] - row).abs().max().item(), "OK" if ok else "FAIL"))

# --- Tests 3 & 4: emissivity formulas on the real LUT ------------------------
bcs = [0.03, 0.2, 0.5, 0.8]
rs = [0.09, 0.3, 0.6, 0.99]
ms = [0.0, 0.5, 1.0]
print("\n  bc     r    m    rho_lut    rho_an   old_m0  in[0,1]  nan")
for bc in bcs:
    bc_t = torch.full((1, 3), bc)
    for r in rs:
        rt = torch.full((1, 3), r)
        A, B = lookup(rt, torch.ones_like(rt))

        # (b) strictly dielectric rho > strictly metal rho, same bc/A/B
        rho_die = (bc_t * (1 - 0.0) + (0.04 * (1 - 0.0) + bc_t * 0.0) * A + B).clamp(0, 1)
        rho_met = (bc_t * (1 - 1.0) + (0.04 * (1 - 1.0) + bc_t * 1.0) * A + B).clamp(0, 1)
        diff = (rho_die - rho_met).min().item()
        ok_b = diff > 1e-6
        fail += not ok_b
        print("    bc=%.2f r=%.2f  rho_die-rho_metal=%.5f (want >0)  %s"
              % (bc, r, diff, "OK" if ok_b else "FAIL"))

        for m in ms:
            mt = torch.full((1, 1), m)
            F0 = 0.04 * (1 - mt) + bc_t * mt                               # [1,3]
            rho = (bc_t * (1 - mt) + F0 * A + B).clamp(0, 1)               # [1,3]
            rho_an = (bc_t + 0.04 * (1 - mt)).clamp(0, 1)                  # analytic

            # (a) m=0 must equal the old dielectric formula bc + 0.04*A + B
            ok_a = True
            if m == 0.0:
                old = (bc_t + 0.04 * A + B).clamp(0, 1)
                ok_a = torch.allclose(rho, old, atol=1e-6)
                fail += not ok_a

            ok_range = bool(((rho >= 0) & (rho <= 1)).all().item())
            ok_nan = not bool(torch.isnan(rho).any().item())
            fail += (not ok_range) + (not ok_nan)

            # Test 4: analytic == lut formula at A=1, B=0
            rho_lut_ab = (bc_t * (1 - mt) + F0 * 1.0 + 0.0).clamp(0, 1)
            ok4 = torch.allclose(rho_an, rho_lut_ab, atol=1e-6)
            fail += not ok4

            print("  %.2f  %.2f  %.1f   %.5f   %.5f   %s   %s  %s"
                  % (bc, r, m, rho.mean().item(), rho_an.mean().item(),
                     ("-" if m != 0.0 else ("OK" if ok_a else "FAIL")),
                     "OK" if ok_range else "FAIL",
                     "OK" if ok_nan else "FAIL"))

print("\nFAILURES:", fail)
