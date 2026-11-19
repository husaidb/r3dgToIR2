"""Smoke test for lookup_spec_ab() in gaussian_renderer/neilf.py.

Extracts the loader + lookup source straight from neilf.py (importing the module
needs the CUDA extension simple_knn), so the shipped code is what is tested.

    py -3.12 computebrdf/_test_lookup.py
"""

import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEILF = os.path.join(ROOT, "gaussian_renderer", "neilf.py")

src = open(NEILF, encoding="utf-8").read()
start = src.index("_SPEC_LUT_CACHE = {}")
end = src.index("def rendering_equation")
ns = {"os": os, "np": np, "torch": torch, "__file__": NEILF}
exec(compile(src[start:end], NEILF, "exec"), ns)
lookup = ns["lookup_spec_ab"]
lut = lambda: ns["_get_spec_lut"](torch.device("cpu"))

# --- synthetic table: A[i_ndv, i_r] = 10*i_ndv + i_r ------------------------
A0, B0, r0, v0 = lut()
ns["_SPEC_LUT_CACHE"].clear()
ns["_SPEC_LUT_CACHE"][torch.device("cpu")] = (
    (10.0 * torch.arange(3.0)[:, None] + torch.arange(4.0)[None, :]),
    torch.zeros(3, 4),
    torch.tensor([0.1, 0.4, 0.7, 1.0]),
    torch.tensor([0.0, 0.5, 1.0]),
)
cases = [(0.10, 0.00, 0.0), (0.40, 0.50, 11.0), (1.00, 1.00, 23.0),
         (0.25, 0.25, 5.5), (0.70, 1.00, 22.0)]
fail = 0
for r, v, want in cases:
    got = lookup(torch.tensor([[r]]), torch.tensor([[v]]))[0].item()
    ok = abs(got - want) < 1e-5
    fail += not ok
    print("  r=%.2f ndv=%.2f  A=%7.3f want %7.3f  %s" % (r, v, got, want, "OK" if ok else "FAIL"))

rt = torch.tensor([[0.55]], requires_grad=True)
a = lookup(rt, torch.ones_like(rt))[0]
a.sum().backward()
print("  grad: A(0.55,1)=%.4f dA/dr=%.4f want 3.3333" % (a.item(), rt.grad.item()))
fail += abs(rt.grad.item() - 10.0 / 3.0) > 1e-4
print("  broadcast:", tuple(lookup(torch.rand(4, 3), torch.rand(4, 3))[0].shape))

# --- real LUT --------------------------------------------------------------
ns["_SPEC_LUT_CACHE"].clear()
A, B, rax, vax = lut()
print("\n  real LUT: A[min,max]=%.4f,%.4f  B[min,max]=%.4f,%.4f  nan=%d"
      % (A.min(), A.max(), B.min(), B.max(), int(torch.isnan(A).sum() + torch.isnan(B).sum())))
print("\n  bc     r    rho_spec    eps=1-rho_d-base_color")
for bc, r in [(0.5, 0.5), (0.15, 0.9), (0.09, 0.09), (0.8, 1.0)]:
    aa, bb = lookup(torch.tensor([[r]]), torch.ones(1, 1))
    rho = (0.04 * aa + bb).item()
    print("  %.2f  %.2f    %.4f      %.4f" % (bc, r, rho, 1.0 - bc - rho))

print("\nFAILURES:", fail)
