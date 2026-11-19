"""Regression test for the metallic BRDF / emissivity code in gaussian_renderer/neilf.py.

Extracts the LUT loader + lookup_spec_ab, GGX_specular and rendering_equation's
metallic core (F0 / f_d) + analytic & lut emissivity branches straight from neilf.py
(importing the module needs the CUDA extension simple_knn), so the shipped
expressions are what is tested -- not a transcription of them.

    /d/anaconda3/envs/3dgs/python.exe computebrdf/_test_l2.py   # py -3.12 here has a torch/NumPy ABI clash
"""

import os
import textwrap
import types
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


# --- Extract the shipped metallic / emissivity expressions from rendering_equation.
# rendering_equation is not importable wholesale, but these two blocks are valid
# standalone and reference only names we provide. Anchors, if they vanish, must make
# the test fail loudly rather than silently pass.
def _line_index(lines, needle, start=0):
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith(needle):
            return i
    raise RuntimeError(
        "neilf.py extraction anchor missing: %r (rendering_equation shape changed -- "
        "update _test_l2.py anchors)" % needle)


def _slice(func_start, begins, ends):
    lines = src.splitlines()
    start = _line_index(lines, "def %s" % func_start)
    s = _line_index(lines, begins, start)
    e = _line_index(lines, ends, s)        # exclusive: stop at the end-anchor line
    block = "\n".join(lines[s:e])
    return textwrap.dedent(block) + "\n"


METALLIC_SRC = _slice("rendering_equation", "if metallic is None:", "f_s = GGX_specular(")
EMISSIVITY_SRC = _slice("rendering_equation",
                        'if emissivity_mode == "analytic" and emissivity is None:',
                        'elif emissivity_mode == "auto" and emissivity is None:')
# compile now so a broken extraction fails loudly up front (exec happens in run_shipped)
METALLIC_CODE = compile(METALLIC_SRC, NEILF, "exec")
EMISSIVITY_CODE = compile(EMISSIVITY_SRC, NEILF, "exec")

# --- Extract project_points from script/prepare_object_metalness.py (importing it
# pulls in CUDA modules). Anchor: "def project_points(" up to the next top-level def.
PROJ_SCRIPT = os.path.join(ROOT, "script", "prepare_object_metalness.py")
psrc = open(PROJ_SCRIPT, encoding="utf-8").read()
pslines = psrc.splitlines()
pstart = None
for _i, _l in enumerate(pslines):
    if _l.startswith("def project_points("):
        pstart = _i
        break
if pstart is None:
    raise RuntimeError(
        "prepare_object_metalness.py extraction anchor missing: 'def project_points(' -- "
        "update _test_l2.py anchors")
pend = len(pslines)
for _j in range(pstart + 1, len(pslines)):
    if pslines[_j].startswith("def "):
        pend = _j
        break
PROJECT_POINTS_CODE = compile("\n".join(pslines[pstart:pend]) + "\n", PROJ_SCRIPT, "exec")
pns = {"torch": torch}
exec(PROJECT_POINTS_CODE, pns)
project_points = pns["project_points"]


def run_shipped(base_color, roughness, metallic, emissivity_mode):
    """Run the shipped rendering_equation slices on the given inputs.

    Returns (F0, eps). metallic=None exercises the shipped default-zero branch.
    """
    ns["base_color"] = base_color
    ns["roughness"] = roughness
    ns["metallic"] = metallic
    ns["emissivity"] = None
    ns["emissivity_mode"] = emissivity_mode
    exec(METALLIC_CODE, ns)      # sets metallic (default if None), F0, f_d
    exec(EMISSIVITY_CODE, ns)    # sets rho, eps via analytic or lut branch
    out = (ns["F0"], ns["eps"])
    for k in ("base_color", "roughness", "metallic", "emissivity",
              "emissivity_mode", "F0", "eps", "rho", "f_d"):
        ns.pop(k, None)
    return out


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

# --- Tests 3 & 4: emissivity formulas run through the shipped neilf.py slices
bcs = [0.03, 0.2, 0.5, 0.8]
rs = [0.09, 0.3, 0.6, 0.99]
ms = [0.0, 0.5, 1.0]
print("\n  bc     r    m    eps_lut  eps_an   old_m0  in[0,1]  nan")
for bc in bcs:
    bc_t = torch.full((1, 3), bc)
    for r in rs:
        rt = torch.full((1, 3), r)
        A, B = lookup(rt, torch.ones_like(rt))

        # (b) strictly dielectric rho > strictly metal rho, through the shipped lut branch
        _, eps_die = run_shipped(bc_t, rt, torch.zeros(1, 1), "lut")
        _, eps_met = run_shipped(bc_t, rt, torch.ones(1, 1), "lut")
        rho_die = 1.0 - eps_die
        rho_met = 1.0 - eps_met
        diff = (rho_die - rho_met).min().item()
        ok_b = diff > 1e-6
        fail += not ok_b
        print("    bc=%.2f r=%.2f  rho_die-rho_metal=%.5f (want >0)  %s"
              % (bc, r, diff, "OK" if ok_b else "FAIL"))

        for m in ms:
            mt = torch.full((1, 1), m)
            F0, eps = run_shipped(bc_t, rt, mt, "lut")
            _, eps_an = run_shipped(bc_t, rt, mt, "analytic")

            # (a) m=0 must equal the old dielectric formula (through the shipped lut branch)
            ok_a = True
            if m == 0.0:
                old = 1.0 - (bc_t + 0.04 * A + B).clamp(0, 1)
                ok_a = torch.allclose(eps, old, atol=1e-6)
                fail += not ok_a

            # emission is the quantity shipped to the renderer: must be in [0,1], no NaN
            ok_range = bool(((eps >= 0) & (eps <= 1)).all().item())
            ok_nan = not bool(torch.isnan(eps).any().item())
            fail += (not ok_range) + (not ok_nan)

            # extracted lut branch must be exactly eps == 1 - rho with its own rho
            rho = (bc_t * (1 - mt) + F0 * A + B).clamp(0, 1)
            ok_eps = torch.allclose(eps, 1.0 - rho, atol=1e-6)
            fail += not ok_eps

            # Test 4: analytic == lut formula at A=1, B=0
            rho_lut_ab = (bc_t * (1 - mt) + F0 * 1.0 + 0.0).clamp(0, 1)
            ok4 = torch.allclose(eps_an, 1.0 - rho_lut_ab, atol=1e-6)
            fail += not ok4

            flag = "OK" if (ok_range and ok_nan and ok_eps and ok4
                            and (m != 0.0 or ok_a)) else "FAIL"
            print("  %.2f  %.2f  %.1f   %.5f   %.5f   %s   %s  %s"
                  % (bc, r, m, eps.mean().item(), eps_an.mean().item(),
                     ("-" if m != 0.0 else ("OK" if ok_a else "FAIL")),
                     "OK" if ok_range else "FAIL",
                     "OK" if ok_nan else "FAIL"))
            if flag == "FAIL":
                print("    (eps==1-rho:%s  analytic@A=1:B=0:%s)"
                      % ("OK" if ok_eps else "FAIL", "OK" if ok4 else "FAIL"))

# --- Test 5: project_points perspective divide (regression for the Critical bug) ----
# A camera at the origin looking down +z, identity w2c, fx=fy=100, cx=50, cy=40.
_cam = types.SimpleNamespace()
_cam.world_view_transform = torch.eye(4)
_cam.intrinsics = torch.tensor([[100., 0., 50.],
                                [0., 100., 40.],
                                [0., 0., 1.]])
# (world point, expected u, expected v). The (1, 0.5, 2) case discriminates: a divide
# by 1 (missing perspective divide) yields (200, 130) instead of (100, 65).
for _x, _y, _z, _eu, _ev in [(0., 0., 2., 50., 40.),
                             (1., 0.5, 2., 100., 65.),
                             (0., 0., 0.5, 50., 40.)]:
    _u, _v, _zc = project_points(torch.tensor([[_x, _y, _z]]), _cam)
    _ok = (abs(_u[0].item() - _eu) < 1e-4 and abs(_v[0].item() - _ev) < 1e-4
           and abs(_zc[0].item() - _z) < 1e-6)
    fail += not _ok
    print("Test5 project (%g,%g,%g): u=%.4f v=%.4f z=%.4f  %s"
          % (_x, _y, _z, _u[0].item(), _v[0].item(), _zc[0].item(),
             "OK" if _ok else "FAIL"))

print("\nFAILURES:", fail)
if fail:
    raise SystemExit(1)
