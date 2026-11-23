"""端到端流水线纯逻辑的 CPU 测试：python script/_test_pipeline.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "script"))
import run_pipeline as rp  # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"{name}: OK")
    else:
        print(f"{name}: FAIL {detail}")
        failures.append(name)


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("")


def make_blender_root(root):
    root = Path(root)
    touch(root / "train" / "a.png")
    touch(root / "points3d.ply")
    with open(root / "transforms_train.json", "w") as f:
        json.dump({"camera_angle_x": 0.8, "frames": []}, f)
    return root


def make_blender_raw(root):
    return make_blender_root(Path(root) / "raw")


def make_colmap(root, sub="sparse/0", ext="txt", with_images=True):
    root = Path(root)
    sparse = root / "raw" / sub
    for name in ("cameras", "images", "points3D"):
        touch(str(sparse / f"{name}.{ext}"))
    if with_images:
        touch(root / "raw" / "images" / "a.png")
    return root


def make_data(train_stems=("a", "b")):
    d = Path(tempfile.mkdtemp()) / "data"
    for stem in train_stems:
        touch(d / "train" / f"{stem}.png")
    return d


def expect_error(fn, needle=""):
    try:
        fn()
    except rp.PipelineError as e:
        return needle in str(e) if needle else True
    return False


def has_pair(cmd, flag, value):
    return any(cmd[i] == flag and cmd[i + 1] == value for i in range(len(cmd) - 1))


def test_detect_input():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        check("detect blender_root",
              rp.detect_input(make_blender_root(Path(tmp) / "a")) == "blender_root")
        make_blender_raw(Path(tmp) / "b")
        check("detect blender_raw",
              rp.detect_input(Path(tmp) / "b") == "blender_raw")
        check("detect colmap",
              rp.detect_input(make_colmap(Path(tmp) / "c")) == "colmap")
        empty = Path(tmp) / "d"
        os.makedirs(empty)
        check("detect empty raises", expect_error(lambda: rp.detect_input(empty)))


def test_resolve_paths():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds, work = Path(tmp) / "ds", Path(tmp) / "work"
        p = rp.resolve_paths(ds, work, "kettle")
        check("scene_dir", p.scene_dir == ds / "kettle")
        check("work_dir", p.work_dir == work / "kettle")
        check("data_dir", p.data_dir == work / "kettle" / "data")
        check("stage1_dir", p.stage1_dir == work / "kettle" / "stage1")
        check("prep_dir", p.prep_dir == work / "kettle" / "prep")
        check("stage2_dir", p.stage2_dir == work / "kettle" / "stage2")
        check("log_dir", p.log_dir == work / "kettle" / "logs")
        check("meta_path", p.meta_path == work / "kettle" / "scene_meta.json")

        abs_scene = Path(tmp) / "abc"
        q = rp.resolve_paths(ds, work, str(abs_scene))
        check("absolute scene_dir", q.scene_dir == abs_scene)
        check("absolute work_dir", q.work_dir == work / "abc")

        for name in ("raw", "data", "sparse", "train", "masks", "masks_ir", "material_prior"):
            check(f"reserved {name}",
                  expect_error(lambda n=name: rp.resolve_paths(ds, work, str(Path(tmp) / n))))


def test_resolve_sparse_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        raw = make_colmap(Path(tmp) / "a", sub="sparse/0") / "raw"
        check("sparse/0", rp.resolve_sparse_dir(raw) == raw / "sparse" / "0")

        raw2 = make_colmap(Path(tmp) / "b", sub="sparse") / "raw"
        check("sparse flat", rp.resolve_sparse_dir(raw2) == raw2 / "sparse")

        raw3 = Path(tmp) / "c" / "raw"
        os.makedirs(raw3)
        check("sparse missing raises", expect_error(lambda: rp.resolve_sparse_dir(raw3)))


def test_validate_input():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds, work = Path(tmp) / "ds", Path(tmp) / "work"

        good = make_blender_root(ds / "good")
        check("validate blender ok",
              rp.validate_input(rp.resolve_paths(ds, work, "good"), "blender_root") is None)

        bad = make_blender_root(ds / "bad")
        os.remove(bad / "points3d.ply")
        check("validate missing ply",
              expect_error(lambda: rp.validate_input(rp.resolve_paths(ds, work, "bad"), "blender_root"),
                           "points3d.ply"))

        empty = make_blender_root(ds / "empty")
        os.remove(empty / "train" / "a.png")
        check("validate empty train",
              expect_error(lambda: rp.validate_input(rp.resolve_paths(ds, work, "empty"), "blender_root")))

        make_colmap(ds / "cm", with_images=False)
        check("validate colmap no images",
              expect_error(lambda: rp.validate_input(rp.resolve_paths(ds, work, "cm"), "colmap")))

        make_colmap(ds / "cm2")
        check("validate colmap ok",
              rp.validate_input(rp.resolve_paths(ds, work, "cm2"), "colmap") is None)

        make_blender_raw(ds / "rawfmt")
        check("validate blender_raw ok",
              rp.validate_input(rp.resolve_paths(ds, work, "rawfmt"), "blender_raw") is None)


def test_masks():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        data = make_data()
        masks = Path(tmp) / "masks"
        touch(masks / "a.png")
        check("masks incomplete", not rp.masks_complete(data, masks))
        touch(masks / "b.png")
        check("masks complete", rp.masks_complete(data, masks))
        touch(masks / "c.png")
        check("masks extra", not rp.masks_complete(data, masks))
        check("masks None", not rp.masks_complete(data, None))

        scene = Path(tmp) / "scene"
        work = Path(tmp) / "work"
        os.makedirs(work / "masks_ir")
        check("masks work fallback", rp.resolve_masks_dir(scene, work) == work / "masks_ir")
        os.makedirs(scene / "masks_ir")
        check("masks dataset first", rp.resolve_masks_dir(scene, work) == scene / "masks_ir")
        check("masks none", rp.resolve_masks_dir(Path(tmp) / "x", Path(tmp) / "y") is None)


def test_builders():
    d, o = Path("data"), Path("out")
    ckpt, prob = Path("ck.pth"), d / "material_prior" / "metal_prob.npy"
    masks, meta = Path("masks_ir"), Path("scene_meta.json")

    cmd = rp.build_convert_cmd(Path("sparse"), Path("images"), d)
    check("convert script", "convert_colmap_to_blender.py" in cmd[1])
    check("convert args", has_pair(cmd, "--sparse", "sparse")
          and has_pair(cmd, "--images", "images") and has_pair(cmd, "--out", "data"))

    cmd = rp.build_stage1_cmd(d, o)
    check("stage1 core", has_pair(cmd, "-t", "render") and has_pair(cmd, "-s", "data")
          and has_pair(cmd, "-m", "out"))
    check("stage1 zeros", all(has_pair(cmd, f, v) for f, v in rp.STAGE1_ZERO))

    cmd = rp.build_stage2_cmd(d, o, ckpt, prob, 30)
    check("stage2 core", has_pair(cmd, "-t", "neilf") and has_pair(cmd, "-s", "data")
          and has_pair(cmd, "-m", "out"))
    check("stage2 ckpt/prob", has_pair(cmd, "--checkpoint", "ck.pth")
          and has_pair(cmd, "--material_prob_path", str(prob)))
    check("stage2 lut/temp", has_pair(cmd, "--emissivity_mode", "lut")
          and has_pair(cmd, "--temperature", "30"))
    check("stage2 freeze", all(has_pair(cmd, f, v) for f, v in rp.STAGE2_FREEZE))
    check("stage2 zeros", all(has_pair(cmd, f, v) for f, v in rp.STAGE2_ZERO))
    check("stage2 no metallic", "--metallic" not in cmd)

    cmd = rp.build_prepare_cmd(d, Path("prep"), ckpt, masks, d / "material_prior", 3,
                               scene_meta=str(meta))
    check("prepare core", has_pair(cmd, "-s", "data") and has_pair(cmd, "-m", "prep")
          and has_pair(cmd, "--checkpoint", "ck.pth")
          and has_pair(cmd, "--masks_dir", "masks_ir")
          and has_pair(cmd, "--out_dir", str(d / "material_prior"))
          and has_pair(cmd, "--erode", "3"))
    check("prepare meta", has_pair(cmd, "--scene_meta", str(meta)))

    cmd = rp.build_prepare_cmd(d, Path("prep"), ckpt, masks, d / "material_prior", 3,
                               metallic=0.8)
    check("prepare metallic", has_pair(cmd, "--metallic", "0.8") and "--scene_meta" not in cmd)

    cmd = rp.build_generate_masks_cmd(Path("w"), d)
    check("gen masks", has_pair(cmd, "-s", "w")
          and has_pair(cmd, "--images_dir", str(d / "train"))
          and has_pair(cmd, "--masks_dir", "w" + os.sep + "masks_ir"))

    cmd = rp.build_vlm_cmd(Path("s"), Path("m.json"), "qwen3-vl-flash")
    check("vlm", has_pair(cmd, "-s", "s") and has_pair(cmd, "--out", "m.json")
          and has_pair(cmd, "--model", "qwen3-vl-flash")
          and has_pair(cmd, "--missing", "error"))

    all_cmds = [rp.build_stage1_cmd(d, o), rp.build_stage2_cmd(d, o, ckpt, prob, 30),
                rp.build_prepare_cmd(d, Path("prep"), ckpt, masks, d, 3, metallic=0.5),
                rp.build_generate_masks_cmd(Path("w"), d),
                rp.build_vlm_cmd(Path("s"), Path("m.json"), "m"),
                rp.build_convert_cmd(Path("sp"), Path("im"), d)]
    check("cmds list[str]", all(isinstance(c, list) and all(isinstance(x, str) for x in c)
                               and None not in c for c in all_cmds))


def run_cli(tmp, extra):
    ds, work = Path(tmp) / "ds", Path(tmp) / "work"
    make_blender_root(ds / "kettle")
    cmd = [sys.executable, "script/run_pipeline.py",
           "--dataset_root", str(ds), "--work_root", str(work), "-d", "kettle"] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r, work


def test_cli_dry_run():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r, work = run_cli(tmp, ["--temperature", "30", "--auto_mask",
                                "--metallic", "0.5", "--dry_run"])
        out = r.stdout + r.stderr
        check("dry rc", r.returncode == 0, out[-400:])
        check("dry train", "train.py" in out)
        check("dry lut", "--emissivity_mode" in out and "lut" in out)
        check("dry masks", "generate_masks.py" in out)
        check("dry prepare", "prepare_object_metalness.py" in out)
        check("dry no work", not work.exists(), str(work))


def test_missing_temperature():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r, _ = run_cli(tmp, ["--auto_mask", "--metallic", "0.5", "--dry_run"])
        out = r.stdout + r.stderr
        check("temp rc", r.returncode != 0, out[-400:])
        check("temp message", "temperature" in out or "温度" in out, out[-400:])


def test_missing_masks():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r, _ = run_cli(tmp, ["--temperature", "30", "--metallic", "0.5", "--dry_run"])
        out = r.stdout + r.stderr
        check("masks rc", r.returncode != 0, out[-400:])
        check("masks message", "masks_ir" in out and "--auto_mask" in out, out[-400:])


def main():
    test_detect_input()
    test_resolve_paths()
    test_resolve_sparse_dir()
    test_validate_input()
    test_masks()
    test_builders()
    test_cli_dry_run()
    test_missing_temperature()
    test_missing_masks()
    print(f"\nFAILURES: {len(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
