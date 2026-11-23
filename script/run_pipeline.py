"""一组自采数据的端到端流水线：转换 → 阶段一 → mask → metalness → 阶段二。

用法：
    python script/run_pipeline.py -d <scene> --temperature 30 [--auto_mask] [--dry_run]

输入只读，产物全部落在 --work_root/<scene>/ 下。
"""
import argparse
import collections
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
RESERVED_WORK_NAMES = {"raw", "data", "sparse", "train", "masks", "masks_ir", "material_prior"}

STAGE1_ZERO = (("--lambda_depth", "0"), ("--lambda_depth_smooth", "0"),
               ("--lambda_depth_var", "0"), ("--lambda_mask_entropy", "0"),
               ("--lambda_opacity", "0"), ("--lambda_surface", "0"),
               ("--lambda_scaling", "0"), ("--lambda_normal_render_depth", "0"),
               ("--lambda_normal_mvs_depth", "0"), ("--lambda_normal_smooth", "0"),
               ("--lambda_point_entropy", "0"), ("--lambda_orientation", "0"))
STAGE2_FREEZE = (("--densify_until_iter", "0"), ("--position_lr_init", "0"),
                 ("--position_lr_final", "0"), ("--normal_lr", "0"),
                 ("--rotation_lr", "0"), ("--scaling_lr", "0"),
                 ("--opacity_lr", "0"), ("--sh_lr", "0"))
STAGE2_ZERO = (("--lambda_depth", "0"), ("--lambda_depth_smooth", "0"),
               ("--lambda_mask_entropy", "0"), ("--lambda_opacity", "0"),
               ("--lambda_surface", "0"), ("--lambda_normal_render_depth", "0"),
               ("--lambda_normal_smooth", "0"), ("--lambda_point_entropy", "0"),
               ("--lambda_orientation", "0"), ("--lambda_depth_var", "0"),
               ("--lambda_scaling", "0"), ("--lambda_light", "0"),
               ("--lambda_base_color", "0"), ("--lambda_base_color_smooth", "0"),
               ("--lambda_roughness_smooth", "0"), ("--lambda_light_smooth", "0"),
               ("--lambda_visibility", "0"), ("--lambda_visibility_smooth", "0"),
               ("--lambda_env_smooth", "0"))


class PipelineError(RuntimeError):
    pass


@dataclass
class Paths:
    scene_dir: Path
    work_dir: Path
    data_dir: Path
    stage1_dir: Path
    prep_dir: Path
    stage2_dir: Path
    log_dir: Path
    meta_path: Path


def detect_input(scene_dir: Path) -> str:
    if (scene_dir / "transforms_train.json").is_file() and (scene_dir / "train").is_dir():
        return "blender_root"
    if ((scene_dir / "raw" / "transforms_train.json").is_file()
            and (scene_dir / "raw" / "train").is_dir()):
        return "blender_raw"
    if (scene_dir / "raw" / "sparse").is_dir():
        return "colmap"
    raise PipelineError(
        f"{scene_dir} 不是支持的输入布局：需要 根/transforms_train.json+train/、"
        "raw/transforms_train.json+train/，或 raw/sparse/")


def resolve_paths(dataset_root: Path, work_root: Path, dataset: str) -> Paths:
    p = Path(dataset)
    if p.is_absolute():
        scene_dir, work_name = p, p.name
    else:
        scene_dir, work_name = dataset_root / dataset, dataset
    if work_name in RESERVED_WORK_NAMES:
        raise PipelineError(
            f"场景名 {work_name!r} 是保留名，不能用作 work 目录（{sorted(RESERVED_WORK_NAMES)}）")
    work_dir = work_root / work_name
    return Paths(
        scene_dir=scene_dir,
        work_dir=work_dir,
        data_dir=work_dir / "data",
        stage1_dir=work_dir / "stage1",
        prep_dir=work_dir / "prep",
        stage2_dir=work_dir / "stage2",
        log_dir=work_dir / "logs",
        meta_path=work_dir / "scene_meta.json",
    )


def resolve_sparse_dir(raw_dir: Path) -> Path:
    for cand in (raw_dir / "sparse" / "0", raw_dir / "sparse"):
        if any((cand / f).exists() for f in ("cameras.bin", "cameras.txt")):
            return cand
    raise PipelineError(f"{raw_dir} 下找不到 COLMAP 相机文件（sparse/0 或 sparse）")


def _images(directory: Path):
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def validate_input(paths: Paths, kind: str) -> None:
    if kind in ("blender_root", "blender_raw"):
        src = paths.scene_dir if kind == "blender_root" else paths.scene_dir / "raw"
        if not (src / "transforms_train.json").is_file():
            raise PipelineError(f"{src} 缺少 transforms_train.json")
        if not _images(src / "train"):
            raise PipelineError(f"{src / 'train'} 中没有图片")
        if not (src / "points3d.ply").is_file():
            raise PipelineError(f"{src} 缺少 points3d.ply（不允许随机初始化点云）")
    else:
        raw = paths.scene_dir / "raw"
        if not _images(raw / "images"):
            raise PipelineError(f"{raw / 'images'} 中没有图片")
        resolve_sparse_dir(raw)


def expected_stems(data_dir: Path) -> set:
    return {p.stem for p in _images(data_dir / "train")}


def resolve_masks_dir(scene_dir: Path, work_dir: Path):
    for cand in (scene_dir / "masks_ir", work_dir / "masks_ir"):
        if cand.is_dir():
            return cand
    return None


def masks_complete(data_dir: Path, masks_dir) -> bool:
    if masks_dir is None or not Path(masks_dir).is_dir():
        return False
    return {p.stem for p in _images(Path(masks_dir))} == expected_stems(data_dir)


def prepare_is_complete(prob: Path, ckpt: Path) -> bool:
    """prob 存在且不早于阶段一 checkpoint 才算完成；旧 prob 需重新生成。"""
    prob, ckpt = Path(prob), Path(ckpt)
    if not prob.is_file():
        return False
    if not ckpt.is_file():
        return True
    return prob.stat().st_mtime >= ckpt.stat().st_mtime


def build_convert_cmd(sparse_dir, images_dir, out_dir) -> list:
    return [sys.executable, "script/convert_colmap_to_blender.py",
            "--sparse", str(sparse_dir), "--images", str(images_dir), "--out", str(out_dir)]


def build_generate_masks_cmd(work_dir, data_dir) -> list:
    return [sys.executable, "generate_masks.py", "-s", str(work_dir),
            "--images_dir", str(Path(data_dir) / "train"),
            "--masks_dir", str(Path(work_dir) / "masks_ir")]


def build_vlm_cmd(scene_dir, out_path, model) -> list:
    return [sys.executable, "script/query_vlm_metalness.py", "-s", str(scene_dir),
            "--out", str(out_path), "--model", model, "--missing", "error"]


def build_prepare_cmd(data_dir, prep_dir, ckpt, masks_dir, out_dir, erode,
                      metallic=None, scene_meta=None) -> list:
    cmd = [sys.executable, "script/prepare_object_metalness.py",
           "-s", str(data_dir), "-m", str(prep_dir),
           "--checkpoint", str(ckpt), "--masks_dir", str(masks_dir),
           "--out_dir", str(out_dir), "--erode", str(erode)]
    if metallic is not None:
        cmd += ["--metallic", str(metallic)]
    elif scene_meta is not None:
        cmd += ["--scene_meta", str(scene_meta)]
    return cmd


def build_stage1_cmd(data_dir, out_dir) -> list:
    cmd = [sys.executable, "train.py", "-s", str(data_dir), "-m", str(out_dir), "-t", "render"]
    for flag, value in STAGE1_ZERO:
        cmd += [flag, value]
    return cmd


def build_stage2_cmd(data_dir, out_dir, ckpt, prob_path, temperature) -> list:
    cmd = [sys.executable, "train.py", "-s", str(data_dir), "-m", str(out_dir), "-t", "neilf",
           "--checkpoint", str(ckpt), "--material_prob_path", str(prob_path),
           "--emissivity_mode", "lut", "--temperature", str(temperature)]
    for flag, value in STAGE2_FREEZE + STAGE2_ZERO:
        cmd += [flag, value]
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MWIR 单场景端到端流水线")
    parser.add_argument("-d", "--dataset", required=True,
                        help="场景名（在 dataset_root 下）或绝对路径")
    parser.add_argument("--dataset_root", default=str(ROOT / "datasets"))
    parser.add_argument("--work_root", default=str(ROOT / "work"))
    parser.add_argument("--temperature", type=float, default=None,
                        help="物体温度（°C），阶段二必需")
    parser.add_argument("--auto_mask", action="store_true",
                        help="没有 masks_ir 时用 rembg 自动生成")
    parser.add_argument("--metallic", type=float, default=None,
                        help="物体级 metalness，优先于 scene_meta/VLM")
    parser.add_argument("--vlm_model", default="qwen3-vl-flash")
    parser.add_argument("--erode", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="忽略已有产物重跑")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--strict_vlm", action="store_true",
                        help="VLM 置信度 <0.5 时报错而不是仅警告")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    return parser


def run_step(name: str, cmd, log_dir: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY_RUN] {name}: {shlex.join(cmd)}")
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[STEP] {name}")
    tail = collections.deque(maxlen=40)
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=child_env)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            tail.append(line)
        rc = proc.wait()
    if rc != 0:
        print("".join(tail))
        raise PipelineError(f"{name} 失败（exit {rc}），日志: {log_path}")


def copy_training_files(src: Path, data_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY_RUN] copy: {src} -> {data_dir}")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "transforms_train.json", data_dir / "transforms_train.json")
    shutil.copy2(src / "points3d.ply", data_dir / "points3d.ply")
    shutil.copytree(src / "train", data_dir / "train", dirs_exist_ok=True)


def read_meta(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decide_metalness(args, paths: Paths, dry_run: bool) -> dict:
    """返回 build_prepare_cmd 的 metallic / scene_meta 二选一参数。"""
    if args.metallic is not None:
        print(f"[INFO] 物体 metalness 来自 --metallic: {args.metallic}")
        return {"metallic": args.metallic}
    for cand in (paths.meta_path, paths.scene_dir / "scene_meta.json"):
        if cand.is_file():
            print(f"[INFO] 物体 metalness 来自 {cand}")
            return {"scene_meta": str(cand)}
    visible = paths.scene_dir / "visible"
    if visible.is_dir() and any(visible.iterdir()):
        run_step("vlm", build_vlm_cmd(paths.scene_dir, paths.meta_path, args.vlm_model),
                 paths.log_dir, dry_run)
        return {"scene_meta": str(paths.meta_path)}
    raise PipelineError(
        "无法确定物体 metalness：给 --metallic、提供 scene_meta.json，"
        "或在 <scene>/visible/ 放可见光照片让 VLM 判断")


def check_confidence(args, metal_kwargs: dict, dry_run: bool) -> None:
    meta_path = metal_kwargs.get("scene_meta")
    if not meta_path or not Path(meta_path).is_file():
        return
    meta = read_meta(Path(meta_path))
    conf = float(meta.get("confidence", 1.0))
    if conf >= 0.5:
        return
    msg = f"[WARN] scene_meta 置信度低 {conf:.2f}（metallic={meta.get('metallic')}）：{meta_path}"
    if args.strict_vlm:
        raise PipelineError(msg)
    print(msg)


def _run(args) -> int:
    paths = resolve_paths(Path(args.dataset_root), Path(args.work_root), args.dataset)
    kind = detect_input(paths.scene_dir)
    validate_input(paths, kind)
    print(f"[INFO] 输入形态: {kind}，场景: {paths.scene_dir}")

    if not args.skip_stage2 and args.temperature is None:
        raise PipelineError("阶段二需要 --temperature（物体温度 °C）；只跑阶段一请加 --skip_stage2")

    ckpt1 = paths.stage1_dir / "chkpnt30000.pth"
    prob = paths.data_dir / "material_prior" / "metal_prob.npy"

    if (paths.data_dir / "transforms_train.json").is_file() and not args.force:
        print(f"[SKIP] data 已存在: {paths.data_dir}")
    elif kind == "colmap":
        raw = paths.scene_dir / "raw"
        run_step("convert", build_convert_cmd(resolve_sparse_dir(raw), raw / "images",
                                              paths.data_dir), paths.log_dir, args.dry_run)
    else:
        src = paths.scene_dir if kind == "blender_root" else paths.scene_dir / "raw"
        copy_training_files(src, paths.data_dir, args.dry_run)

    if not args.dry_run and (paths.data_dir / "sparse").exists():
        raise PipelineError(f"{paths.data_dir / 'sparse'} 存在：训练会走 COLMAP 读取器，"
                            "请确认并移除该目录（流水线不会自动删除）")

    if args.skip_stage1:
        print("[SKIP] stage1 (--skip_stage1)")
    elif ckpt1.is_file() and not args.force:
        print(f"[SKIP] stage1 已完成: {ckpt1}")
    else:
        run_step("stage1", build_stage1_cmd(paths.data_dir, paths.stage1_dir),
                 paths.log_dir, args.dry_run)

    masks_dir = resolve_masks_dir(paths.scene_dir, paths.work_dir)
    if masks_dir is not None and masks_complete(paths.data_dir, masks_dir) and not args.force:
        print(f"[SKIP] mask 已完成: {masks_dir}")
    elif args.auto_mask:
        run_step("masks", build_generate_masks_cmd(paths.work_dir, paths.data_dir),
                 paths.log_dir, args.dry_run)
        masks_dir = paths.work_dir / "masks_ir"
        if not args.dry_run and not masks_complete(paths.data_dir, masks_dir):
            raise PipelineError(f"mask 生成后仍与 train/ 不一一对应: {masks_dir}")
    elif masks_dir is not None:
        raise PipelineError(f"{masks_dir} 中 mask 与 train/ 不一一对应；补齐 mask 或加 --auto_mask 重新生成")
    else:
        raise PipelineError("没有 mask：把每视角 mask 放到 <scene>/masks_ir/（文件名=图片 stem），"
                            "或加 --auto_mask 用 rembg 生成")

    metal_kwargs = decide_metalness(args, paths, args.dry_run)
    check_confidence(args, metal_kwargs, args.dry_run)

    if not args.force and prepare_is_complete(prob, ckpt1):
        print(f"[SKIP] material_prior 已存在: {prob}")
    else:
        if not args.dry_run and prob.is_file():
            print(f"[INFO] {prob} 早于阶段一 checkpoint，重新生成")
        if not args.dry_run and not ckpt1.is_file():
            raise PipelineError(f"缺少阶段一 checkpoint: {ckpt1}（先跑阶段一或去掉 --skip_stage1）")
        run_step("prepare", build_prepare_cmd(
            paths.data_dir, paths.prep_dir, ckpt1, masks_dir,
            paths.data_dir / "material_prior", args.erode, **metal_kwargs),
            paths.log_dir, args.dry_run)

    ckpt2 = paths.stage2_dir / "chkpnt30000.pth"
    if args.skip_stage2:
        print("[SKIP] stage2 (--skip_stage2)")
    elif ckpt2.is_file() and not args.force:
        print(f"[SKIP] stage2 已完成: {ckpt2}")
    else:
        if not args.dry_run:
            if (paths.data_dir / "sparse").exists():
                raise PipelineError(f"{paths.data_dir / 'sparse'} 存在，拒绝训练")
            if not ckpt1.is_file():
                raise PipelineError(f"缺少阶段一 checkpoint: {ckpt1}")
            if not prob.is_file():
                raise PipelineError(f"缺少 {prob}（先跑 prepare_object_metalness）")
        run_step("stage2", build_stage2_cmd(paths.data_dir, paths.stage2_dir, ckpt1, prob,
                                            args.temperature), paths.log_dir, args.dry_run)

    print(f"[INFO] 产物目录: {paths.work_dir}")
    stats = paths.data_dir / "material_prior" / "metal_prior_stats.json"
    if stats.is_file():
        print("[INFO] metal_prior_stats.json:",
              json.dumps(read_meta(stats), ensure_ascii=False)[:400])
    print(f"[INFO] TensorBoard: tensorboard --logdir {paths.work_dir}")
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except PipelineError as e:
        print("[ERROR]", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
