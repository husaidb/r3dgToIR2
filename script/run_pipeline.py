"""一组自采数据的端到端流水线：转换 → 阶段一 → mask → metalness → 阶段二。

用法：
    python script/run_pipeline.py -d <scene> --temperature 30 [--auto_mask] [--dry_run]

输入只读，产物全部落在 --work_root/<scene>/ 下。
"""
import argparse
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


def main(argv=None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
