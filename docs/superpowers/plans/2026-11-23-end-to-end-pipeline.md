# 端到端流水线（run_pipeline.py）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让一组自采数据只需 `bash run_scene.sh <场景名>` 就完成 COLMAP→Blender 转换、阶段一、mask、物体级 metalness、逐点投影投票、阶段二全流程。

**Architecture:** 一个纯 Python 编排器（`script/run_pipeline.py`）按步骤表调用已有脚本与 `train.py`，所有中间产物落在 `work/<scene>/`，输入目录只读；COLMAP→Blender 转换器（`script/convert_colmap_to_blender.py`）独立可执行、不依赖 torch，可纯 CPU 单测。编排逻辑拆成纯函数（路径解析、命令构造、完成判据）+ 薄执行层（子进程 + tee 日志），纯函数用直接运行的测试脚本覆盖。

**Tech Stack:** Python 3.12（服务器 Linux + CUDA；本地 Windows 仅跑转换器与 CPU 测试）、numpy、cv2、plyfile、subprocess、bash wrapper。

**Spec:** `docs/superpowers/specs/2026-11-23-end-to-end-pipeline-design.md`（执行时必须同时读它；本计划的裁定以 spec 为准，冲突处以本计划的"裁定"为准）

## Global Constraints

- 训练永远走 Blender 读取器：`work/<scene>/data/` 只允许 `transforms_train.json`、`train/`、`points3d.ply`、`material_prior/`；**任何步骤都不创建 `data/sparse` 或 `data/masks`**，训练前断言 `data/sparse` 不存在。
- mask 只放 `<scene>/masks_ir/`（dataset 根）或 `work/<scene>/masks_ir/`（自动生成）。
- 不修改任何训练代码（`train.py`、`gaussian_renderer/`、`scene/gaussian_model.py`、`arguments/`）。现有文件只改两处：`scene/dataset_readers.py`（storePly 改为 import）、`generate_masks.py`（加两个可选目录参数）。
- 复用 `scene/colmap_loader.py` 的读取函数与 `storePly`，不重写 COLMAP 解析。
- 训练命令参数必须与 `docs/L2完整训练流程.md` 逐字一致（零正则列表见 Task 3 的常量）。
- 图像统一输出 8-bit 3 通道 PNG（`scene/utils.py:load_img_rgb` 一律 `/255` 且不补通道）。
- 转换器与流水线纯逻辑部分**不得 import torch / scene / gaussian_renderer**（本地无 `simple_knn`，`import scene` 会失败）。
- 测试脚本直接运行（`python script/_test_*.py`），不引入 pytest；打印 `FAILURES: N` 并以非零退出码表示失败。
- 所有子进程以仓库根为 cwd 执行（`script/` 与 `train.py` 都依赖 cwd 导入）。
- 本分支 `change_fashelv` 仅本地，不 push；`script/query_vlm_metalness.py` 内嵌真实 API key，任何输出与日志不得打印它。

## Review Focus

1. **COLMAP sparse 目录两种形态**（`raw/sparse/0/` 与 `raw/sparse/` 直接放文件）：两者都必须能被解析，否则真实数据一放就停在第 1 步。
2. **text 格式 COLMAP 的非 PINHOLE 相机**：`read_intrinsics_text` 内部断言 `model == "PINHOLE"`，必须转成清晰报错（提示用 binary 或先 undistort），不能静默把 `SIMPLE_RADIAL` 的 cx 当 fy。
3. **灰度 / 16-bit / RGBA 红外图**：必须变成 8-bit 3 通道，RGBA 去 alpha 而不做预乘，否则亮度与训练读取约定不一致。
4. **半截产物 / 旧产物**：mask 必须按 stem 与 `train/` 完全一致才算完成；两阶段只看 `chkpnt30000.pth`；stem 不一致（多 mask 或漏 mask）不能当成完成。
5. **`-d` 传绝对路径且 basename 为保留名**（`raw`/`data`/`sparse`/`train`/`masks`/`masks_ir`/`material_prior`）：work 目录会撞车，必须报错。

---

### Task 1: PLY 写入模块抽出 + COLMAP→Blender 转换器 + CPU 测试

**Files:**
- Create: `utils/ply_utils.py`
- Modify: `scene/dataset_readers.py:145-161`（删除 `def storePly`，改为 import）
- Create: `script/convert_colmap_to_blender.py`
- Create: `script/_test_convert.py`

**Interfaces:**
- Consumes: `scene/colmap_loader.py` 的 `read_intrinsics_binary/read_intrinsics_text/read_extrinsics_binary/read_extrinsics_text/read_points3D_binary/read_points3D_text`；`utils/ply_utils.py:storePly(path, xyz, rgb, normals=None)`。
- Produces:
  - `utils/ply_utils.storePly(path: str, xyz: np.ndarray[N,3], rgb: np.ndarray[N,3], normals=None) -> None`（原实现逐字搬移）
  - `script/convert_colmap_to_blender.convert_colmap_to_blender(sparse_dir: str, images_dir: str, out_dir: str) -> dict`，返回 `{"frames": int, "points": int, "camera_models": list[str]}`；失败抛 `ValueError`（中文消息）。
  - CLI：`python script/convert_colmap_to_blender.py --sparse <dir> --images <dir> --out <dir>`（三者必填）。

**裁定（与 spec 文字的偏差，已核对代码）:**
- spec §8 说 text 回退支持 `SIMPLE_PINHOLE`/`PINHOLE`，但 `colmap_loader.read_intrinsics_text` 断言只有 `PINHOLE`。本任务 text 实际只支持 `PINHOLE`；binary 支持 `SIMPLE_PINHOLE`/`PINHOLE`/`SIMPLE_RADIAL`/`RADIAL`/`OPENCV`（后三者取 fx,fy,cx,cy 并 WARN 丢弃畸变）。非 PINHOLE 的 text 模型抛错，消息含 "undistort" 与 "binary"。
- 不把 `storePly` 整体搬走会让转换器间接 import `scene.dataset_readers` → `simple_knn`，本地测试无法运行。因此抽到 `utils/ply_utils.py`；`dataset_readers` 改为 `from utils.ply_utils import storePly`，其余使用者（模块内 4 处、`vismvsnet/prepare_neilf.py` 有自己的一份）不受影响。
- 相机中心/位姿校验在测试里**复刻读取器三行语义**（`c2w[:3,1:3] *= -1` → `w2c=inv(c2w)`），另加一段"若本地能 import scene 则用 `readNerfSyntheticInfo` 复核，ImportError 则打印 SKIP"的守卫块（spec §12 的要求在服务器上仍然被满足）。

- [ ] **Step 1: 写失败测试 `script/_test_convert.py`**

测试用 `sys.path.insert(0, repo/script)` 后 `import convert_colmap_to_blender`；在 `tempfile.TemporaryDirectory()` 里构造 text 模型：

```
sparse/cameras.txt :  "1 PINHOLE 8 6 10 10 4 3" 与 "2 PINHOLE 8 6 10 10 4 3"
sparse/images.txt  :  相机 1：qvec=[1,0,0,0], tvec=[0,0,5], name="img1.png"，第二行两条观测 "4 3 -1 5 5 -1"
                      相机 2：qvec=[√2/2,0,0,√2/2]（绕 Z 90°）, tvec=[1,0,5], name="missing.png"，第二行空
images/img1.png    :  16-bit 灰度 (dtype uint16, 形状 6x8)
points3D.txt       :  5 行，如 "1 0 0 0 255 0 0 0.5 0 0 1 0"
```

断言（每条一个 `check(...)` 调用，风格同 `utils/_test_mask.py`）：

```python
info = convert_colmap_to_blender.convert_colmap_to_blender(sparse, images, out)
check("frames skip missing image", info["frames"] == 1)          # 相机 2 的图不存在 → WARN 跳过
check("points", info["points"] == 5)
check("model", info["camera_models"] == ["PINHOLE"])
contents = json.load(open(os.path.join(out, "transforms_train.json")))
check("camera_angle_x", abs(contents["camera_angle_x"] - 2*math.atan(8/(2*10))) < 1e-9)
frames = contents["frames"]
check("file_path", frames[0]["file_path"] == "train/img1")
M = np.array(frames[0]["transform_matrix"])
check("json 4x4", M.shape == (4, 4) and np.allclose(M[3], [0, 0, 0, 1]))
# 复刻读取器：negate cols 1:3 后应还原 COLMAP c2w（相机1 是单位旋转）
c2w = M.copy(); c2w[:3, 1:3] *= -1
w2c = np.linalg.inv(c2w)
check("reader pose roundtrip", np.allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
      and np.allclose(w2c[:3, 3], [0, 0, 5], atol=1e-6))
img = cv2.imread(os.path.join(out, "train", "img1.png"), cv2.IMREAD_UNCHANGED)
check("8-bit 3ch", img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3)
check("ply count", len(plyfile.PlyData.read(os.path.join(out, "points3d.ply"))["vertex"]) == 5)
check("no sparse/masks", not os.path.exists(os.path.join(out, "sparse"))
      and not os.path.exists(os.path.join(out, "masks")))
```

附加用例（各自独立函数/重新构造 fixture）：
- RGBA 输入：`images/img1.png` 写成 BGRA、alpha=0 的纯红；转换后断言 3 通道且像素 ≈ 红（alpha 被丢弃）。
- 非 PINHOLE text：`cameras.txt` 写 `1 SIMPLE_RADIAL 8 6 10 4 3 0.0`，断言 `ValueError` 消息同时含 `"PINHOLE"` 与 `"undistort"`。
- 读取器守卫块：

```python
try:
    from scene.dataset_readers import readNerfSyntheticInfo
except ImportError:
    print("readNerfSyntheticInfo: SKIP (本地缺 simple_knn/pyexr)")
else:
    cams = readNerfSyntheticInfo(out, False, False).train_cameras
    check("reader cams", len(cams) == 1 and cams[0].image.shape[-1] == 3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_convert.py`
Expected: `ModuleNotFoundError: No module named 'convert_colmap_to_blender'`

- [ ] **Step 3: 实现 `utils/ply_utils.py` 并改 `dataset_readers.py`**

`ply_utils.py` 只 import `numpy`、`from plyfile import PlyData, PlyElement`，函数体逐字搬自 `scene/dataset_readers.py:145-161`。`dataset_readers.py` 删除该函数，在文件顶部 import 区加 `from utils.ply_utils import storePly`（保留 `fetchPly` 原样）。

- [ ] **Step 4: 实现 `script/convert_colmap_to_blender.py`**

签名与流程（`import numpy as np, cv2, os, json, sys, argparse`；`from scene.colmap_loader import ...`；`from utils.ply_utils import storePly`；模块顶部 `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`）：

- `_load_sparse(sparse_dir) -> (cameras, images, xyz, rgb)`：每个文件独立优先 `.bin` 再 `.txt`（`cameras.bin/txt`、`images.bin/txt`、`points3D.bin/txt`）；全缺则 `ValueError`。`read_points3D_text` 可能返回 `xyz=None`（空文件）→ 报错。
- `_qvec2rotmat(qvec) -> np.ndarray[3,3]`：标准 COLMAP 四元数转旋转矩阵（`w,x,y,z` 顺序）。
- `_camera_params(cam) -> (fx, fy, cx, cy)`：按 `cam.model` 分支；`SIMPLE_PINHOLE`=`(f,f,cx,cy)`、`PINHOLE`=`(fx,fy,cx,cy)`、`SIMPLE_RADIAL`=`(f,f,cx,cy)`、`RADIAL`=`(f,f,cx,cy)`、`OPENCV`=`(fx,fy,cx,cy)`；其余模型抛错列出模型名。后三种打印一次 `[WARN] 忽略 <model> 畸变参数，请先 colmap image_undistorter`。
- 逐图：`src = os.path.join(images_dir, os.path.basename(img.name))`；不存在 → `[WARN] 图片缺失，跳过 <name>` 并 continue；stem 冲突（不同图像 basename 相同）→ `ValueError`。
- `_convert_image(src, dst)`：`cv2.imread(src, cv2.IMREAD_UNCHANGED)`；`uint16` → `/65535*255` 取整并 `[WARN] 16-bit 转 8-bit`；float → `clip(0,1)*255`；2 维 → `COLOR_GRAY2BGR`；4 通道 → `COLOR_BGRA2BGR`；`cv2.imwrite(dst, img)`（PNG）。
- `transforms_train.json`：`{"camera_angle_x": 2*atan(cam.width/(2*fx)), "frames": [...]}`，frames 按 image name 排序；`file_path = "train/" + stem`；`transform_matrix`：

```python
w2c = np.eye(4)
w2c[:3, :3] = _qvec2rotmat(img.qvec)
w2c[:3, 3] = img.tvec
c2w = np.linalg.inv(w2c)
c2w[:3, 1:3] *= -1          # 预先抵消 Blender 读取器的 c2w[:3,1:3] *= -1
```

- `points3d.ply`：`storePly(os.path.join(out_dir, "points3d.ply"), xyz, rgb)`（rgb 为 0-255 浮点也可，storePly 内转 u1）；点数 < 1000 打印 WARN。
- `os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)`。
- CLI `main()`：`try: convert... except ValueError as e: print("[ERROR]", e); sys.exit(1)`；成功打印 `[INFO] frames=..., points=..., models=...`。

- [ ] **Step 5: 运行测试确认通过**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_convert.py`
Expected: 全部 `OK`，末行 `FAILURES: 0`，退出码 0（本地 `readNerfSyntheticInfo` 行显示 SKIP）

- [ ] **Step 6: 提交**

```bash
git add utils/ply_utils.py scene/dataset_readers.py script/convert_colmap_to_blender.py script/_test_convert.py
git commit -m "feat(script): add COLMAP to Blender converter with CPU tests"
```

---

### Task 2: `generate_masks.py` 目录覆盖参数

**Files:**
- Modify: `generate_masks.py:9-30`（函数签名与路径）、`:62-67`（parser）

**Interfaces:**
- Consumes: 现有 `create_masks(source_path)` 行为。
- Produces: `create_masks(source_path, images_dir=None, masks_dir=None) -> None`；CLI `--images_dir`、`--masks_dir`（默认空 → 保持 `<source_path>/images`、`<source_path>/masks`）。`-s` 仍必填（spec §10）。

- [ ] **Step 1: 修改函数与 CLI**

`create_masks` 内两行改为：

```python
image_dir = images_dir or os.path.join(source_path, "images")
mask_dir = masks_dir or os.path.join(source_path, "masks")
```

parser 增加：

```python
parser.add_argument("--images_dir", type=str, default="", help="输入图像目录，默认 <source>/images")
parser.add_argument("--masks_dir", type=str, default="", help="输出掩码目录，默认 <source>/masks")
```

调用处 `create_masks(args.source_path, args.images_dir or None, args.masks_dir or None)`。

- [ ] **Step 2: 语法检查（本地无 rembg，无法 import）**

Run: `"D:/Program Files/Python/Python312/python.exe" -m py_compile generate_masks.py`
Expected: 退出码 0、无输出

- [ ] **Step 3: 提交**

```bash
git add generate_masks.py
git commit -m "feat(script): allow custom image/mask dirs in generate_masks"
```

---

### Task 3: `run_pipeline.py` 纯逻辑（路径、检测、命令构造、完成判据）+ 单测

**Files:**
- Create: `script/run_pipeline.py`（本任务只写纯函数与常量；`main()` 在 Task 4 补齐，本任务先放一个只做参数解析后调用 `raise SystemExit(0)` 的骨架）
- Create: `script/_test_pipeline.py`

**Interfaces:**
- Consumes: Task 1 的 `script/convert_colmap_to_blender.py` CLI；Task 2 的参数。
- Produces（Task 4 依赖这些名字与类型）:
  - `class PipelineError(RuntimeError)`
  - `ROOT: Path`（`Path(__file__).resolve().parents[1]`）
  - `@dataclass Paths`: `scene_dir, work_dir, data_dir, stage1_dir, prep_dir, stage2_dir, log_dir, meta_path`（全 `Path`）
  - `detect_input(scene_dir: Path) -> str` → `"blender_root" | "blender_raw" | "colmap"`
  - `resolve_paths(dataset_root: Path, work_root: Path, dataset: str) -> Paths`
  - `resolve_sparse_dir(raw_dir: Path) -> Path`
  - `validate_input(paths: Paths, kind: str) -> None`
  - `expected_stems(data_dir: Path) -> set[str]`
  - `resolve_masks_dir(scene_dir: Path, work_dir: Path) -> Path | None`
  - `masks_complete(data_dir: Path, masks_dir: Path | None) -> bool`
  - `build_convert_cmd(sparse_dir, images_dir, out_dir) -> list[str]`
  - `build_generate_masks_cmd(work_dir, data_dir) -> list[str]`
  - `build_vlm_cmd(scene_dir, out_path, model) -> list[str]`
  - `build_prepare_cmd(data_dir, prep_dir, ckpt, masks_dir, out_dir, erode, metallic=None, scene_meta=None) -> list[str]`
  - `build_stage1_cmd(data_dir, out_dir) -> list[str]`
  - `build_stage2_cmd(data_dir, out_dir, ckpt, prob_path, temperature) -> list[str]`
  - 常量 `STAGE1_ZERO` / `STAGE2_ZERO`（`tuple[tuple[str, str], ...]`）与 `STAGE2_FREEZE`，值见下。

**固定值（与 `docs/L2完整训练流程.md` 一致）:**

```python
STAGE1_ZERO = (("--lambda_depth", "0"), ("--lambda_depth_smooth", "0"), ("--lambda_depth_var", "0"),
               ("--lambda_mask_entropy", "0"), ("--lambda_opacity", "0"), ("--lambda_surface", "0"),
               ("--lambda_scaling", "0"), ("--lambda_normal_render_depth", "0"),
               ("--lambda_normal_mvs_depth", "0"), ("--lambda_normal_smooth", "0"),
               ("--lambda_point_entropy", "0"), ("--lambda_orientation", "0"))
STAGE2_FREEZE = (("--densify_until_iter", "0"), ("--position_lr_init", "0"), ("--position_lr_final", "0"),
                 ("--normal_lr", "0"), ("--rotation_lr", "0"), ("--scaling_lr", "0"),
                 ("--opacity_lr", "0"), ("--sh_lr", "0"))
STAGE2_ZERO = (("--lambda_depth", "0"), ("--lambda_depth_smooth", "0"), ("--lambda_mask_entropy", "0"),
               ("--lambda_opacity", "0"), ("--lambda_surface", "0"), ("--lambda_normal_render_depth", "0"),
               ("--lambda_normal_smooth", "0"), ("--lambda_point_entropy", "0"), ("--lambda_orientation", "0"),
               ("--lambda_depth_var", "0"), ("--lambda_scaling", "0"), ("--lambda_light", "0"),
               ("--lambda_base_color", "0"), ("--lambda_base_color_smooth", "0"),
               ("--lambda_roughness_smooth", "0"), ("--lambda_light_smooth", "0"),
               ("--lambda_visibility", "0"), ("--lambda_visibility_smooth", "0"), ("--lambda_env_smooth", "0"))
```

- [ ] **Step 1: 写失败测试 `script/_test_pipeline.py`**

风格同 `utils/_test_mask.py`；夹具在临时目录手工建三类布局。断言：

- `detect_input`：`(root)/transforms_train.json + train/` → `"blender_root"`；`raw/transforms_train.json` → `"blender_raw"`；`raw/sparse/0/cameras.txt` → `"colmap"`；空目录 → `PipelineError`。
- `resolve_paths`：名字 `kettle` → `work_root/"kettle"`；绝对路径 `tmp/abc` → `work_dir == work_root/"abc"`；basename 为 `raw`、`data`、`sparse`、`train`、`masks`、`masks_ir`、`material_prior` 各抛 `PipelineError`。
- `resolve_sparse_dir`：`raw/sparse/0/cameras.bin` 存在 → 返回 `.../0`；只有 `raw/sparse/cameras.txt` → 返回 `.../sparse`；两者都无 → `PipelineError`。
- `validate_input`：blender 形态缺 `points3d.ply` → 抛错且消息含 `points3d.ply`；`train/` 空 → 抛错；colmap 形态缺 `raw/images` 或为空 → 抛错。
- `masks_complete`：mask stem 集合与 `train/` 完全一致 → True；多一个 mask → False；少一个 → False；`masks_dir is None` → False。
- `resolve_masks_dir`：dataset 根有 `masks_ir/` → 返回它；只有 `work/<scene>/masks_ir` → 返回它；都没有 → `None`。
- `build_stage1_cmd(d, o)`：含 `-s`/`-t`,`render`/`-m`；`STAGE1_ZERO` 每一项按相邻顺序出现。
- `build_stage2_cmd(d, o, ckpt, prob, "30")`：含 `--checkpoint`,ckpt；`--material_prob_path`,prob；`--emissivity_mode`,`lut`；`--temperature`,`30`；`STAGE2_FREEZE`、`STAGE2_ZERO` 全部出现；不含 `--material_prob_path` 以外的 `--metallic`。
- `build_prepare_cmd(..., erode=3, metallic=None, scene_meta=meta)`：含 `-s`,d；`-m`,prep；`--checkpoint`,ckpt；`--masks_dir`,masks；`--out_dir`,d/"material_prior"；`--erode`,`3`；`--scene_meta`,meta；给 `metallic=0.8` 时改为含 `--metallic`,`0.8` 且不含 `--scene_meta`。
- `build_generate_masks_cmd(work, data)`：含 `-s`,work，`--images_dir`,data/"train"，`--masks_dir`,work/"masks_ir"。
- `build_vlm_cmd(scene, out, "qwen3-vl-flash")`：含 `-s`,scene，`--out`,out，`--model`,`qwen3-vl-flash`，`--missing`,`error`。
- 所有 build 函数返回 `list[str]` 且不含 `None`。

- [ ] **Step 2: 运行确认失败**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_pipeline.py`
Expected: `ModuleNotFoundError: No module named 'run_pipeline'`

- [ ] **Step 3: 实现纯函数与常量**

关键规则：
- `resolve_paths`：`p = Path(dataset)`；`is_abs = p.is_absolute()`；`work_name = p.name if is_abs else dataset`；`work_name in RESERVED` → `PipelineError`；`scene_dir = p if is_abs else dataset_root/dataset`。不创建任何目录。
- `detect_input` 顺序：`scene/transforms_train.json` 且 `scene/train` 存在 → blender_root；`scene/raw/transforms_train.json` 且 `scene/raw/train` → blender_raw；`scene/raw/sparse` 存在 → colmap；否则 `PipelineError`。
- `validate_input`：blender 形态要求源目录（`scene_dir` 或 `scene_dir/"raw"`）内 `transforms_train.json`、`train/` 非空（至少一张 `.png/.jpg/.jpeg`）、`points3d.ply` 三者存在；colmap 形态要求 `scene_dir/"raw"/"images"` 非空且 `resolve_sparse_dir(scene/raw)` 成功。错误消息写清缺什么。
- `expected_stems`：`train/` 下图片 stem 集合。
- `masks_complete`：`masks_dir` 存在时 `{p.stem for p in masks_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")} == expected_stems(data_dir)`。
- `build_*` 用 `sys.executable` + 相对路径（`"script/convert_colmap_to_blender.py"`、`"generate_masks.py"`、`"script/query_vlm_metalness.py"`、`"script/prepare_object_metalness.py"`、`"train.py"`），路径参数统一 `str(path)`。
- 骨架 CLI：argparse 定义 Task 4 的全部参数（`-d/--dataset` required、`--dataset_root` 默认 `ROOT/"datasets"`、`--work_root` 默认 `ROOT/"work"`、`--temperature` float 默认 None、`--auto_mask`、`--metallic` float、`--vlm_model` 默认 `qwen3-vl-flash`、`--erode` int 默认 3、`--force`、`--skip_stage1`、`--skip_stage2`、`--strict_vlm`、`--dry_run`），`main(argv=None)` 本任务内只 `return 0`。

- [ ] **Step 4: 运行确认通过**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_pipeline.py`
Expected: `FAILURES: 0`

- [ ] **Step 5: 提交**

```bash
git add script/run_pipeline.py script/_test_pipeline.py
git commit -m "feat(script): add pipeline path resolution and command builders"
```

---

### Task 4: 执行层、编排、`run_scene.sh`、CLI 级测试

**Files:**
- Modify: `script/run_pipeline.py`（新增 `run_step`、`copy_training_files`、`read_meta`、完整 `main()`）
- Modify: `script/_test_pipeline.py`（追加 CLI 级用例）
- Create: `run_scene.sh`

**Interfaces:**
- Consumes: Task 3 的全部函数/常量；Task 1/2 的脚本 CLI。
- Produces:
  - `run_step(name: str, cmd: list[str], log_dir: Path, dry_run: bool = False) -> None`（失败抛 `PipelineError`，消息含 exit code 与日志路径）
  - `copy_training_files(src: Path, data_dir: Path, dry_run: bool) -> None`
  - `read_meta(path: Path) -> dict`
  - `decide_metalness(args, paths: Paths, dry_run: bool) -> dict`（返回 `{"metallic": x}` 或 `{"scene_meta": path}`）
  - `main(argv: list[str] | None = None) -> int`（0 成功，2 业务错误）

**步骤行为（对应 spec §9 表格）:**

| 步 | name（日志名） | 行为 | 跳过条件（非 `--force`） |
|---|---|---|---|
| 0 | — | `detect_input` + `validate_input`；`--skip_stage2` 可省 temperature，否则 temperature 为 None 直接 `PipelineError` | — |
| 1 | `convert` / `copy` | blender 形态复制 `transforms_train.json`、`train/`（`shutil.copytree(dir, dir, dirs_exist_ok=True)`）、`points3d.ply`；colmap 形态跑 `build_convert_cmd` | `data/transforms_train.json` 存在 |
| 1b | — | 断言 `data/sparse` 不存在（`PipelineError`） | — |
| 2 | `stage1` | `build_stage1_cmd` | `stage1/chkpnt30000.pth` 存在 |
| 3 | `masks` | 解析 mask 目录；缺失且 `--auto_mask` → `build_generate_masks_cmd`；之后必须 `masks_complete`，否则报错 | `masks_complete` |
| 4 | `vlm`（仅自动时） | `--metallic` > `work/<scene>/scene_meta.json` > `dataset/<scene>/scene_meta.json` > 有 `visible/` 时跑 VLM（`--missing error`，输出 `work/<scene>/scene_meta.json`，`DASHSCOPE_API_KEY` 缺失时报错）；全无则报错。meta 的 `confidence < 0.5` → WARN，`--strict_vlm` → 报错 | VLM 子步：meta 已存在 |
| 5 | `prepare` | 断言 `stage1/chkpnt30000.pth`；`build_prepare_cmd(..., erode, metallic 或 scene_meta)` | `data/material_prior/metal_prob.npy` 存在 |
| 6 | `stage2` | 断言 temperature 与 `data/sparse` 不存在；`build_stage2_cmd` | `stage2/chkpnt30000.pth` 存在 |
| 7 | — | 打印产物路径、`metal_prior_stats.json` 存在时打印其 `json.dumps` 前 400 字符、TensorBoard 命令 | — |

- 步骤 3 无 mask 且无 `--auto_mask`：报错消息同时给出 `masks_ir/` 与 `--auto_mask` 两个选项。
- `--force` 只重新执行，不删除任何已有目录/文件（红线：不删文件）。
- `--skip_stage1` 跳过步骤 2（步骤 5 的断言语义不变）；`--skip_stage2` 跳过步骤 6（步骤 4/5 仍可运行）。
- `run_step` 用 `subprocess.Popen(cmd, cwd=ROOT, stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace")`，逐行 `print` 同时写 `<log_dir>/<name>.log`；失败时打印 `deque(maxlen=40)` 的日志尾部再抛错。
- `main` 捕获 `PipelineError` → `print("[ERROR]", e)`，`return 2`。

- [ ] **Step 1: 追加 CLI 级失败测试**

在 `_test_pipeline.py` 增加三个子进程用例（`cwd=ROOT`，`[sys.executable, "script/run_pipeline.py", ...]`，用 `subprocess.run(..., capture_output=True, text=True)`）：

1. `test_dry_run_full`：构造 blender_root 夹具（合法 `transforms_train.json`：`{"camera_angle_x": 0.8, "frames": [{"file_path": "train/a", "transform_matrix": 4x4 identity}]}`、`train/a.png` 8x6、`points3d.ply` 随便一个文件）；命令加 `--dataset_root <tmp>/ds --work_root <tmp>/work -d kettle --temperature 30 --auto_mask --metallic 0.5 --dry_run`。断言 `returncode == 0`；`out = r.stdout` 同时满足 `"train.py" in out`、`"--emissivity_mode" in out and "lut" in out`、`"generate_masks.py" in out`、`"prepare_object_metalness.py" in out`；`<tmp>/work` 目录未被创建。
2. `test_missing_temperature`：同一夹具去掉 `--temperature` 与 `--skip_stage2`，断言 `returncode != 0` 且 stderr+stdout 含 `temperature` 或 `温度`。
3. `test_missing_masks`：夹具无 `masks_ir`、不加 `--auto_mask`，断言 `returncode != 0` 且输出含 `masks_ir` 与 `--auto_mask`。

- [ ] **Step 2: 运行确认失败**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_pipeline.py`
Expected: 三个新用例 FAIL（`main` 尚未执行步骤，dry-run 无输出）；原有纯函数用例仍 OK

- [ ] **Step 3: 实现执行层与 `main`**

按上表实现。`copy_training_files` 在 `dry_run` 时只打印 `[DRY_RUN] copy <src> -> <data>`。步骤 4 里 metalness 来源解析写成小函数便于阅读：

```python
def decide_metalness(args, paths: Paths, dry_run: bool) -> dict:
    """返回 build_prepare_cmd 的 metallic/scene_meta 二选一参数。"""
```

（`--metallic` 优先；work 的 meta 优先于 dataset 的 meta；需要 VLM 时返回 `{"scene_meta": str(paths.meta_path)}` 并在返回前执行 VLM 步骤；无任何来源 → `PipelineError`。）

- [ ] **Step 4: 写 `run_scene.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python script/run_pipeline.py \
  --dataset_root /data/mwir/scenes --work_root /data/mwir/output \
  -d "$1" --temperature 30 --auto_mask "${@:2}"
```

（spec §11 原文写 `"$@"` 会把场景名重复传给 argparse，改用 `"${@:2}"`；这是本计划的裁定。）

- [ ] **Step 5: 运行全部测试确认通过**

Run: `"D:/Program Files/Python/Python312/python.exe" script/_test_pipeline.py && "D:/Program Files/Python/Python312/python.exe" script/_test_convert.py`
Expected: 两个脚本都 `FAILURES: 0`，退出码 0

- [ ] **Step 6: 提交**

```bash
git add script/run_pipeline.py script/_test_pipeline.py run_scene.sh
git commit -m "feat(script): add end-to-end pipeline entry and run_scene wrapper"
```

---

## 服务器上的验收（实现完成后由用户执行，不写进任何任务）

```bash
# 1. 一组已有 Blender 格式数据（形态 1）
bash run_scene.sh <scene> --skip_stage1      # 或完整跑
# 2. 一组 COLMAP 原始数据（形态 3）
ls datasets/<scene>/raw/sparse datasets/<scene>/raw/images
bash run_scene.sh <scene> --auto_mask
# 3. 检查 work/<scene>/data 无 sparse/、masks/，且 masks_ir 与 train/ 的 stem 一一对应
```

## 附：spec 覆盖自查

| spec 章节 | 覆盖任务 |
|---|---|
| §6 三种输入形态、points3d.ply 必需 | Task 3 检测/校验 + Task 1 转换 |
| §7 产物布局、prep 目录 | Task 3 `Paths` + Task 4 步骤 5 |
| §8 转换规则（模型、跳过缺图、8-bit、json、ply） | Task 1 |
| §9 步骤表、完成判据、temperature、VLM、dry_run、tee | Task 3 判据/命令 + Task 4 编排 |
| §10 generate_masks 可选目录 | Task 2 |
| §11 新文件清单 | Task 1/3/4 |
| §12 CPU 测试 | Task 1（含服务器守卫块）+ Task 3/4 CLI 测试 |
| §13 风险表中的 16-bit/灰度、点云过少、mask stem 不一致 | Task 1 测试与 WARN、Task 3 `masks_complete` |
| §14 留给实现者的决定 | Task 4（tee、日志尾、dry_run 排版）；Task 1（text 二维观测由 `read_extrinsics_text` 自行忽略） |
| §15 评审裁定 | Global Constraints + Task 3/4 规则 |
