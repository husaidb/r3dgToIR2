# 设计：一组自采数据的端到端流水线（run_pipeline.py）

日期：2026-11-23
状态：待评审

## 1. 意图与为什么

自采 MWIR 数据集目前需要手工执行六七个步骤（COLMAP 输出 → 转 Blender 格式 → 阶段一 → mask → VLM → prepare_object_metalness → 阶段二），每步命令分散在 `docs/L2完整训练流程.md` 里，参数多、路径容易错、中间产物位置不统一。目标是把整条链路收进一个 Python 入口：

```bash
bash run_scene.sh kettle
```

换一组数据只改场景名。用户只需要按约定放好 COLMAP 原始输出和图片，其余（格式转换、掩码、metalness、两阶段训练）由流水线完成。

## 2. 目标

- 一条命令跑完单场景：转换 → 阶段一 → mask → 物体级 metalness → 逐点投影投票 → 阶段二
- 幂等：某步产物已存在则跳过，`--force` 强制重跑
- 失败即停：任一步非零退出，打印该步日志尾部并退出非零
- 从不修改输入数据：`datasets/<scene>/` 只读，所有权产物落在 `work/<scene>/`
- 转换器可独立单测（纯 CPU）

## 3. 非目标

- 不做评测（`eval_relighting_syn4.py` 不接入）
- 不做自动调参：λ 全 0 的固定配置（同训练文档）、temperature 由 wrapper 传入
- 不做训练中断续跑（只做"产物存在则跳过"）
- 不修 `dataset_readers.py:113` 的 COLMAP mask 空操作（见 §5 规避）
- 不做多卡、不做 VLM 重试/人工审核流程
- 不生成可见光照片

## 4. 反目标（技术上都成功也算失败）

- 改变现有训练语义或默认参数
- 把转换逻辑塞进 `train.py` 或改动训练读取器
- 静默失败（吞掉子进程错误、隐藏 WARN）
- 流水线产物散落在输入目录

## 5. 约束

1. **训练永远走 Blender 读取器**：`work/<scene>/data/` 只含 `transforms_train.json`、`train/`、`points3d.ply`、`material_prior/`；**禁止**创建 `sparse/`，训练前断言其不存在。
2. **mask 只放 `masks_ir/`**：dataset 目录和 work 目录都不创建 `masks/`，这是第一道防线。COLMAP 读取器的 mask bug（原 `scene/dataset_readers.py:113`：`1.0 - load_mask_bool(...)/255` 恒为 `{1.0, 0.996}`）已于 2026-11-23 修复：改为 `utils/mask_utils.py:load_colmap_foreground_mask`（白=前景，背景置黑，支持灰度/RGBA/尺寸缩放，含 CPU 测试 `utils/_test_mask.py`）。直接以 COLMAP 布局训练时不再静默失效。
3. 仓库已有 `scene/colmap_loader.py`（binary + text 读取）与 `storePly`，转换器必须复用，不重写解析。
4. 服务器为 Linux + CUDA；本地 Windows 只跑转换器与 CPU 测试。
5. 训练参数与 `docs/L2完整训练流程.md` 完全一致（细节正则全 0、阶段二冻结几何、`--densify_until_iter 0`）。
6. 图像统一转 8-bit 3 通道 PNG：`scene/utils.py:load_img_rgb` 对非 EXR 一律 `/255` 且不补齐通道，16-bit 或灰度图会静默出错。

## 6. 输入布局

基准根可配置：`--dataset_root`（默认 `<repo>/datasets`）、`--work_root`（默认 `<repo>/work`）。`-d` 可传场景名（在 `dataset_root` 下）或绝对路径（此时 work 目录为 `<work_root>/<basename>`，`basename` 不能是 `raw`、`data` 等保留名）。服务器上的 `run_scene.sh` 显式传入 `/data/mwir/scenes` 与 `/data/mwir/output`。

```
datasets/<scene>/
├── raw/
│   ├── sparse/0/            # COLMAP 输出（binary 或 text）
│   └── images/              # 原始图片
├── masks_ir/                # 可选：每视角 mask，文件名 = 图片名去扩展名
├── visible/                 # 可选：VLM 用可见光照片
└── scene_meta.json          # 可选：手工填写（VLM 产物写 work/，见 §9）
```

支持三种输入形态，按优先级检测（`masks_ir/`、`visible/`、`scene_meta.json` 一律在场景目录根查找）：

1. 场景目录根下有 `transforms_train.json` + `train/` → 已是 Blender 格式，把训练件（json、`train/`、`points3d.ply`）复制到 `data/`
2. `raw/transforms_train.json` 存在 → 同上，源为 `raw/`
3. `raw/sparse/` 存在 → 走 §8 转换

形态 1 是为了兼容现有已转好的自采数据（旧布局就是根即 Blender）。**三种形态都要求源侧 `points3d.ply` 存在**，缺失直接报错——不允许 Blender 读取器静默随机生成 10 万点。

## 7. 产物布局

```
work/<scene>/
├── data/                    # 训练用 Blender 格式场景（-s）
│   ├── transforms_train.json
│   ├── train/*.png
│   ├── points3d.ply
│   └── material_prior/{object_prob.npy, metal_prob.npy, metal_prior_stats.json}
├── stage1/                  # 阶段一 -m（含 chkpnt30000.pth、env_light_*.pth）
├── prep/                    # prepare_object_metalness 的 -m（只写 input.ply/cameras.json）
├── stage2/                  # 阶段二 -m
├── scene_meta.json          # VLM 产物（不写回 datasets/）
└── logs/<step>.log
```

`prepare_object_metalness.py` 的 `-m` 指向 `work/<scene>/prep`（该脚本会往 `-m` 里写 `input.ply`/`cameras.json`，不能污染 `stage1/`），`-s` 指向 `data/`，并显式传 `--out_dir work/<scene>/data/material_prior`。

`masks_ir` 来自 dataset 目录；若 `--auto_mask` 生成，写入 `work/<scene>/masks_ir/`。

## 8. 转换规则（COLMAP → Blender）

新文件 `script/convert_colmap_to_blender.py`，也可作为 `run_pipeline` 的一个函数被调用，但保持可独立执行：

1. 读取 `sparse/0/{cameras,images,points3D}.bin`，失败回退 `.txt`
   - 相机模型：`SIMPLE_PINHOLE` / `PINHOLE` 精确转换；`SIMPLE_RADIAL` / `RADIAL` / `OPENCV` 从参数中取 fx,fy,cx,cy 并 WARN 忽略畸变（热像镜头畸变通常小；畸变大时先 `colmap image_undistorter`）。text 回退仅支持前两种，其余给出明确错误并提示先 undistort
2. 跳过注册了但图片文件缺失的相机（WARN）；打印相机数与相机模型
3. 图片逐张转 8-bit 3 通道 PNG → `data/train/<stem>.png`；文件名 stem 保持 COLMAP image name 的 basename（保证 mask 文件名对应关系）
4. `transforms_train.json`：
   - `camera_angle_x = 2·atan(width / (2·fx))`（SIMPLE_PINHOLE/PINHOLE）
   - `file_path = "train/<stem>"`（正斜杠）
   - `transform_matrix`：取 COLMAP 的 c2w = inv(w2c)，再把第 1、2 列的符号取反。原因：Blender 读取器会执行 `c2w[:3,1:3] *= -1`（`dataset_readers.py:238`），转换器必须预先抵消，否则位姿差一个 Y/Z 翻转
5. `points3d.ply`：`read_points3D_binary/text` 取 xyz+rgb，调用 `storePly`（normals 传 None，由 storePly 生成随机单位法线，与现有随机初始化法线行为一致）；打印点数（过少时 WARN）
6. 不做点云裁剪/清理

## 9. 流水线步骤（`script/run_pipeline.py`）

参数：

```
-d/--dataset        场景名（在 datasets/ 下）或绝对路径
--temperature       物体温度（°C）；未传时阶段二不发热辐射，打印 WARN
--auto_mask         无 masks_ir 时用 generate_masks.py 自动生成
--metallic          float，手动指定物体级 metalness（优先于 scene_meta/VLM）
--vlm_model         默认 qwen3-vl-flash
--erode             默认 3
--force             忽略已有产物重跑
--skip_stage1 / --skip_stage2
--dry_run           只打印将执行的命令，不执行
```

步骤（每步一条日志，`subprocess.run` 检查 returncode）：

| 步 | 动作 | 跳过条件 |
|---|---|---|
| 0 | 校验 dataset 布局（三种形态之一成立、图片非空、源侧 `points3d.ply` 存在） | — |
| 1 | 转换/复制到 `work/<scene>/data/` | `data/transforms_train.json` 存在且非 `--force` |
| 2 | 阶段一 `train.py -t render`（全 0 正则） | `stage1/chkpnt30000.pth` 存在 |
| 3 | 确定 mask 来源：`datasets/<scene>/masks_ir` → `work/masks_ir`；缺且 `--auto_mask` → 调用 `generate_masks.py --images_dir work/data/train --masks_dir work/masks_ir` | `work/masks_ir` 里与 `train/` 图数一致（按 stem 匹配） |
| 4 | 物体级 metalness：`--metallic` > `scene_meta.json`（dataset 根或 work 根）> 有 `visible/` 且有 `DASHSCOPE_API_KEY` 时跑 VLM（`--out work/<scene>/scene_meta.json --missing error`）；三者皆无则报错退出 | `--metallic` 已给或有 `scene_meta.json` |
| 5 | `prepare_object_metalness.py -s data -m prep --out_dir data/material_prior` | `data/material_prior/metal_prob.npy` 存在 |
| 6 | 阶段二 `train.py -t neilf`（冻结几何 + `--emissivity_mode lut` + `--material_prob_path`） | `stage2/chkpnt30000.pth` 存在 |
| 7 | 汇总：产物路径、`metal_prior_stats.json` 关键字段、提示 TensorBoard 命令 | — |

补充规则：

- **完成判据用最终产物而非"目录非空"**：mask 按 stem 计数匹配 `train/` 图数；两阶段都看 `chkpnt30000.pth`。半截产物（中断留下）视为未完成，重跑该步
- 步骤 3 如果没有 mask 且没传 `--auto_mask`，直接报错，提示两个选项（放 masks_ir/ 或加 --auto_mask）。mask 是逐点 metalness 的必需输入
- 步骤 6 要求 `--temperature`：未传且未 `--skip_stage2` 时报错（lut 无温度时热辐射项不存在，ε 不参与渲染，静默继续等于没做实验）。wrapper 固定传 30
- VLM 低置信（<0.5）只 WARN 并继续（批处理优先，事后看 `metal_prior_stats.json`）；`--strict_vlm` 可改为停在步骤 4
- 步骤 5 前断言 `stage1/chkpnt30000.pth` 存在；`metal_prob` 点数一致性由 prepare 脚本自身的断言兜底，失败即停
- 步骤 6 前断言 `work/<scene>/data/sparse` 不存在
- `dry_run` 打印每步完整命令，便于对照训练文档
- 日志用 tee 语义同时写文件与终端

## 10. 对现有文件的最小改动

- `generate_masks.py`：新增可选 `--images_dir` / `--masks_dir`，默认值与现状一致（`<source>/images`、`<source>/masks`），向后兼容。`-s` 仍为必填（流水线传 `-s work/<scene>` 再加两个目录覆盖）
- 其余现有脚本不改（`query_vlm_metalness.py` 已有 `--out` / `--missing` / `--vis_dir`，直接够用）

## 11. 新增文件

```
script/run_pipeline.py             # 编排器
script/convert_colmap_to_blender.py# 转换器（可独立运行）
script/_test_convert.py            # 转换器 CPU 单测
run_scene.sh                       # 一行 wrapper：python script/run_pipeline.py -d "$1" --temperature 30 --auto_mask "$@"
```

## 12. 测试

`script/_test_convert.py`（CPU，直接 `python script/_test_convert.py` 运行，与 `computebrdf/_test_l2.py` 同风格，不依赖 pytest）：

1. 在临时目录构造最小 COLMAP text 模型（2 台相机、1 张图对应、数十个 3D 点）
2. 跑转换器
3. 用 `readNerfSyntheticInfo` 加载产物，断言：相机数一致；相机中心与 COLMAP c2w 平移在 1e-5 内一致；`image.shape[-1] == 3`
4. 断言 ply 点数等于 points3D 数量
5. 断言 `data/sparse`、`data/masks` 均不存在

## 13. 风险

| 风险 | 处理 |
|---|---|
| rembg 在红外图失效，自动 mask 含背景 | 只作为起点；`metal_prior_stats.json` 的 `object_ratio_at_thresh` >50% 会 WARN；宁漏勿多，漏掩只在缺少金属先验，多掩会把桌子标成金属 |
| 16-bit / 灰度图 | 转换器统一转 8-bit 3 通道并 WARN |
| COLMAP 点云过少 | 打印点数并 WARN；不自动补点 |
| 未传 temperature | WARN，阶段二无热辐射项，`lut` ε 不参与渲染 |
| mask 文件名与图片 stem 不一致 | 转换器以 COLMAP 文件名 basename 为 stem，`find_mask` 按 stem 匹配；准备脚本会统计无 mask 视角数 |
| 阶段一法线无监督 | 沿用训练文档的已知取舍，不在本设计处理 |

## 14. 留给实现者的决定

- `tee` 与日志尾部的具体实现
- text 格式 `images.txt` 中二维观测列的忽略方式
- 汇总输出中提取 PSNR 的方式（可选，简单 grep 日志即可）
- `--dry_run` 输出的排版

## 15. 评审问题与裁定（设计评审后补充）

评审子代理（独立上下文）提出的问题及处理：

| 问题 | 裁定 |
|---|---|
| `datasets/`、`work/` 的根在哪 | 可配置：`--dataset_root` / `--work_root`，默认在仓库根；`-d` 支持绝对路径 |
| 已有数据是根即 Blender 格式 | 支持三种输入形态（根 Blender / raw Blender / raw COLMAP），见 §6 |
| prepare 脚本的 `-m` 污染 stage1 | 单独用 `work/<scene>/prep`，显式 `--out_dir` |
| VLM 产物写回输入目录 | 写 `work/<scene>/scene_meta.json`（`--out`），输入只读 |
| VLM 低置信是否停 | 默认 WARN 继续，`--strict_vlm` 可停 |
| 半截产物被当成完成 | 完成判据改为最终产物 + mask 计数匹配，见 §9 |
| COLMAP 相机模型可能不是 PINHOLE | binary 支持五种常用模型（畸变忽略+WARN），text 仅两种，见 §8 |
| `storePly` 逐条 tuple 写入慢 | 接受（数千~数十万点，秒级）；不复用改造 |
| 阶段二缺 temperature 怎么办 | 报错（`--skip_stage2` 除外），见 §9 |
| 兼容复制缺 `points3d.ply` 会静默随机 | 步骤 0 强制要求源侧 `points3d.ply` |
| `generate_masks.py` 的 `-s` 与目录参数关系 | `-s` 保持必填，新增两个可选目录覆盖 |
| 测试文件不被 pytest 收集 | 不做 pytest 用例，直接 `python script/_test_convert.py` |
