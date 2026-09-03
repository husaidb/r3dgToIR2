## 结论概览
- 不是只支持 Ubuntu。核心是 PyTorch CUDA 扩展，Windows 可构建可运行。
- Linux/Ubuntu 专用主要是 `.sh` 脚本与 Dockerfile；扩展的 `setup.py` 已考虑 Windows。
- 参考：`submodules/simple-knn/setup.py:18-20`（Windows 特判），`nvdiffrast/setup.py:42`（Windows 打包 `lib/*.lib`），`readme.md:104-121`（训练脚本为 `sh`）、`r3dg-rasterization/CMakeLists.txt:33`（与 OS 无关）。

## 环境准备（Windows）
1. 安装匹配的 `CUDA Toolkit`（建议 11.8）与 NVIDIA 驱动；确保与 PyTorch CUDA 版本匹配。
2. 安装 `Visual Studio Build Tools`（含 C++、Windows SDK），用于编译 C++/CUDA 扩展。
3. 通过 Conda：`conda env create --file environment.yml`；`conda activate r3dg`。
4. 安装 PyTorch（带 CUDA）：`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`；再安装 `torch_scatter==2.1.1`、`kornia==0.6.12`。

## 编译与安装本地扩展
1. `pip install ./submodules/simple-knn`（Windows 特判在 `setup.py` 已处理）。
2. `pip install ./bvh`（`--expt-extended-lambda`，适配 NVCC）。
3. `pip install ./r3dg-rasterization`（包含 GLM 头文件）。
4. `git clone https://github.com/NVlabs/nvdiffrast && pip install ./nvdiffrast`（Windows 包含 `lib/*.lib`）。

## 转换训练/评测脚本为 PowerShell
- 将以下 `.sh` 逐行转为 `.ps1` 并保持命令参数一致：
  - `script/run_nerf.sh`
  - `script/run_dtu.sh`
  - `script/run_tnt.sh`
  - `script/run_syn4.sh`
  - `script/relighting.sh`
- 也可在 WSL/WSL2 中直接执行原始 `.sh`；但纯 Windows 下建议提供 `.ps1` 版本以便复现实验。

## 运行与验证
1. 训练：使用 `.ps1` 等效脚本或直接在 PowerShell 逐条运行 `python train.py ...`。
2. 评测：使用 README 中的 `python eval_nvs.py ...`、`python eval_relighting_syn4.py ...`。
3. GUI：`python gui.py -m <model_dir> -t render/neilf` 检查是否能正常显示与交互。

## 可选：数据预处理
- 若需 `COLMAP` 流程，Windows 可安装 COLMAP GUI/CLI；命令行示例需按 Windows 路径调整。
- 或在 WSL/Ubuntu 侧完成预处理，再回到 Windows 训练。

## 风险与注意
- NVCC 与 PyTorch CUDA 版本不匹配会导致编译失败；按 README 建议优先用 `CUDA 11.8` 编译扩展。
- 不要在 Windows 设置 `PYOPENGL_PLATFORM=egl`（这是 Linux/EGL 无头渲染路径）。
- 需要足够显存（论文使用 RTX3090 24G）。

## 下一步（如确认）
- 我将为 `script/*.sh` 提供一套 `.ps1` 等效脚本，适配 Windows；并添加一个简短的 `Windows 快速上手` 文档，确保你本机可一键复现训练/评测与 GUI。