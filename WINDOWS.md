# Windows 快速上手

## 先决条件
- 安装 NVIDIA 驱动与 `CUDA Toolkit 11.8`（或与 PyTorch 匹配的版本）。
- 安装 `Visual Studio Build Tools`（含 C++、Windows SDK）。
- 安装 Anaconda/Miniconda。

## 创建环境并安装依赖
- `conda env create --file environment.yml`
- `conda activate r3dg`
- `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`
- `pip install torch_scatter==2.1.1`
- `pip install kornia==0.6.12`

## 安装本地扩展
- `pip install ./submodules/simple-knn`
- `pip install ./bvh`
- `pip install ./r3dg-rasterization`
- `git clone https://github.com/NVlabs/nvdiffrast && pip install ./nvdiffrast`

## 训练与评测（PowerShell）
- NeRF Synthetic：`pwsh script/run_nerf.ps1`
- DTU：`pwsh script/run_dtu.ps1`
- Tanks and Temples：`pwsh script/run_tnt.ps1`
- Synthetic4Relight：`pwsh script/run_syn4.ps1`
- 重光照示例：`pwsh script/relighting.ps1`

## GUI
- `python gui.py -m output/NeRF_Syn/lego/3dgs -t render`
- `python gui.py -m output/NeRF_Syn/lego/neilf -t neilf`

## 注意事项
- 编译扩展建议使用 `CUDA 11.8`，避免 ABI 问题。
- 不要在 Windows 设置 `PYOPENGL_PLATFORM=egl`；这是 Linux/EGL 路径。
- 若有数据预处理需求，COLMAP 可在 Windows 安装；或使用 WSL 完成后回到 Windows 训练。
