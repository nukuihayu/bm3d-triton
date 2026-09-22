# bm3d-triton

[English](README.md) · [安装](#快速开始) · [API](#python) · [算法](#算法如何工作) · [Benchmark](#自己运行-benchmark)

BM3D Triton 使用 Triton GPU 内核实现图块匹配与协同图像滤波，提供
PyTorch 张量接口、仅推理的 `nn.Module` 封装和图片命令行工具。

算法采用 8×8 DCT、组间 Hadamard 变换和 Kaiser 窗加权聚合，支持单阶段硬阈值
以及双阶段 Wiener 细化。无需模型权重、ComfyUI 或项目专用 C++/CUDA 扩展；
Triton 会在首次调用时编译所需内核。

| 能力 | 说明 |
|---|---|
| 张量接口 | 接收 NCHW，支持批量与任意通道数，各通道独立处理 |
| 两种处理模式 | 单阶段硬阈值；硬阈值 + Wiener 双阶段细化 |
| 图片命令行 | 8-bit 灰度、RGB、RGBA，保留 alpha，输出无损 PNG |
| 便于集成 | 同设备、同形状的连续 float32 输出，无原工程依赖 |
| 实现可检查 | 完整 Triton 源码、独立数值参考测试、可运行 benchmark |

## 定性示例

![从左到右：干净参考图、添加合成噪声的输入、真实 BM3D Triton 输出](assets/denoising.png)

**图 1.** (a) 参考图像，(b) 添加 `sigma=25/255` 高斯噪声的输入，
(c) 双阶段 BM3D 重建。矩形框标记下方展示的同一区域。输出由当前实现实际生成，
没有额外锐化、修图或学习型后处理。

<details>
<summary>展开查看相同位置的局部细节</summary>

![相同区域的局部裁剪：参考图、含噪输入与 BM3D 重建](assets/detail.png)

**图 2.** 图 1 中相同坐标区域的局部图，使用最近邻采样显示。
全部配图由 Python/Matplotlib 生成，提供 PNG、矢量 PDF 及流程图 SVG。
来源、参数与复现命令见 [assets 说明](assets/README.md)，不附 benchmark 分数。

</details>

## 快速开始

准备 Python 3.10+、NVIDIA GPU，以及相互兼容的 CUDA 版 PyTorch 和 Triton。
先按 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装相应版本，
然后在仓库根目录执行：

```bash
python -m pip install -e ".[cuda,cli]"
```

如果 PyTorch 已带有匹配的 Triton，只需安装 `.[cli]`。Linux 是主要运行环境；
Windows 的 `triton-windows` 路线属于实验性支持。请保持 Triton 与 PyTorch 版本匹配，
使用前检查 `torch.cuda.is_available()`；首次调用包含 JIT 编译开销。

### Python

```python
import torch
from bm3d_triton import denoise

# 替换为自己的浮点 NCHW 图像，值域为 [0, 1]。
image = torch.rand(1, 3, 256, 256, device="cuda")
restored = denoise(image, sigma=25 / 255, two_step=True)
```

Python 的 `sigma` 是**归一化标准差**：8-bit 单位的 25 对应 `25/255`，不是 `25`，
也不是方差。如果已有噪声方差，或者需要 `nn.Module` 形式：

```python
from bm3d_triton import BM3D, denoise_variance

restored = denoise_variance(image, variance=(25 / 255) ** 2)
restored = BM3D(sigma=25 / 255)(image)
```

输出与输入同形状、同 CUDA 设备，类型为连续 float32。输入和默认输出均截断到
`[0,1]`；`clamp=False` 可保留重建越界值。小于 8×8 的图像先复制边缘补齐，
处理后裁回原尺寸。

输入 NCHW 各维度必须非零，且使用浮点类型；`sigma`、`variance` 为有限非负 Python
标量。零噪声返回截断后的 float32 副本。张量接口会处理所有通道，包括 alpha，
只有 CLI 单独保留 alpha。`BM3D` 没有可训练参数，其设置不会写入 `state_dict()`。

### 图片命令行

```bash
bm3d-triton noisy.png restored.png --sigma 25
bm3d-triton noisy.png single-stage.png --sigma 25 --single-stage
bm3d-triton noisy.png restored.png --sigma 15 --device cuda:0 --overwrite
```

命令行使用 **8-bit 单位的 sigma**。支持 8-bit L/RGB/RGBA 输入，应用 EXIF 方向、
保留 alpha，并输出 PNG。覆盖已有文件需要 `--overwrite`。元数据和 ICC 配置不会
复制；高位深或颜色管理工作流应先进行明确转换。

## 算法如何工作

![BM3D 两阶段流程：含噪图像、硬阈值滤波、基础估计、Wiener 细化、降噪输出](assets/pipeline.svg)

**图 3.** 两阶段计算流程。实线表示信号流，虚线表示基础估计对匹配和 Wiener
增益估计的引导。

BM3D 把相似图块组成一组，利用重复结构在变换域内协同抑制噪声。第一阶段通过
硬阈值构建基础估计；第二阶段用它指导匹配和 Wiener 增益，再对原始含噪图块滤波。
所有重叠图块最终按组权重和 Kaiser 窗聚合。

本项目是**各通道独立处理的 DCT/Hadamard BM3D 变体**，并非官方参考实现或联合颜色
处理的 CBM3D。

| 参数 | 硬阈值阶段 | Wiener 阶段 |
|---|---:|---:|
| 图块大小 | 8×8 | 8×8 |
| 参考步长 / 搜索半径 | 3 / 19 像素 | 3 / 19 像素 |
| 最大组大小，含参考块 | 16 | 32 |
| 匹配阈值，8-bit 单位的平均平方距离 | 2500 | 400 |
| Kaiser 窗 beta | 2.0 | 2.0 |

组大小向下取二次幂。每个图块做正交 DCT，再沿组维度做未归一化 Hadamard 变换。
若组大小为 `m`、噪声方差为 `v`，硬阈值为 `2.7*sqrt(m*v)`；Wiener 增益根据
基础估计系数 `b` 计算：`(b*b/m)/(b*b/m + v)`。逆 Hadamard 除以 `m`，
最后按组权重和 Kaiser 窗聚合。

匹配使用 float32 距离，参考位置包含右、下边界。位置与数量缓冲以最多 4096 组分块复用，
图像级聚合内存仍随分辨率增长；第二阶段保留含噪原图和基础估计。

## 自己运行 Benchmark

仓库提供可复现的脚本和测量方法，不发布预填的耗时表、画质分数或特定硬件加速结论。

```bash
# 热启动 GPU 耗时；尺寸格式为 高x宽。
python benchmarks/benchmark.py --sizes 256x256 512x512 1080x1920 --warmup 3 --repeats 10

# 使用自己的图片评估画质，生成本地对照图。
python -m pip install -e ".[eval]"
python benchmarks/evaluate.py /path/to/image.png --sigma 10 25 50 --seeds 123 456 789
```

性能脚本在预热后同步计时，输入输出驻留 GPU；画质脚本按固定种子添加高斯噪声，
以原图计算 PSNR/SSIM。输出保存在 Git 忽略的 `benchmark-output/` 与 `quality-output/`。
计时不包含首次 JIT、CPU/GPU 传输、图片 I/O 和输出截断，包含调用内存分配与调度
（`clamp=False`）。脚本记录 wall-clock 与 CUDA event 样本；冷启动成本应单独测量。

画质评估把本地 8-bit 灰度/RGB 图片中心裁剪至最多 512×512，不缩放，按种子加噪，
输入输出截断至 `[0,1]`。SSIM 使用 11 像素高斯窗、sigma 1.5、总体协方差、
`data_range=1`，彩色通道取平均。脚本还生成局部对照图，不下载数据集。
观察指标时也应检查边缘与纹理。

两种脚本均可用 `--reference-module`、`--reference-path` 接入外部 CUDA 对照模块，
要求接口为 `forward(contiguous_HW_float32, normalized_variance, two_step) -> HW_float32`。
仓库不捆绑对照扩展。使用 `--help` 查看参数；比较时保持负载、输入、精度和噪声参数一致。

## 开发与测试

```bash
python -m pip install -e ".[dev,cli]"
ruff check .
ruff format --check .
pytest -m "not cuda"  # 接口与图片 I/O
pytest               # 有可用 GPU 时运行内核测试
python -m build
python -m twine check dist/*
```

GPU 测试将真实 Triton 内核与独立矩阵参考实现对照；CPU CI 覆盖导入、接口、图片 I/O
和打包。GPU 工作流需要配置 NVIDIA runner 后手动触发。
测试覆盖范围、协作规范和发布步骤统一列在[贡献指南](CONTRIBUTING.md)。

```text
src/bm3d_triton/   公共 API、CLI 与 Triton 内核
assets/           README 配图、算法流程图与来源说明
tools/            Matplotlib 文档配图生成脚本
benchmarks/       性能与画质评估工具
tests/            CPU 接口测试与 GPU 数值测试
.github/          CPU 与手动触发的 GPU 工作流
```

## 适用范围

算法假定输入是有限浮点图像，使用单一标量噪声水平，不自动估计噪声。
没有实现联合颜色、视频时域、HDR、空间变化噪声或学习型增强；高噪声下可能平滑细纹理。
不支持 CPU、MPS、ROCm 执行。

仅支持推理，不提供反向传播。首次调用需要 JIT 编译，批次和通道依次调度。
浮点原子累加与近似并列图块的重新匹配可能造成轻微运行间差异，不保证逐位确定性，
也未承诺 CUDA Graph 或 `torch.compile` 支持。

## 许可证与致谢

代码使用 BSD-2-Clause，BM3D 论文和 bm3d-gpu 来源见 [LICENSE](LICENSE) 与
[NOTICE](NOTICE)。文档照片的来源单独列在 [assets/README.md](assets/README.md)。
社区规则：[贡献指南](CONTRIBUTING.md) · [安全政策](SECURITY.md)。
