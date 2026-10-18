# Forecasting 3D Scanpaths in Egocentric Video：独立 PyTorch 复现

基于 CVPR 2026 Ryan 论文正文实现。**这是方法级独立实现，不是作者官方代码，也尚未复现论文的 ADT 数值。** 已下载并验证一个官方 ADT 10 秒样例和真实 DINOv2-B 权重，跑通真实数据接入。尚无完整训练数据、论文官方 split 和补充材料；单序列检查不能代替论文实验。

## 已实现

- 最后观测 RGB 相机坐标系作为 canonical frame；世界坐标与预测结果可双向变换。
- 冻结 DINOv2-B/14：前 9 帧 CLS 特征 + 最后一帧 256 个 patch 特征。
- 7 维相对头部位姿（四元数 xyzw + 平移 xyz），同一时刻跨模态共享正弦时间编码。
- 2 层视觉上下文 Transformer encoder、2 层轨迹 decoder、10 个可学习未来 query。
- 双向注意力、三维位置与持续时间 MSE、AdamW，训练 3 epochs。
- 轨迹独立、单图、去头部位姿以及 causal/partial 注意力消融开关。
- 训练／验证／检查点、EUC／DTW／离散 Fréchet、最后点／均值／线性外推基线、预测导出。
- 近似 fixation 聚类、离线 DINO 特征提取；新增真实 ADT VRS 导出器，使用 ADT 已提供的真值 gaze 深度；没有重新计算 mesh 射线交点。

## 安装与快速运行

在本项目目录运行（已创建本地 `.venv`）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/make_demo.py
OMP_NUM_THREADS=2 python -m scanpaths train --manifest data/demo/manifest.json --output runs/smoke --device cpu
python -m scanpaths evaluate --checkpoint runs/smoke/best.pt --manifest data/demo/manifest.json --output runs/smoke/metrics.json
python -m scanpaths predict --checkpoint runs/smoke/best.pt --manifest data/demo/manifest.json --output runs/smoke/predictions.npz
```

`demo` 是随机特征和人造轨迹，不能用于性能比较。`runs/smoke/best.pt` 也不是可用的 ADT 预训练权重。

若提示 `setup.py or setup.cfg not found`，通常是当前解释器的 pip 太旧，不支持 `pyproject.toml` 可编辑安装。先执行上面的构建工具升级命令，再重试。使用 `python -m pip` 可以确保安装到当前 Python 环境。构建工具已升级但隔离构建无法联网时，可执行 `python -m pip install --no-build-isolation -e '.[test]'`。

## 真实数据接入

所有距离单位为米、持续时间及时间戳为秒。相机矩阵使用 **camera-to-world**：`X_world = R @ X_camera + t`。必须使用 RGB 相机坐标系，不能直接混入 Aria device／CPF 坐标。

### 1. 将 ADT 导出为按时间对齐的记录

从 ADT 数据与 calibration 取得 RGB、对应的 RGB 相机位姿、眼动射线与场景的首个交点。每个视频独立保存一个 raw NPZ：

| 字段 | 形状 | 内容 |
|---|---|---|
| `gaze_world` | `[F,3]` | 世界坐标 gaze point，无效点为 NaN |
| `timestamps` | `[F]` | 严格递增的秒数，默认 30 Hz |
| `world_camera` | `[F,4,4]` | RGB 相机到世界的刚体变换 |
| `frame_paths` | `[F]` | Unicode 图像路径，相对于 raw NPZ 所在目录 |

已提供 `scripts/export_adt.py`，在官方 10 秒样例上验证了 VRS / GT 导出；下文接口也支持其他导出来源。不要把深度值直接当作世界坐标或用未来 RGB 作为输入。

```bash
python -m scanpaths prepare --raw data/sequence.raw.npz --output data/sequence.aligned.npz --threshold 0.05
```

`0.05 m` **仅是命令示例，并非论文参数**。当前聚类规则是相邻有效点距离不超过阈值，位置取均值，持续时间为末帧减首帧加一个采样周期，选末帧及对应相机位姿。无效点或时间缺口切断 cluster。论文正文没有给出聚类阈值、最短持续时间等，需要补充材料才能严格对齐。也可以直接提供官方 fixation-level `aligned.npz`，跳过近似聚类。

### 2. 提取真实视觉特征

下载 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)和其中的 **ViT-B/14 无 registers** 权重，并安装该仓库所需依赖。代码只从显式本地路径加载，不会自动下载。

```bash
python -m scanpaths extract --aligned data/sequence.aligned.npz --output data/sequence.features --dino-repo /path/to/dinov2 --weights /path/to/dinov2_vitb14_pretrain.pth --device cuda
```

图像转 RGB、bicubic resize 到 224×224、ImageNet 均值方差归一化。resize 方式属于本实现选择，正文只指定分辨率。提取器冻结网络并保持 eval。缓存每个 fixation 对应帧的 dense 特征，是因为每个 fixation 都可能成为训练窗口的 canonical frame；模型每次只读取最后观测帧的 dense 特征。

提取器输出包含下列同名 `.npy` 文件的目录（例如 `sequence.features`），训练通过 memory map 读取，避免把所有 patch 特征一次性装入 RAM。也兼容小数据集 NPZ 缓存，但 NPZ 会整体载入内存。每个 fixation 的 dense float32 特征约 0.75 MiB，完整数据需预留足够磁盘空间。

缓存字段为：

| 字段 | 形状 |
|---|---|
| `fixations_world` | `[N,4]`，xyz + duration |
| `world_camera` | `[N,4,4]` |
| `global_features` | `[N,768]` |
| `dense_features` | `[N,256,768]` |

### 3. 按完整视频划分 train / val / test

创建 `data/manifest.json`，路径相对于 manifest：

```json
[
  {"path": "recording_01.features", "split": "train"},
  {"path": "recording_02.features", "split": "val"},
  {"path": "recording_03.features", "split": "test"}
]
```

必须先划分视频，再生成窗口；不要将同一视频切片后随机分到不同 split。加载器拒绝重复路径。论文用 147 / 18 / 19 个视频；正文没有具体视频名单，本实现不杜撰官方 split。

训练窗口 stride=1；验证／测试 stride=20（完整 10+10 fixation 窗口不重叠，是对正文的解释）。固定前 10 点输入、后 10 点监督；所有点与观测位姿投影到第 10 点对应相机坐标系。

```bash
python -m scanpaths train --config configs/paper.json --manifest data/manifest.json --output runs/adt --device cuda
python -m scanpaths evaluate --checkpoint runs/adt/best.pt --manifest data/manifest.json --output runs/adt/test.json --device cuda
python -m scanpaths evaluate --checkpoint runs/adt/best.pt --manifest data/manifest.json --baseline last --output runs/adt/last_point.json
```

`best.pt` 根据验证集 EUC 选取，测试集不参与选择。固定随机种子，但不承诺不同硬件或 PyTorch 版本逐 bit 一致。

## 推理接口

不需要未来标签即可调用模型：

```python
import torch
from scanpaths.engine import load_model
model = load_model('runs/adt/best.pt', torch.device('cpu'))
# observed: [B,10,4]，canonical xyz + 秒
# global_features: [B,9,768]；dense_features: [B,256,768]
# poses: [B,10,7]，canonical xyzw + xyz
with torch.no_grad():
    prediction = model(observed, global_features, dense_features, poses)
# prediction: [B,10,4]，canonical xyz + 秒
```

命令行 `predict` 作用于包含完整序列的评估 manifest，但模型本身从不接收 `target` 或未来帧。NPZ 导出 `canonical`、`world` 和 `world_camera`，按 manifest 视频及窗口顺序排列。输出层遵循正文直接线性回归，持续时间可能为负；没有擅自加入 softplus，也没有在评估时裁剪。

## 与论文严格复现的差距

| 项目 | 已知／当前实现 |
|---|---|
| 观测与预测长度、图像尺寸、特征、隐藏维度 | 正文指定 10/10、224²、DINOv2-B、256 |
| Encoder / decoder 层数、主模型双向注意力 | 正文指定各 2 层 |
| 优化器、学习率、weight decay、epochs | 正文指定 AdamW、2e-4、1e-2、3 |
| Heads、FFN、dropout、batch size | 正文未指定；当前 8、1024、0.1、32 |
| MSE 权重 λ1 / λ2 | 正文未指定；当前 1 / 1 |
| 归一化顺序、激活、初始化 | 实现选择：post-norm、ReLU、矩阵 Xavier 初始化 |
| 未来 query 的时间编码 | 按正文仅给观测轨迹加固定时间编码；未来位置由独立 query 表示 |
| 单图+pose 消融 | 保留全部历史 pose；正文对该消融具体 token 数描述不足 |
| fixation 聚类及官方 split | 未取得，需进一步核实 |
| FRE | 实现离散 Fréchet；需官方实现确认其具体定义 |
| DTW | 点间欧氏代价，累计最短路径除 fixation 数；需官方脚本最终对齐 |
| EYE / TDE / 3D MultiMatch | 尚未实现，避免用相似但不同的公式冒充论文指标 |
| TPP-Gaze、2D 投影实验、全部消融 | 尚未复现 |
| 真实 DINO 权重、ADT 训练 | 已跑通一个 10 秒样例的特征提取和单批训练；未进行完整 ADT 训练 |

论文 Table 1 的 full model 报告 DTW=1.377、EUC=1.382、FRE=2.173（米）；这些是论文参考值，不是本项目测试成绩。

## 文件索引

- `scanpaths/model.py`：架构、位置编码、注意力消融、loss
- `scanpaths/geometry.py`：canonical frame 和相对位姿
- `scanpaths/data.py`：fixation 聚类、序列缓存、窗口和校验
- `scanpaths/features.py`：冻结 DINOv2 提取
- `scanpaths/engine.py`：训练、验证、检查点
- `scanpaths/metrics.py`：距离指标及启发式基线
- `scanpaths/__main__.py`：命令行
- `tests/test_core.py`：坐标系、无未来泄漏、指标、梯度和小样本过拟合测试

来源：用户提供论文 §3–4；[作者主页](https://fkryan.github.io/)（本次访问对应条目仅列 paper 链接），[DINOv2 官方代码](https://github.com/facebookresearch/dinov2)。补充材料下载未成功，因此所有未确认细节均以上表标注。


## 已接通的真实 ADT 样例（2026-10-11）

序列：`Apartment_release_golden_skeleton_seq100_10s_sample_M1292`。
下载清单为用户提供的 `data/adt_download_urls-2.json`。只下载 `main_vrs` 和 `main_groundtruth`，合计 158,074,119 字节，均通过清单 SHA1。数据位于 `data/adt/`；导出和特征位于 `data/adt_sample/`。

- 原始 RGB 300 帧，有效 gaze 299 帧；无效深度不会被替换为常数。
- 默认示例阈值 0.05 m 得到 78 个 fixation、59 个训练窗口；该阈值仍然不是已确认的论文参数。
- 缺失 gaze / 大时间间隔生成独立 segment，窗口不会跨过这些间断。
- gaze 从 CPF 转到 world，相机姿态从 device 转到 RGB，相机图像保持原始朝向。
- 目前使用 **ADT 真值 trajectory**，尚未对齐论文描述的 SLAM pose 设置；这是需要标注的实现差异。
- DINOv2 官方仓库 commit：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`，权重为官方 `dinov2_vitb14_pretrain.pth`。
- 该序列仅设置 `train`。没有把同一个短视频复制到 val/test 来生成虚假的评估成绩。

### 使用已安装的真实数据环境

当前 Python 3.9 下的 projectaria-tools 1.3.1 wheel 存在 CPython 二进制版本不匹配，且最新 DINOv2 使用 Python 3.10+ 语法。因此为真实数据流程建立了独立的 **Python 3.12 `.venv-adt`**，原 `.venv` 保留。

```bash
source .venv-adt/bin/activate
python -m pytest -q
python scripts/smoke_real.py
python -m scanpaths predict --checkpoint runs/adt_sample/smoke.pt --manifest data/adt_sample/manifest.json --split train --output runs/adt_sample/predictions.npz
```

`smoke_real.py` 在两个真实训练窗口上运行 30 个优化步骤，仅检验前向、反向和损失下降；`smoke.pt` 不是论文预训练模型。记录见 `runs/adt_sample/validation.json`。

### 从下载清单重新接入一个序列

```bash
python scripts/download_adt.py
python scripts/export_adt.py --sequence data/adt/Apartment_release_golden_skeleton_seq100_10s_sample_M1292 --output data/adt_sample
python -m scanpaths prepare --raw data/adt_sample/sequence.raw.npz --output data/adt_sample/sequence.aligned.npz --threshold 0.05
# 特征输出目录需要不存在，避免覆盖已有缓存：
OMP_NUM_THREADS=2 python -m scanpaths extract --aligned data/adt_sample/sequence.aligned.npz --output data/adt_sample/sequence.features --dino-repo third_party/dinov2 --weights weights/dinov2_vitb14_pretrain.pth --device cpu
```

下载其他单序列可给 `download_adt.py` 指定 `--sequence NAME`；正式训练仍需独立的 train / val / test 视频。`.venv-adt` 的实际依赖记录在 `requirements-adt-lock.txt`；完整复现还需数据清单与显式的 DINOv2 仓库／权重，pip 不会下载这些资源。

官方接口依据：[ADT 数据格式](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)、[眼动坐标转换](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/eye_gaze_code)。
