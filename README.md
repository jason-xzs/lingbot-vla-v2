# LingBot-VLA 2：RoboTwin 场景增强与两阶段训练

基于 [Robbyant/LingBot-VLA 2](https://github.com/Robbyant/lingbot-vla-v2) 的 RoboTwin clean 数据训练方案。本分支提供场景增强、episode 内 seen 指令采样、两阶段训练、权重线性合并，以及 RoboTwin 推理与评测入口。上游模型介绍见 [README_UPSTREAM.md](README_UPSTREAM.md)。

## 方法

训练使用 RoboTwin 2.0 的 50 个任务，每任务 50 条 clean 轨迹，共 2,500 episodes。

| 阶段 | 步数 | 配置 |
| --- | --- | --- |
| 阶段一 | 0 → 50,000 | 从官方基础模型全量微调，不启用图像增强 |
| 阶段二 | 50,000 → 60,000 | 恢复阶段一的模型和优化器，启用场景增强及 seen 指令采样 |
| 合并 | 训练完成后 | `θ = 0.25 × θ_50k + 0.75 × θ_60k`，生成一份推理权重 |

阶段二的图像增强分支为 original 0.40、clutter 0.35、texture 0.10、scene 0.15。clutter 从 clean 示范首帧提取物体图块并粘贴到桌面；texture 使用程序生成的纹理；scene 组合两者。当前帧和未来帧共享增强计划，state、action、mask 和原始数据不变。指令从同一 episode 的 seen 池采样。

训练配置为 6 × A800 80GB、Muon、学习率 5e-5、全局 batch 384、每卡 micro-batch 32、梯度累积 2、action chunk 50。完整参数见 [阶段一](configs/vla/robotwin/clean_scene_stage1.yaml) 和 [阶段二](configs/vla/robotwin/clean_scene_stage2.yaml)。

## 1. 安装训练与推理环境

```bash
git clone --branch feat/robotwin-scene-augmentation https://github.com/jason-xzs/lingbot-vla-v2.git
cd lingbot-vla-v2
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
python -m pip install --no-deps numpy==2.2.6 huggingface-hub==0.34.0
```

训练／推理使用 Python 3.12、PyTorch 2.8.0、CUDA 12.8、Transformers 4.57.3、Flash Attention 2.8.3。安装脚本支持 `--flash-attn-wheel /path/to/wheel.whl`。RoboTwin 仿真使用另一个 Python 环境，见第 5 节。

以下命令均从仓库根目录执行；路径可改为自己的存储位置。

```bash
export TRAIN_PYTHON="$(command -v python)"
export LINGBOT_BASE="$PWD/pretrained/lingbot-vla-v2-6b"
export MOGE_MODEL="$PWD/pretrained/moge-2-vitb-normal/model.pt"
export QWEN3VL_PATH="$PWD/pretrained/Qwen3-VL-4B-Instruct"
export CLEAN_DATA_ROOT="$PWD/data/lerobot_clean"
export ROBOTWIN_RAW_ROOT="$PWD/data/raw_clean"
export ROBOTWIN_WORK_DIR="$PWD/runs/clean_scene"
```

## 2. 准备基础模型和数据

下载固定版本的基础模型、训练教师和 tokenizer。仅推理已有模型时，只需 Qwen 配置／tokenizer 和对应的最终权重。

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("robbyant/lingbot-vla-v2-6b",
    revision="11c703bf6a5c1f45b3b69168482da11fdbba53d7",
    local_dir=os.environ["LINGBOT_BASE"])
snapshot_download("Ruicheng/moge-2-vitb-normal",
    revision="ca5f0e07ff01d3e5a364c1d954ed12ee1814b368",
    local_dir=os.path.dirname(os.environ["MOGE_MODEL"]), allow_patterns=["model.pt"])
snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
    revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
    local_dir=os.environ["QWEN3VL_PATH"], allow_patterns=["*.json", "*.txt", "*.jinja"])
PY

python scripts/download_hf_data.py \
  --repo_id TianxingChen/RoboTwin2.0 \
  --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 \
  --allow_patterns 'dataset/*/aloha-agilex_clean_50.zip' \
  --local_dir "$ROBOTWIN_RAW_ROOT"
python scripts/prepare_robotwin_clean.py \
  --input-root "$ROBOTWIN_RAW_ROOT" --output-root "$CLEAN_DATA_ROOT" \
  --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 \
  --workers 4 --manifest assets/training_data/robotwin_clean.txt
```

转换读取 seen 指令，保持 `state[t]、action[t+1]、images[t]` 对齐。训练使用仓库内的 `assets/norm_stats/robotwin_clean.json`。

生成场景增强所需的物体库和指令池：

```bash
python scripts/prepare_robotwin_augmentation.py instructions \
  --tasks assets/training_data/robotwin_clean.txt \
  --raw-root "$ROBOTWIN_RAW_ROOT" \
  --output assets/augmentation/full50_v1 \
  --tokenizer "$QWEN3VL_PATH"
python scripts/prepare_robotwin_scene_augmentation.py \
  --tasks assets/training_data/robotwin_clean.txt \
  --output assets/augmentation/clutter_v1 --workers 4
```

数据集、模型、生成的增强资产和训练输出由上述步骤准备，不存入 Git。若增强资产放在其他目录，训练时通过 `--augmentation-root /path/to/augmentation` 指定，其中保留 `full50_v1/seen_pools.json` 和 `clutter_v1/library.npz` 两个相对路径。

## 3. 训练与权重合并

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
bash train_robotwin.sh --stage stage1 --work-dir "$ROBOTWIN_WORK_DIR"
bash train_robotwin.sh --stage stage2 --work-dir "$ROBOTWIN_WORK_DIR"
bash train_robotwin.sh --stage merge  --work-dir "$ROBOTWIN_WORK_DIR"
```

阶段二自动读取阶段一的 `checkpoints/global_step_50000`，总训练步数为 60,000。它需要含 `model/`、`optimizer/` 和 `extra_state/` 的完整训练断点；HF 导出的推理权重只用于加载模型和权重合并。

最终目录：

```text
runs/clean_scene/merged/
├── lingbotvla_cli.yaml
└── checkpoints/wise_50k60k_a0.75/hf_ckpt/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

合并逐分片在 CPU 上计算，发布的训练配方导出 FP32 权重。迁移模型时保留上述目录结构和 `lingbotvla_cli.yaml`。

可选参数：`--stage1-checkpoint` / `--stage2-checkpoint` 指定已有训练断点；`--resume-from` 恢复中断训练，配合 `--resume-dataloader` 恢复数据加载状态；`--nproc` 调整卡数并自动保持全局 batch 384；`--dry-run` 检查资源并打印命令，不启动训练。

## 4. 单独启动推理服务

```bash
export LINGBOT_CHECKPOINT="$ROBOTWIN_WORK_DIR/merged/checkpoints/wise_50k60k_a0.75/hf_ckpt"
export ROBOTWIN_NORM_STATS="$PWD/assets/norm_stats/robotwin_clean.json"

# 检查模型、配置和 tokenizer 路径，不加载模型
python robotwin_policy/linbot-vla-v2/serve.py \
  --checkpoint "$LINGBOT_CHECKPOINT" --check

# 启动 WebSocket 推理服务
CUDA_VISIBLE_DEVICES=0 python robotwin_policy/linbot-vla-v2/serve.py \
  --checkpoint "$LINGBOT_CHECKPOINT" --host 127.0.0.1 --port 8006 \
  --precision fp32 --use-length 50 --seed 42
```

默认关闭 compile，执行长度为 50，去噪 10 步，推理 seed 42。客户端实现位于 `robotwin_policy/linbot-vla-v2/deploy_policy.py`。仅运行已有权重的推理无需先完成本仓库训练。

## 5. RoboTwin 仿真评测

按 [RoboTwin 安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 准备单独的仿真环境；使用的版本为 Python 3.10、PyTorch 2.4.1 + CUDA 12.1、SAPIEN 3.0.0b1、MPLib 0.2.1。需要 NVIDIA Vulkan 和 FFmpeg。

在当前仓库根目录执行：

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
export LINGBOT_PYTHON="$TRAIN_PYTHON"

git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT"
git -C "$ROBOTWIN_ROOT" checkout c3ddfa8b97d5519efa828b075999bd0006778e5e
git -C "$ROBOTWIN_ROOT" apply "$PWD/patches/robotwin_local_changes.patch"
"$ROBOTWIN_PYTHON" -m pip install -r robotwin_policy/linbot-vla-v2/requirements-client.txt
```

该 RoboTwin 版本的安装入口为 `script/_install.sh`。仿真资产取自 `TianxingChen/RoboTwin2.0` revision `9dc9299c163db059931898a9f0852098a61155a1` 的 `objects.zip`、`embodiments.zip`、`background_texture.zip`，解压至 `RoboTwin/assets/`，再在 RoboTwin 根目录运行 `python script/update_embodiment_config_path.py`。补丁提供测试次数参数和录像处理，不改变任务成功判据、物理或随机化参数。

停止第 4 节手动启动的服务后，运行评测入口；它会自动启动和关闭推理服务：

```bash
bash eval_robotwin.sh --checkpoint "$LINGBOT_CHECKPOINT" \
  --setting both --task all --test-num 100 --seed 0 --gpu 0
```

clean 使用 seen 指令，randomized 使用 unseen 指令。每个场景 seed 包含 50 个任务 × 2 种环境 × 100 回合，两种环境共享同一权重和推理参数。`--seed 1` / `--seed 2` 可运行其他场景种子；`--setting clean` / `--setting random` 可单独评测；`--task handover_mic` 可只运行一个任务。

首次运行会将客户端复制到 RoboTwin 的 `policy/linbot-vla-v2/`。若已有不同版本的同名客户端，请使用独立 RoboTwin checkout。汇总结果和日志写入 `runs/eval/`，原生结果保存在 RoboTwin 的 `eval_result/`。`--dry-run` 只打印服务和任务命令。

## 代码位置与测试

- `lingbotvla/data/vla_data/augmentation.py`、`scene_augmentation.py`：增强及指令采样。
- `scripts/prepare_robotwin_*.py`：clean 数据转换、增强资产生成。
- `tools/train_robotwin.py`、`merge_checkpoints.py`：训练入口和权重合并。
- `tools/eval_robotwin.py`、`robotwin_policy/`：推理服务和评测客户端。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_*augmentation.py' -v
```

默认执行 CPU 单元测试；需要真实数据或 GPU 教师的集成测试需显式启用对应环境变量。代码沿用上游 [Apache-2.0 许可](LICENSE)，各预训练模型和数据遵循其原始许可。
