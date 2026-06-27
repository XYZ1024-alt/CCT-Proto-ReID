# 完整训练流程设计

## 目标

为当前 T2C-CLIP 项目补齐可在服务器上真实启动的训练流程。训练入口继续使用 `scripts/train.py`，新增一个推荐的 job builder：`t2c_clip.jobs.clip_reid:build_training_job`。该 builder 负责加载真实 CLIP、读取 Market-1501 或 MSMT17 图片、构建 DataLoader、执行 Stage-2 训练、周期性验证 mAP，并复用现有 checkpoint 与 MLflow 机制。

## 推荐方案

使用 `transformers.CLIPModel` 作为 CLIP 后端。当前环境已有 `transformers`，而 `open_clip` 和 `clip` 之前没有安装，因此 Transformers 是最少依赖、最容易在 WSL `reid` 环境和服务器环境复现的方案。

不引入 mock encoder、mock dataset 或假 mAP。缺少依赖、权重、数据集文件或 manifest 时直接抛出明确错误，让失败暴露在训练启动阶段或对应 batch 阶段。

## 范围

本次补齐 Stage-2 完整训练流程：

- 支持 `market1501` 和 `msmt17`。
- 支持真实图片读取和 CLIP 图像预处理。
- 支持 Transformers CLIP 图像编码器和文本投影维度。
- 使用现有 `PromptBank`、`T2CClipModel`、`stage2_loss_breakdown`、`TFCCenterBank`。
- 验证时抽取 query/gallery retrieval feature，调用现有 `evaluate_reid`。
- 训练循环仍由 `scripts/train.py` 和 `run_training_loop` 负责。

本次不实现两阶段调度器，不自动下载数据集，不做 rerank，不做 Text-to-Image 评估，不使用测试身份 prompt。

## CLI 入口

用户启动训练时使用：

```bash
python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --clip-model-name openai/clip-vit-base-patch16 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --batch-size 64 \
  --num-workers 4 \
  --lr 0.0001 \
  --device cuda \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name msmt17-stage2
```

`scripts/train.py` 将增加 project-specific 参数，但仍保持 job-builder 注入模式。参数解析只负责接收配置，不创建训练对象。

## 组件设计

### `t2c_clip/datasets.py`

负责把现有 `ReIDSample` 转成 PyTorch Dataset。

- `ReIDImageDataset` 返回图片 tensor、重映射后的 person ID、重映射后的 camera ID、原始 person ID、原始 camera ID。
- 训练 split 必须重映射 person ID 到 `[0, num_train_ids)`，满足 classifier、PromptBank 和 TFC 的索引要求。
- camera ID 也重映射到 `[0, num_cameras)`，并在 train/query/gallery 之间共享同一套映射。
- 图片读取使用 Pillow。图片缺失或损坏时不吞异常。

### `t2c_clip/transforms.py`

负责从 `CLIPProcessor` 提供的 image processor 构建单图 transform。

- 输入 Pillow image。
- 输出 `pixel_values[0]`，形状与 CLIP 期望一致。
- 不自己手写归一化参数，避免和具体 CLIP checkpoint 不一致。

### `t2c_clip/clip_backbone.py`

负责把 Transformers CLIP 包装成当前模型需要的接口。

- `TransformersCLIPImageEncoder` 调用 `CLIPModel.get_image_features(pixel_values=images)`。
- `PromptTextEncoder` 接收 learnable prompt embedding，做 mean pooling 后接线性投影到 CLIP projection dim。
- 这个 text encoder 不伪装成原始 tokenizer 文本输入，因为当前 `PromptBank` 产出的是 learnable embedding tensor，不是 token IDs。

### `t2c_clip/jobs/clip_reid.py`

负责实现 `build_training_job(args)`。

- 加载 dataset samples。
- 建立 ID/camera 映射。
- 建立 DataLoader。
- 加载 CLIPModel 和 CLIPProcessor。
- 创建 PromptBank、T2CClipModel、classifier、TFCCenterBank。
- 创建 optimizer。
- 返回 `TrainingJob(model, optimizer, train_one_epoch, validate)`。

训练时每个 batch：

1. 移动 batch 到 device。
2. 先用当前 retrieval feature 更新 TFC centers。
3. 计算 `stage2_loss_breakdown`。
4. 反向传播并 optimizer step。

验证时：

1. `model.eval()`。
2. 对 query/gallery 分别抽取 retrieval feature。
3. 汇总原始 person ID 和原始 camera ID。
4. 调用 `evaluate_reid`。
5. 恢复训练模式由下一轮训练函数负责。

## 错误处理

失败必须清楚暴露：

- `transformers` 缺失时抛出 ImportError。
- `--dataset` 不支持时抛出 ValueError。
- `--data-root` 不存在时抛出 FileNotFoundError。
- train/query/gallery 为空时抛出 ValueError。
- CLIP 权重加载失败时保留原始异常。
- TFC center 未初始化等已有错误继续向外抛出。

不添加静默 fallback，不添加假成功路径，不自动切换到 CPU，除非用户显式传入 `--device cpu`。

## 测试策略

采用 TDD。

- Dataset 测试使用临时小图片，验证 Market/MSMT samples 转 batch 的字段、ID 映射和 camera 映射。
- Transform 测试用假的 image processor，验证单图 transform 返回 `pixel_values[0]`。
- CLIP wrapper 测试用小型 fake CLIP module，验证 image encoder 调用 `get_image_features`，text encoder 输出维度正确。
- Job builder 测试通过依赖注入 fake CLIP loader 和小图片数据集，不下载真实权重，不造假训练结果。
- CLI 参数测试验证 `scripts/train.py` 接收完整训练参数并传给 builder。
- 最后运行完整 unittest 和 compileall。

## 服务器运行流程

1. 准备 conda 环境，安装 `torch`、`torchvision`、`transformers`、`mlflow`、`tqdm`、`numpy`、`pillow`。
2. 拉取项目代码。
3. 放置 `MSMT17_V1/` 或 `Market-1501/` 到服务器路径。
4. 初始化 MLflow SQLite。
5. 启动 MLflow UI 到 `6006`。
6. 使用 `scripts/train.py --job-builder t2c_clip.jobs.clip_reid:build_training_job` 启动训练。

## 验收标准

- `scripts/train.py` 能用推荐 job builder 启动真实训练配置。
- 验证间隔仍可通过 `--validation-interval` 设置。
- `best.pth` 和 `last.pth` 仍由已有训练循环保存。
- 单元测试覆盖新增训练流程组件。
- WSL `reid` 环境下 unittest 和 compileall 通过。
