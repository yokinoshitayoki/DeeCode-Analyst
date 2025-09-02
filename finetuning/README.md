# DeepCode-Analyst 微调模块

本目录包含了DeepCode-Analyst项目的模型微调相关代码和工具。

## 文件说明

### 核心微调脚本
- **`run_datadir_finetune.py`** - 主要的QLoRA微调脚本，支持数据盘缓存
- **`run_finetune.py`** - 原始的微调脚本（已更新，使用新的Trainer API）
- **`prepare_dataset.py`** - 数据集准备工具

### 测试脚本
- **`test_fixed_model.py`** - 修复版模型测试脚本，解决了设备不匹配问题
- **`comprehensive_test.py`** - 全面的模型测试脚本，包含多种场景测试
- **`test_finetuned_model.py`** - 基础模型测试脚本

### 数据处理工具
- **`fix_data.py`** - 训练数据清理和修复工具
- **`training_data.jsonl`** - 训练数据文件（需上传）

### 部署脚本
- **`autodl_deploy_script.sh`** - AutoDL服务器一键部署脚本
- **`config_example.json`** - 配置文件示例

### 依赖文件
- **`requirements.txt`** - 微调所需的Python包列表

## 快速开始

### 1. AutoDL服务器部署

```bash
# 上传项目到服务器
scp -r -P 18812 ./DeepCode-Analyst root@connect.bjb1.seetacloud.com:/root/autodl-tmp/

# 运行一键部署脚本
cd /root/autodl-tmp/DeepCode-Analyst/finetuning
chmod +x autodl_deploy_script.sh
./autodl_deploy_script.sh
```

### 2. 准备训练数据

```bash
# 数据格式检查
python fix_data.py --validate training_data.jsonl

# 数据修复（如需要）
python fix_data.py --fix training_data.jsonl --output training_data_fixed.jsonl

# 创建示例数据
python fix_data.py --create-sample sample_data.jsonl
```

### 3. 开始微调训练

```bash
# 激活环境
source ~/miniconda/etc/profile.d/conda.sh
conda activate deepcode

# 运行微调
python run_datadir_finetune.py
```

### 4. 测试微调结果

```bash
# 基础测试
python test_fixed_model.py

# 全面测试
python comprehensive_test.py
```

## 配置说明

### 环境变量
```bash
export HF_HOME="/root/autodl-tmp/huggingface_cache"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface_cache/transformers"
export HF_HUB_CACHE="/root/autodl-tmp/huggingface_cache/hub"
export HF_ENDPOINT="https://hf-mirror.com"
```

### 微调参数
- **基础模型**: Qwen/Qwen1.5-7B-Chat
- **量化方式**: 4-bit NF4量化
- **LoRA参数**: r=16, alpha=32, dropout=0.05
- **训练步数**: 50步（可调整）
- **学习率**: 2e-5
- **批次大小**: 1 (梯度累积4步)

### 硬件要求
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **内存**: 32GB+ 系统内存
- **存储**: 50GB+ 数据盘空间

## 训练数据格式

训练数据使用JSONL格式，每行一个JSON对象：

```json
{
  "instruction": "根据提供的代码分析结果，生成技术报告",
  "input": "{\"functions\": [\"main\", \"process\"], \"classes\": [\"DataHandler\"]}",
  "output": "# 技术分析报告\n\n## 项目概述\n该项目包含2个函数和1个类..."
}
```

### 必需字段
- `instruction`: 任务指令
- `input`: 输入数据（JSON字符串）
- `output`: 期望输出（技术报告内容）

## 常见问题

### 1. CUDA内存不足
```bash
# 减少批次大小
per_device_train_batch_size=1
gradient_accumulation_steps=8

# 启用梯度检查点
gradient_checkpointing=True
```

### 2. 磁盘空间不足
```bash
# 清理缓存
pip cache purge
conda clean --all
rm -rf /tmp/*

# 移动缓存到数据盘
export HF_HOME="/root/autodl-tmp/huggingface_cache"
```

### 3. 网络连接问题
```bash
# 使用国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
```

### 4. 设备不匹配错误
```python
# 确保输入张量在正确设备
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

## 性能优化

### 训练速度优化
- 使用`bf16`混合精度训练
- 启用`gradient_checkpointing`节省显存
- 设置`dataloader_pin_memory=True`加速数据加载
- 使用适当的`gradient_accumulation_steps`

### 推理速度优化
- 使用4-bit量化模型
- 批量推理减少开销
- 合理设置`max_new_tokens`

## 参考资料

- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [Transformers文档](https://huggingface.co/docs/transformers/)
- [PEFT文档](https://huggingface.co/docs/peft/)
- [AutoDL使用指南](https://www.autodl.com/docs/)

## 技术支持

如遇到问题，请参考：
1. **技术博客**: `../DeepCode-Analyst_开发技术博客.md`
2. **问题日志**: 查看训练输出和错误信息
3. **GPU监控**: `nvidia-smi` 或 `watch -n 1 nvidia-smi`
4. **磁盘监控**: `df -h` 和 `du -sh *`

---

*最后更新：2024年8月*
