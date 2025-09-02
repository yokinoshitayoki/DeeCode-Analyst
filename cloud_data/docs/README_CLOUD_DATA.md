# DeepCode-Analyst 云端数据移植指南

## Cloud Data 文件夹结构

```
cloud_data/
├── models/                 # 模型文件
│   ├── teacher_7b/        # 7B教师模型 (LoRA适配器)
│   ├── student_0.5b/      # 0.5B学生模型 (LoRA蒸馏版)
│   └── base_models/       # 基础预训练模型缓存
├── knowledge/             # 教师知识文件
│   └── teacher_knowledge_lite.pkl  # 2.2GB知识文件
├── data/                  # 训练数据
│   └── training_data.jsonl        # 训练数据集
├── scripts/               # 训练和推理脚本
│   ├── lora_micro_distillation.py
│   ├── test_lora_model.py
│   └── sequential_distillation.py
├── configs/               # 配置文件
│   └── config_example.json
├── logs/                  # 训练日志
└── docs/                  # 文档
```

## 快速开始

### 1. 环境设置
```bash
# 安装依赖和设置环境
python setup_local_environment.py
```

### 2. 创建文件夹结构
```bash
# 创建cloud_data文件夹
python create_cloud_data_folder.py
```

### 3. 下载云端文件
```bash
# Linux/Mac
./cloud_data/download_models.sh

# Windows
# 运行 cloud_data/download_models_windows.bat
# 或使用WinSCP手动下载
```

### 4. 启动本地版本
```bash
# 快速启动交互界面
python quick_start.py
```

## 模型信息

### 教师模型 (7B)
- **基础模型**: Qwen1.5-7B-Chat
- **微调方法**: LoRA (r=64, α=16)
- **专业领域**: 代码分析与技术问答
- **显存需求**: 8-16GB (量化后)

### 学生模型 (0.5B)
- **基础模型**: Qwen1.5-0.5B-Chat  
- **蒸馏方法**: LoRA知识蒸馏
- **参数量**: 0.67%可训练参数
- **显存需求**: 1-2GB

### 教师知识
- **文件大小**: 2.2GB
- **样本数量**: 15个专业样本
- **知识格式**: logits + 元数据

## 使用示例

### 加载学生模型 (推荐)
```python
from local_model_loader import LocalModelLoader

# 创建加载器
loader = LocalModelLoader("./cloud_data")

# 加载轻量级学生模型
model, tokenizer = loader.load_student_model()

# 推理示例
prompt = """### 指令:
分析代码性能

### 输入:
def slow_function(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

### 输出:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

### 加载教师模型 (需要更多GPU)
```python
# 加载完整教师模型
teacher_model, tokenizer = loader.load_teacher_model()
```

## 配置选项

### local_config.json
```json
{
  "local_config": {
    "cloud_data_dir": "./cloud_data",
    "use_gpu": true,
    "use_quantization": true
  },
  "inference_config": {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查cloud_data目录是否存在
   - 确认模型文件已下载完整
   - 验证依赖包安装正确

2. **显存不足**
   - 使用量化: `use_quantization=True`
   - 选择0.5B学生模型
   - 降低batch_size

3. **下载失败**
   - 检查网络连接
   - 验证SSH密码是否过期
   - 使用WinSCP等工具手动下载

### 依赖要求
- Python 3.8+
- PyTorch 2.0+
- transformers 4.35+
- peft 0.7+
- bitsandbytes 0.41+

## 性能对比

| 模型 | 参数量 | 显存需求 | 推理速度 | 专业性 |
|------|--------|----------|----------|--------|
| 教师7B | 7B | 8-16GB | 慢 | 五星 |
| 学生0.5B | 0.5B | 1-2GB | 快 | 四星 |
| 原始0.5B | 0.5B | 1-2GB | 快 | 二星 |

## 技术亮点

- **成功在24GB GPU上完成知识蒸馏**  
- **14倍参数压缩 (7B → 0.5B)**  
- **保持专业代码分析能力**  
- **LoRA + 量化 + 蒸馏技术组合**  
- **完整的本地部署方案**

## 支持

如果遇到问题，请检查：
1. Python环境和依赖
2. 模型文件完整性  
3. GPU驱动和CUDA版本
4. 配置文件正确性

---

**恭喜！您现在拥有了一个完整的本地化DeepCode-Analyst系统！**