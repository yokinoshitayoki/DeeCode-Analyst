# DeepCode-Analyst

开源项目深度解析与技术问答系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![AI](https://img.shields.io/badge/AI-LangGraph-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

## 项目概述

DeepCode-Analyst 是一个先进的代码分析系统，结合了**大语言模型微调**和**大语言模型蒸馏**技术，能够：

- **自动代码解析**：使用 tree-sitter 解析多种编程语言
- **智能报告生成**：通过 QLoRA 微调的专业模型生成技术分析报告
- **云端部署优化**：完整的 AutoDL 部署方案，支持 GPU 加速训练

## 核心功能

### 代码分析能力
- **多语言支持**：Python、JavaScript、TypeScript、Java、C++、Go、Rust
- **结构提取**：函数、类、调用关系、导入依赖自动识别
- **语法解析**：基于 tree-sitter 的高精度语法分析

### 知识图谱
- **图谱构建**：将代码结构转换为可查询的知识图谱
- **关系挖掘**：自动发现函数调用、继承关系、模块依赖
- **多格式存储**：支持 pickle、GraphML、JSON 格式

### 智能分析
- **多智能体系统**：基于 LangGraph 的任务分解和协作
- **专业模型**：QLoRA 微调的代码分析专家模型
- **知识蒸馏**：0.5B 学生模型，快速推理，性能卓越

## 快速开始

### 环境配置

#### 本地环境
```bash
# 1. 克隆项目
git clone https://github.com/yokinoshitayoki/DeeCode-Analyst.git
cd DeepCode-Analyst

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

#### 云端环境（AutoDL 推荐）
```bash
# 快速部署到 AutoDL
bash one_click_deploy.sh

# 或使用 Windows 一键部署
cloud_data/一键部署.bat
```

### 环境变量配置

创建 `.env` 文件：
```env
# OpenAI API Key（用于多智能体系统）
OPENAI_API_KEY=your_openai_api_key_here

# 华为云镜像加速（可选）
HF_ENDPOINT=https://hf-mirror.com
```

### 基础使用

#### 1. 分析 GitHub 仓库
```bash
# 分析开源项目
python main.py analyze https://github.com/pallets/flask

# 详细输出模式
python main.py analyze https://github.com/fastapi/fastapi --verbose

# 指定输出格式
python main.py analyze https://github.com/django/django --graph-format graphml
```

#### 2. 查询知识图谱
```bash
# 查询函数节点
python main.py query ./data/graphs/flask_graph.pickle --node-type function

# 按名称模式查找
python main.py query ./data/graphs/fastapi_graph.pickle --name-pattern "router"

# 查询特定文件
python main.py query ./data/graphs/django_graph.pickle --file-path "models.py"
```

#### 3. 使用微调模型
```python
from cloud_data.scripts.local_model_loader import load_student_model

# 加载0.5B学生模型
model, tokenizer = load_student_model()

# 代码分析示例
prompt = """### 指令:
分析这段代码的性能问题

### 输入:
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

### 输出:
"""

# 生成分析报告
response = generate_analysis(model, tokenizer, prompt)
print(response)
```

## 项目架构

### 目录结构
```
DeepCode-Analyst/
├── src/                          # 核心源码模块
│   ├── data_ingestion/            # 数据获取和解析
│   │   ├── repo_cloner.py        # Git仓库克隆器
│   │   └── code_parser.py        # 代码结构解析器（tree-sitter）
│   ├── agents/                    # 多智能体系统
│   │   └── graph.py              # LangGraph智能体图
│   └── graph_builder.py          # 知识图谱构建器（NetworkX）
├── finetuning/                   # 模型微调模块
│   ├── run_finetune.py           # QLoRA微调主程序
│   ├── prepare_dataset.py        # 训练数据准备
│   ├── knowledge_distillation.py  # 知识蒸馏训练
│   ├── test_distilled_model.py   # 蒸馏模型测试
│   └── training_data.jsonl       # 训练数据集
├── cloud_data/                   # 云端部署相关
│   ├── models/                   # 预训练和微调模型
│   │   ├── teacher_7b/           # 7B教师模型（LoRA适配器）
│   │   └── student_0.5b/         # 0.5B学生模型（知识蒸馏）
│   ├── scripts/                  # 部署和测试脚本
│   └── 完整部署指南.md            # 详细部署文档
├── notebooks/                    # Jupyter开发笔记本
├── main.py                       # 主程序CLI入口
└── requirements.txt              # 项目依赖包
```

### 核心组件

#### 数据解析层
- **RepoCloner**: Git仓库自动克隆和管理
- **CodeParser**: 基于tree-sitter的多语言代码解析
  - 支持Python、JavaScript、TypeScript、Java等
  - 提取函数、类、调用关系、导入依赖

#### 知识图谱层
- **KnowledgeGraph**: 基于NetworkX的图谱构建
  - 节点类型：function、class、module、variable
  - 边类型：calls、inherits、imports、defines
  - 查询功能：类型过滤、路径查找、模式匹配

#### 智能体层
- **AgentGraph**: 基于LangGraph的多智能体系统
  - 任务分解智能体：将复杂问题分解为子任务
  - 代码分析智能体：专门分析代码结构和逻辑
  - 报告生成智能体：整合结果生成技术报告

#### 模型层
- **Teacher Model (7B)**: Qwen1.5-7B-Chat + LoRA微调
  - 专业的代码分析和报告生成能力
  - 支持复杂的代码理解和技术问答
- **Student Model (0.5B)**: 知识蒸馏的轻量化模型
  - 快速推理，适合实时分析
  - 保持高质量的代码分析能力

## 使用示例

```

### 场景1：代码质量评估
```python
# 使用微调模型分析代码质量
from cloud_data.scripts.local_model_loader import load_student_model

model, tokenizer = load_student_model()

code_snippet = """
class UserManager:
    def __init__(self):
        self.users = []
    
    def add_user(self, user):
        self.users.append(user)
        return len(self.users)
    
    def find_user(self, name):
        for user in self.users:
            if user.name == name:
                return user
        return None
"""

analysis = analyze_code_quality(model, tokenizer, code_snippet)
print(analysis)
```

### 场景2：架构理解助手
```bash
# 分析复杂项目的架构
python main.py analyze https://github.com/django/django

# 查询模型相关的类
python main.py query ./data/graphs/django_graph.pickle --node-type class --name-pattern "Model"


## 高级功能

### 模型微调

#### QLoRA 微调训练
```bash
cd finetuning

# 准备训练数据
python prepare_dataset.py --mode sample --num-samples 1000

# 开始微调训练
python run_finetune.py \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --dataset_path ./training_data.jsonl \
    --output_dir ./output \
    --num_train_epochs 3
```

#### 知识蒸馏
```bash
# 使用7B教师模型训练0.5B学生模型
python knowledge_distillation.py \
    --teacher_model ./cloud_data/models/teacher_7b \
    --student_model Qwen/Qwen1.5-0.5B-Chat \
    --output_dir ./cloud_data/models/student_0.5b
```

### 云端部署

#### AutoDL 一键部署
```bash
# 1. 上传到云端
scp -P 18812 -r ./DeepCode-Analyst root@connect.bjb1.seetacloud.com:/root/

# 2. 运行部署脚本
cd /root/DeepCode-Analyst
bash cloud_data/一键部署.bat
```


## 开发指南

### 添加新语言支持

1. **安装language包**：
```bash
pip install tree-sitter-java  # 以Java为例
```

2. **更新CodeParser**：
```python
# src/data_ingestion/code_parser.py
SUPPORTED_EXTENSIONS = {
    '.py': 'python',
    '.java': 'java',  # 新增
    # ...
}
```

3. **实现解析逻辑**：
```python
def _parse_java_file(self, tree: Tree, file_path: str) -> Tuple[List[Dict], List[Dict]]:
    # 实现Java文件解析逻辑
    pass
```



### 自定义模型训练

1. **准备专业数据集**：
```json
{
  "instruction": "分析代码的安全漏洞",
  "input": "SQL查询代码片段",
  "output": "详细的安全分析报告"
}
```

2. **配置微调参数**：
```json
{
  "model_name": "Qwen/Qwen1.5-7B-Chat",
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 2e-5,
  "num_epochs": 3
}
```

## 技术栈

### 核心依赖
```
AI/ML 框架:
├── transformers >= 4.35.0    # Hugging Face 模型库
├── peft >= 0.6.0             # 参数高效微调
├── torch >= 2.0.0            # PyTorch 深度学习
├── langchain >= 0.1.0        # LLM 应用框架
└── langgraph >= 0.0.20       # 图状智能体

代码分析:
├── tree-sitter == 0.20.4     # 语法解析器
├── tree-sitter-languages     # 多语言支持包
├── gitpython == 3.1.40       # Git 操作
└── networkx == 3.2.1         # 图处理

数据处理:
├── pandas == 2.1.4           # 数据分析
├── numpy == 1.25.2           # 数值计算
└── matplotlib == 3.8.2       # 可视化

系统工具:
├── click == 8.1.7            # CLI 工具
├── loguru == 0.7.2           # 日志系统
└── tqdm == 4.66.1            # 进度条
```

### 系统要求
- **Python**: 3.8+
- **内存**: 16GB+ RAM
- **GPU**: 8GB+ VRAM（推荐RTX 4090/V100/A100）
- **存储**: 50GB+ 可用空间
- **操作系统**: Linux/Windows/macOS

## 应用场景

### 1. 开源项目分析
- **项目评估**：快速了解项目架构和代码质量
- **技术选型**：对比不同框架的实现方式
- **学习研究**：深入理解优秀项目的设计模式

### 2. 代码审查
- **自动化审查**：批量分析代码结构和潜在问题
- **质量评估**：生成代码质量报告和改进建议
- **规范检查**：验证代码是否符合团队规范

### 3. 技术文档生成
- **API文档**：自动生成函数和类的文档
- **架构说明**：生成项目架构图和说明文档
- **使用指南**：基于代码分析生成使用教程

### 4. 教育培训
- **代码讲解**：为学习者生成详细的代码解释
- **最佳实践**：展示优秀代码的设计思路
- **问题诊断**：帮助发现和理解代码问题

## 性能指标

### 解析性能
- **Python项目**: 500+ 文件/分钟
- **准确率**: 95%+ 的代码结构识别
- **支持规模**: 10万+ 行代码项目

### 模型性能
- **7B教师模型**: 专业级代码分析，响应时间2-5秒
- **0.5B学生模型**: 快速推理，响应时间<1秒
- **知识保留率**: 85%+ 的教师模型能力

### 系统性能
- **内存占用**: 基础功能 < 2GB
- **GPU使用**: 推理时 < 4GB VRAM
- **并发支持**: 多用户同时分析

## 贡献指南

### 参与开发

1. **Fork 项目**
```bash
git clone https://github.com/yokinoshitayoki/DeeCode-Analyst.git
cd DeeCode-Analyst
```

2. **创建功能分支**
```bash
git checkout -b feature/amazing-feature
```

3. **开发和测试**
```bash
# 安装开发依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black src/ finetuning/
```

4. **提交更改**
```bash
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

5. **创建 Pull Request**

### 贡献类型
- **Bug修复**：修复已知问题
- **新功能**：添加新的分析能力
- **文档**：完善文档和示例
- **优化**：性能优化和代码重构
- **测试**：增加测试用例

### 开发规范
- 遵循 PEP 8 代码规范
- 添加完整的函数文档
- 确保测试覆盖率 > 80%
- 提交信息使用约定式提交格式


---

<div align="center">

**DeepCode-Analyst** - 让代码分析更智能！

*如果这个项目对您有帮助，请给个 Star 支持一下！*

</div>
