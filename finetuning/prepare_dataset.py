"""
Dataset Preparation for QLoRA Fine-tuning
为综合报告生成智能体准备训练数据集
"""
import json
import jsonlines
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from loguru import logger


class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(self):
        """初始化数据集准备器"""
        self.templates = self._get_instruction_templates()
        logger.info("DatasetPreparer 初始化完成")
    
    def _get_instruction_templates(self) -> List[str]:
        """获取指令模板"""
        return [
            "根据以下对代码库的分析结果，综合撰写一份详尽的技术说明。",
            "请基于代码分析数据，生成一份专业的技术报告。",
            "分析以下代码结构信息，并撰写技术文档。",
            "根据提供的代码分析结果，创建一份comprehensive technical analysis。",
            "基于代码库分析数据，生成结构化的技术说明文档。"
        ]
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        创建示例数据集（用于演示）
        
        Args:
            output_path: 输出路径
            num_samples: 样本数量
            
        Returns:
            创建结果信息
        """
        logger.info(f"创建示例数据集: {num_samples} 个样本")
        
        try:
            samples = []
            
            for i in range(num_samples):
                # 生成示例分析结果
                sample_analysis = self._generate_sample_analysis(i)
                
                # 生成示例报告
                sample_report = self._generate_sample_report(sample_analysis, i)
                
                # 创建训练样本
                instruction = self.templates[i % len(self.templates)]
                
                sample = {
                    "instruction": instruction,
                    "input": json.dumps(sample_analysis, ensure_ascii=False, indent=2),
                    "output": sample_report
                }
                
                samples.append(sample)
            
            # 保存数据集
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with jsonlines.open(output_path, 'w') as writer:
                for sample in samples:
                    writer.write(sample)
            
            # 创建 Hugging Face 数据集格式
            dataset = Dataset.from_list(samples)
            
            # 分割数据集
            train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
            dataset_dict = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
            
            # 保存为 Hugging Face 格式
            hf_output_path = output_path.parent / "dataset_hf"
            dataset_dict.save_to_disk(str(hf_output_path))
            
            result = {
                "success": True,
                "output_path": str(output_path),
                "hf_dataset_path": str(hf_output_path),
                "total_samples": len(samples),
                "train_samples": len(train_test_split['train']),
                "test_samples": len(train_test_split['test']),
                "message": f"成功创建 {len(samples)} 个训练样本"
            }
            
            logger.success(f"数据集创建完成: {output_path}")
            return result
            
        except Exception as e:
            error_msg = f"创建数据集失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "数据集创建失败"
            }
    
    def _generate_sample_analysis(self, index: int) -> Dict[str, Any]:
        """生成示例分析结果"""
        
        # 示例代码分析数据
        sample_projects = [
            {
                "project_type": "Web应用",
                "functions": ["login", "register", "dashboard", "api_handler"],
                "classes": ["User", "Database", "APIClient"],
                "complexity": "中等",
                "patterns": ["MVC", "单例模式"]
            },
            {
                "project_type": "数据分析工具", 
                "functions": ["load_data", "process_data", "visualize", "export_results"],
                "classes": ["DataProcessor", "Visualizer", "Exporter"],
                "complexity": "高",
                "patterns": ["工厂模式", "观察者模式"]
            },
            {
                "project_type": "机器学习库",
                "functions": ["train_model", "predict", "evaluate", "save_model"],
                "classes": ["Model", "Trainer", "Evaluator"],
                "complexity": "高",
                "patterns": ["策略模式", "模板方法"]
            }
        ]
        
        project = sample_projects[index % len(sample_projects)]
        
        return {
            "sub_tasks": [
                {
                    "task_id": f"task_{i+1}",
                    "description": f"分析{project['project_type']}的{desc}",
                    "results": {
                        "nodes": [
                            {"name": func, "type": "function", "file_path": f"/src/{func}.py"}
                            for func in project["functions"]
                        ] + [
                            {"name": cls, "type": "class", "file_path": f"/src/{cls.lower()}.py"} 
                            for cls in project["classes"]
                        ],
                        "connections": [
                            {"source": "main", "target": func, "type": "calls"}
                            for func in project["functions"][:2]
                        ],
                        "statistics": {
                            "total_nodes": len(project["functions"]) + len(project["classes"]),
                            "total_edges": len(project["functions"]),
                            "complexity_score": 0.7 if project["complexity"] == "高" else 0.5
                        }
                    }
                }
                for i, desc in enumerate(["核心功能", "数据流", "架构模式"])
            ],
            "repo_info": {
                "name": f"sample-project-{index+1}",
                "language": "Python",
                "size": "中型",
                "domain": project["project_type"]
            },
            "overall_statistics": {
                "total_files": 15 + (index % 10),
                "total_functions": len(project["functions"]),
                "total_classes": len(project["classes"]),
                "design_patterns": project["patterns"]
            }
        }
    
    def _generate_sample_report(self, analysis: Dict[str, Any], index: int) -> str:
        """生成示例技术报告"""
        
        repo_info = analysis["repo_info"]
        stats = analysis["overall_statistics"]
        
        report = f"""# {repo_info['name']} 技术分析报告

## 执行摘要

本报告对 {repo_info['name']} 项目进行了全面的代码结构分析。该项目是一个{repo_info['domain']}，使用{repo_info['language']}开发，代码规模为{repo_info['size']}。

### 主要发现
- 项目包含 {stats['total_files']} 个源文件
- 定义了 {stats['total_functions']} 个函数和 {stats['total_classes']} 个类
- 采用了 {', '.join(stats['design_patterns'])} 等设计模式

## 详细分析

### 代码结构概览
通过对项目的静态分析，我们识别出了以下核心组件：

#### 功能模块
"""
        
        # 添加子任务分析
        for i, task in enumerate(analysis["sub_tasks"]):
            report += f"\n##### {task['task_id']}: {task['description']}\n"
            task_results = task["results"]
            
            report += f"- 发现 {len(task_results['nodes'])} 个代码元素\n"
            report += f"- 识别出 {len(task_results['connections'])} 个调用关系\n"
            
            if task_results["nodes"]:
                functions = [n for n in task_results["nodes"] if n["type"] == "function"]
                classes = [n for n in task_results["nodes"] if n["type"] == "class"]
                
                if functions:
                    report += f"- 关键函数: {', '.join([f['name'] for f in functions[:3]])}\n"
                if classes:
                    report += f"- 核心类: {', '.join([c['name'] for c in classes[:3]])}\n"
        
        report += f"""
### 架构洞察

#### 设计模式
项目采用了以下设计模式：
{chr(10).join([f'- **{pattern}**: 提高了代码的{pattern}特性' for pattern in stats['design_patterns']])}

#### 代码复杂度
根据分析结果，项目的整体复杂度为{repo_info['size']}，具有良好的结构化设计。

### 技术建议

1. **代码质量**: 当前代码结构清晰，建议继续保持良好的命名规范。
2. **模块化**: 建议进一步细化模块划分，提高代码的可维护性。
3. **文档**: 建议为核心函数和类添加详细的文档字符串。

### 结论

{repo_info['name']} 项目展现了良好的代码组织结构和设计理念。通过采用{', '.join(stats['design_patterns'])}等设计模式，项目具备了良好的可扩展性和可维护性。建议在后续开发中继续遵循当前的设计原则，并适当优化模块间的耦合关系。

---
*本报告基于静态代码分析生成，分析时间: {index+1} 分钟*
"""
        
        return report.strip()
    
    def load_real_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载真实的分析结果数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据列表
        """
        try:
            data_path = Path(data_path)
            
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
            samples = []
            
            if data_path.suffix == '.jsonl':
                with jsonlines.open(data_path) as reader:
                    for item in reader:
                        samples.append(item)
            elif data_path.suffix == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    else:
                        samples = [data]
            else:
                raise ValueError(f"不支持的文件格式: {data_path.suffix}")
            
            logger.info(f"加载了 {len(samples)} 个真实数据样本")
            return samples
            
        except Exception as e:
            logger.error(f"加载真实数据失败: {str(e)}")
            return []
    
    def convert_to_training_format(self, raw_data: List[Dict[str, Any]], 
                                 output_path: str) -> Dict[str, Any]:
        """
        将原始数据转换为训练格式
        
        Args:
            raw_data: 原始数据
            output_path: 输出路径
            
        Returns:
            转换结果
        """
        try:
            training_samples = []
            
            for i, data in enumerate(raw_data):
                # 提取输入数据
                input_data = {
                    "analysis_results": data.get("analysis_results", []),
                    "statistics": data.get("statistics", {}),
                    "repo_info": data.get("repo_info", {})
                }
                
                # 检查是否有人工标注的报告
                if "human_report" in data:
                    output_text = data["human_report"]
                elif "final_report" in data:
                    output_text = data["final_report"]
                else:
                    # 跳过没有输出的数据
                    continue
                
                # 创建训练样本
                instruction = self.templates[i % len(self.templates)]
                
                sample = {
                    "instruction": instruction,
                    "input": json.dumps(input_data, ensure_ascii=False, indent=2),
                    "output": output_text
                }
                
                training_samples.append(sample)
            
            # 保存转换后的数据
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with jsonlines.open(output_path, 'w') as writer:
                for sample in training_samples:
                    writer.write(sample)
            
            result = {
                "success": True,
                "output_path": str(output_path),
                "total_samples": len(training_samples),
                "message": f"成功转换 {len(training_samples)} 个训练样本"
            }
            
            logger.success(f"数据转换完成: {output_path}")
            return result
            
        except Exception as e:
            error_msg = f"数据转换失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "数据转换失败"
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="准备QLoRA微调数据集")
    parser.add_argument("--mode", choices=["sample", "real"], default="sample",
                       help="数据模式：sample(示例数据) 或 real(真实数据)")
    parser.add_argument("--input-path", help="输入数据路径（real模式）")
    parser.add_argument("--output-path", default="./training_data.jsonl",
                       help="输出数据路径")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="示例样本数量（sample模式）")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer()
    
    if args.mode == "sample":
        result = preparer.create_sample_dataset(args.output_path, args.num_samples)
    else:
        if not args.input_path:
            print("❌ real模式需要提供 --input-path 参数")
            return
        
        raw_data = preparer.load_real_data(args.input_path)
        if not raw_data:
            print("❌ 加载真实数据失败")
            return
        
        result = preparer.convert_to_training_format(raw_data, args.output_path)
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"输出路径: {result['output_path']}")
        print(f"样本数量: {result['total_samples']}")
    else:
        print(f"❌ {result['message']}")
        if "error" in result:
            print(f"错误: {result['error']}")


if __name__ == "__main__":
    main()
