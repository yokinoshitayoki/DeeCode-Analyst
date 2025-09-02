"""
本地模型加载器 - 在本机使用云端下载的模型
"""
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import json

class LocalModelLoader:
    def __init__(self, cloud_data_dir=None):
        if cloud_data_dir is None:
            # 默认使用项目根目录下的cloud_data
            project_root = Path(__file__).parent
            cloud_data_dir = project_root / "cloud_data"
        
        self.cloud_data_dir = Path(cloud_data_dir)
        self.models_dir = self.cloud_data_dir / "models"
        
        # 检查目录是否存在
        if not self.cloud_data_dir.exists():
            raise FileNotFoundError(f"cloud_data目录不存在: {self.cloud_data_dir}")
    
    def load_teacher_model(self, use_quantization=True):
        """加载7B教师模型 (LoRA微调版)"""
        print("🎓 加载7B教师模型...")
        
        teacher_path = self.models_dir / "teacher_7b"
        if not teacher_path.exists():
            raise FileNotFoundError(f"教师模型目录不存在: {teacher_path}")
        
        try:
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen1.5-7B-Chat",
                trust_remote_code=True
            )
            
            # 配置量化 (可选)
            quantization_config = None
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-7B-Chat",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            teacher_model = PeftModel.from_pretrained(base_model, teacher_path)
            
            print("✅ 7B教师模型加载成功")
            return teacher_model, tokenizer
            
        except Exception as e:
            print(f"❌ 教师模型加载失败: {e}")
            return None, None
    
    def load_student_model(self, use_quantization=True):
        """加载0.5B学生模型 (LoRA蒸馏版)"""
        print("👨‍🎓 加载0.5B学生模型...")
        
        student_path = self.models_dir / "student_0.5b"
        if not student_path.exists():
            raise FileNotFoundError(f"学生模型目录不存在: {student_path}")
        
        try:
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen1.5-0.5B-Chat",
                trust_remote_code=True
            )
            
            # 配置量化 (可选)
            quantization_config = None
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-0.5B-Chat",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            student_model = PeftModel.from_pretrained(base_model, student_path)
            
            print("✅ 0.5B学生模型加载成功")
            return student_model, tokenizer
            
        except Exception as e:
            print(f"❌ 学生模型加载失败: {e}")
            return None, None
    
    def load_teacher_knowledge(self):
        """加载教师知识文件"""
        print("📖 加载教师知识...")
        
        knowledge_path = self.cloud_data_dir / "knowledge" / "teacher_knowledge_lite.pkl"
        if not knowledge_path.exists():
            raise FileNotFoundError(f"教师知识文件不存在: {knowledge_path}")
        
        try:
            import pickle
            with open(knowledge_path, 'rb') as f:
                knowledge_data = pickle.load(f)
            
            print(f"✅ 教师知识加载成功: {len(knowledge_data['teacher_knowledge'])} 个样本")
            return knowledge_data
            
        except Exception as e:
            print(f"❌ 教师知识加载失败: {e}")
            return None
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            "cloud_data_dir": str(self.cloud_data_dir.absolute()),
            "models": {},
            "files": {}
        }
        
        # 检查模型目录
        teacher_path = self.models_dir / "teacher_7b"
        student_path = self.models_dir / "student_0.5b"
        knowledge_path = self.cloud_data_dir / "knowledge" / "teacher_knowledge_lite.pkl"
        data_path = self.cloud_data_dir / "data" / "training_data.jsonl"
        
        info["models"]["teacher_7b"] = {
            "path": str(teacher_path),
            "exists": teacher_path.exists(),
            "size": self._get_dir_size(teacher_path) if teacher_path.exists() else "N/A"
        }
        
        info["models"]["student_0.5b"] = {
            "path": str(student_path),
            "exists": student_path.exists(),
            "size": self._get_dir_size(student_path) if student_path.exists() else "N/A"
        }
        
        info["files"]["teacher_knowledge"] = {
            "path": str(knowledge_path),
            "exists": knowledge_path.exists(),
            "size": self._get_file_size(knowledge_path) if knowledge_path.exists() else "N/A"
        }
        
        info["files"]["training_data"] = {
            "path": str(data_path),
            "exists": data_path.exists(),
            "size": self._get_file_size(data_path) if data_path.exists() else "N/A"
        }
        
        return info
    
    def _get_dir_size(self, directory):
        """获取目录大小"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return f"{total_size / 1024**2:.1f}MB"
        except:
            return "未知"
    
    def _get_file_size(self, filepath):
        """获取文件大小"""
        try:
            size = os.path.getsize(filepath)
            if size > 1024**3:
                return f"{size / 1024**3:.1f}GB"
            else:
                return f"{size / 1024**2:.1f}MB"
        except:
            return "未知"

def demo_usage():
    """演示如何使用本地模型加载器"""
    print("🚀 本地模型加载器演示")
    print("=" * 50)
    
    try:
        # 创建加载器
        loader = LocalModelLoader()
        
        # 获取模型信息
        info = loader.get_model_info()
        print("📊 模型状态:")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        
        # 测试加载学生模型 (更轻量)
        print("\n🧪 测试加载0.5B学生模型...")
        student_model, tokenizer = loader.load_student_model()
        
        if student_model and tokenizer:
            # 测试推理
            test_prompt = """### 指令:
分析代码并提供优化建议

### 输入:
def slow_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

### 输出:
"""
            
            print("🔬 测试推理...")
            device = next(student_model.parameters()).device
            inputs = tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = student_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            print("📝 模型输出:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            print("✅ 本地模型测试成功!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("\n💡 可能的原因:")
        print("1. cloud_data文件夹不存在")
        print("2. 模型文件未下载")
        print("3. 依赖库未安装")
        print("\n🔧 解决方案:")
        print("1. 运行 python create_cloud_data_folder.py")
        print("2. 运行 python download_cloud_models.py")
        print("3. 安装依赖: pip install transformers peft bitsandbytes")

if __name__ == "__main__":
    demo_usage()
