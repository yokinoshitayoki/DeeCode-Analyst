"""
真正的知识蒸馏实现
序列化方式：先提取教师知识，再训练学生模型
显存需求：约10GB（适合RTX 4090）
"""
import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import Dataset
import json
import time
from datetime import datetime
import numpy as np
import pickle
from tqdm import tqdm

print("🎓 DeepCode-Analyst 真正知识蒸馏")
print("=" * 50)

# 环境配置
cache_dir = "/root/autodl-tmp/huggingface_cache"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir + "/transformers"
os.environ['HF_HUB_CACHE'] = cache_dir + "/hub"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class KnowledgeDistillationTrainer(Trainer):
    """真正的知识蒸馏训练器"""
    
    def __init__(self, teacher_outputs, temperature=4.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.teacher_outputs = teacher_outputs  # 预计算的教师输出
        self.temperature = temperature
        self.alpha = alpha  # KD损失权重
        self.sample_idx = 0
        
    def get_teacher_output(self, batch_size):
        """获取对应的教师输出"""
        teacher_logits = []
        for i in range(batch_size):
            if self.sample_idx < len(self.teacher_outputs):
                teacher_logits.append(self.teacher_outputs[self.sample_idx]['logits'])
                self.sample_idx += 1
            else:
                # 循环使用
                idx = self.sample_idx % len(self.teacher_outputs)
                teacher_logits.append(self.teacher_outputs[idx]['logits'])
                self.sample_idx += 1
        
        return torch.stack(teacher_logits)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算知识蒸馏损失"""
        labels = inputs.get("labels")
        batch_size = inputs['input_ids'].size(0)
        
        # 学生模型前向传播
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 获取预计算的教师logits
        teacher_logits = self.get_teacher_output(batch_size).to(student_logits.device)
        
        # 1. 标准交叉熵损失（硬标签）
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        hard_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 2. 知识蒸馏损失（软标签）
        # 确保维度匹配
        min_seq_len = min(shift_logits.size(1), teacher_logits.size(1) - 1)
        min_vocab_size = min(shift_logits.size(-1), teacher_logits.size(-1))
        
        if min_seq_len > 0 and min_vocab_size > 0:
            # 截取匹配的部分
            student_logits_kd = shift_logits[:, :min_seq_len, :min_vocab_size]
            teacher_logits_kd = teacher_logits[:, :min_seq_len, :min_vocab_size]
            
            # 计算软标签损失
            student_log_probs = F.log_softmax(student_logits_kd / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits_kd / self.temperature, dim=-1)
            
            # KL散度损失
            soft_loss = F.kl_div(
                student_log_probs, 
                teacher_probs, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
        else:
            soft_loss = torch.tensor(0.0, device=student_logits.device)
        
        # 3. 组合损失
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        # 日志记录
        if hasattr(self, 'state') and self.state.global_step % 5 == 0:
            print(f"Step {self.state.global_step}: Hard={hard_loss:.4f}, Soft={soft_loss:.4f}, Total={total_loss:.4f}")
        
        return (total_loss, student_outputs) if return_outputs else total_loss

def extract_teacher_knowledge():
    """第一阶段：提取教师模型知识"""
    print("\n🎓 阶段1: 提取教师模型知识")
    print("-" * 40)
    
    # 加载教师模型
    print("📥 加载教师模型...")
    base_model_id = "Qwen/Qwen1.5-7B-Chat"
    adapter_path = "/root/autodl-tmp/models/deepcode-analyst-finetuned"
    
    # 检查教师模型
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"教师模型不存在: {adapter_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("🔧 加载7B基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # 加载LoRA适配器
    print("🎯 加载微调适配器...")
    teacher_model = PeftModel.from_pretrained(base_model, adapter_path)
    teacher_model.eval()
    
    print("✅ 教师模型加载完成")
    
    # 显存状态
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"📊 当前显存使用: {allocated:.2f}GB")
    
    # 加载训练数据
    print("\n📚 处理训练数据...")
    with open('./training_data.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"📋 总样本数: {len(data)}")
    
    # 提取教师知识
    teacher_outputs = []
    device = next(teacher_model.parameters()).device
    
    print("\n🧠 提取教师模型知识...")
    for i, item in enumerate(tqdm(data, desc="处理样本")):
        # 格式化输入
        text = f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n{item['output']}"
        
        # 分词
        inputs = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=1024,
            return_tensors="pt"
        ).to(device)
        
        # 获取教师模型输出
        with torch.no_grad():
            outputs = teacher_model(**inputs)
            logits = outputs.logits.cpu().detach()
            
            # 同时生成教师模型的实际响应用于对比
            generated = teacher_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            teacher_response = tokenizer.decode(
                generated[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
        
        teacher_outputs.append({
            'sample_id': i,
            'instruction': item['instruction'],
            'input': item['input'],
            'expected_output': item['output'],
            'teacher_response': teacher_response,
            'logits': logits.squeeze(0),  # 移除batch维度
            'input_ids': inputs['input_ids'].cpu().squeeze(0)
        })
        
        # 清理显存
        del outputs, inputs, generated
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    # 保存教师知识
    knowledge_path = "/root/autodl-tmp/teacher_knowledge.pkl"
    print(f"\n💾 保存教师知识到: {knowledge_path}")
    
    with open(knowledge_path, 'wb') as f:
        pickle.dump({
            'teacher_outputs': teacher_outputs,
            'tokenizer_config': {
                'model_id': base_model_id,
                'vocab_size': tokenizer.vocab_size,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id
            },
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(teacher_outputs),
                'teacher_model': adapter_path
            }
        }, f)
    
    # 释放教师模型显存
    print("🗑️ 释放教师模型显存...")
    del teacher_model, base_model
    torch.cuda.empty_cache()
    
    print(f"✅ 知识提取完成! 处理了 {len(teacher_outputs)} 个样本")
    print(f"📁 知识文件大小: {os.path.getsize(knowledge_path) / 1024**2:.1f}MB")
    
    return knowledge_path, tokenizer

def train_student_model(knowledge_path, tokenizer):
    """第二阶段：训练学生模型"""
    print("\n👨‍🎓 阶段2: 训练学生模型")
    print("-" * 40)
    
    # 加载教师知识
    print("📖 加载教师知识...")
    with open(knowledge_path, 'rb') as f:
        knowledge_data = pickle.load(f)
    
    teacher_outputs = knowledge_data['teacher_outputs']
    print(f"📚 加载了 {len(teacher_outputs)} 个教师样本")
    
    # 加载学生模型
    print("👨‍🎓 加载学生模型...")
    student_model_id = "Qwen/Qwen1.5-1.8B-Chat"
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    print("✅ 学生模型加载完成")
    
    # 显存检查
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"📊 当前显存使用: {allocated:.2f}GB")
    
    # 准备训练数据
    print("\n📋 准备训练数据...")
    formatted_data = []
    for item in teacher_outputs:
        text = f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n{item['expected_output']}"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=1024
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="分词处理"
    )
    
    print(f"✅ 数据集准备完成: {len(tokenized_dataset)} 个样本")
    
    # 配置训练
    output_dir = "/root/autodl-tmp/models/deepcode-analyst-kd"
    os.makedirs(output_dir, exist_ok=True)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 知识蒸馏训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,  # 蒸馏通常需要更多轮次
        learning_rate=5e-5,
        logging_steps=5,
        save_steps=20,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],
        save_total_limit=2,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_safetensors=True,
    )
    
    print(f"🎯 训练配置:")
    print(f"   轮次: {training_args.num_train_epochs}")
    print(f"   批次大小: {training_args.per_device_train_batch_size}")
    print(f"   梯度累积: {training_args.gradient_accumulation_steps}")
    print(f"   学习率: {training_args.learning_rate}")
    print(f"   输出目录: {output_dir}")
    
    # 创建知识蒸馏训练器
    trainer = KnowledgeDistillationTrainer(
        teacher_outputs=teacher_outputs,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        temperature=4.0,  # 温度参数
        alpha=0.7       # 70% KD损失 + 30% 硬标签损失
    )
    
    # 开始知识蒸馏训练
    print("\n🔥 开始知识蒸馏训练...")
    start_time = datetime.now()
    
    train_result = trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 保存蒸馏模型
    print("\n💾 保存蒸馏模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 保存蒸馏信息
    distillation_info = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": str(duration),
        "teacher_model": "Qwen1.5-7B + LoRA微调",
        "student_model": "Qwen1.5-1.8B",
        "training_samples": len(teacher_outputs),
        "final_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
        "distillation_config": {
            "temperature": 4.0,
            "alpha": 0.7,
            "epochs": training_args.num_train_epochs
        }
    }
    
    with open(f"{output_dir}/distillation_info.json", 'w', encoding='utf-8') as f:
        json.dump(distillation_info, f, ensure_ascii=False, indent=2)
    
    print("✅ 知识蒸馏训练完成!")
    print(f"⏰ 训练耗时: {duration}")
    print(f"📊 最终损失: {train_result.training_loss:.4f}")
    print(f"🔄 训练步数: {train_result.global_step}")
    print(f"📁 模型保存: {output_dir}")
    
    return output_dir

def test_distilled_model(model_path):
    """第三阶段：测试蒸馏效果"""
    print(f"\n🧪 阶段3: 测试蒸馏模型")
    print("-" * 40)
    
    # 加载蒸馏模型
    print("📥 加载蒸馏模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ 蒸馏模型加载完成")
    
    # 测试用例
    test_cases = [
        {
            "name": "项目架构分析",
            "prompt": """### 指令:
根据提供的代码库结构分析，生成一份关于项目整体架构的技术报告

### 输入:
{"project_name": "Analytics-Platform", "components": ["data-ingestion", "processing", "visualization"], "key_files": ["pipeline.py", "analyzer.py", "dashboard.js"]}

### 输出:
"""
        },
        {
            "name": "函数性能分析", 
            "prompt": """### 指令:
请深入分析指定Python函数的内部实现逻辑、参数和性能特征

### 输入:
{"function_name": "optimize_query", "parameters": ["sql_query", "cache_enabled", "timeout"], "complexity": "high", "performance_critical": true}

### 输出:
"""
        },
        {
            "name": "代码优化建议",
            "prompt": """### 指令:
分析给定的代码片段，识别其中可能存在的性能瓶颈，并提供详细的优化建议

### 输入:
{"code_snippet": "def process_large_dataset(data):\\n    results = []\\n    for item in data:\\n        processed = expensive_transform(item)\\n        results.append(processed)\\n    return results", "context": "处理百万级数据时性能较慢"}

### 输出:
"""
        }
    ]
    
    print(f"🎯 运行 {len(test_cases)} 个测试用例...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}: {test_case['name']}")
        print("-" * 30)
        
        device = next(model.parameters()).device
        inputs = tokenizer(test_case['prompt'], return_tensors="pt").to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"⏱️  推理时间: {inference_time:.3f}秒")
        print(f"📄 响应长度: {len(response)}字符")
        print(f"💬 生成内容:")
        print(response[:200] + "..." if len(response) > 200 else response)
        
        # 简单质量评估
        quality_indicators = 0
        if len(response) > 50:
            quality_indicators += 1
        if any(word in response for word in ['分析', '报告', '建议', '优化', '性能']):
            quality_indicators += 1
        if any(marker in response for marker in ['###', '**', '1.', '2.', '-', '*']):
            quality_indicators += 1
        
        quality_score = quality_indicators / 3
        print(f"📊 质量评分: {quality_score:.1%}")
        print("✅ 测试通过" if quality_score >= 0.6 else "⚠️ 需要改进")

def main():
    """主函数：执行完整的知识蒸馏流程"""
    print("🚀 开始真正的知识蒸馏流程")
    
    # 检查环境
    if not torch.cuda.is_available():
        print("❌ 需要GPU支持")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 显存: {gpu_memory:.1f}GB")
    
    if gpu_memory < 20:
        print("⚠️ 建议使用20GB+显存的GPU以获得最佳效果")
    
    try:
        # 第一阶段：提取教师知识
        knowledge_path, tokenizer = extract_teacher_knowledge()
        
        # 第二阶段：训练学生模型
        model_path = train_student_model(knowledge_path, tokenizer)
        
        # 第三阶段：测试蒸馏效果
        test_distilled_model(model_path)
        
        print(f"\n🎉 知识蒸馏完成!")
        print(f"📁 蒸馏模型: {model_path}")
        print(f"📁 教师知识: {knowledge_path}")
        
        print(f"\n📋 后续操作:")
        print(f"1. 查看模型: ls -la {model_path}")
        print(f"2. 打包下载: tar -czf distilled_model.tar.gz {model_path}")
        print(f"3. 详细测试: python test_distilled_model.py")
        
    except Exception as e:
        print(f"❌ 知识蒸馏失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
