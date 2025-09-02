"""
DeepCode-Analyst 知识蒸馏实现
将7B参数的教师模型蒸馏到更小的学生模型
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import load_dataset
import json
from typing import Dict, List, Any
from loguru import logger

# 设置环境变量
cache_dir = "/root/autodl-tmp/huggingface_cache"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir + "/transformers"
os.environ['HF_HUB_CACHE'] = cache_dir + "/hub"

class DistillationDataset(Dataset):
    """蒸馏数据集类"""
    
    def __init__(self, teacher_outputs: List[Dict], tokenizer, max_length: int = 1024):
        self.teacher_outputs = teacher_outputs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.teacher_outputs)
    
    def __getitem__(self, idx):
        item = self.teacher_outputs[idx]
        
        # 格式化输入文本
        input_text = f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n"
        target_text = item['teacher_output']
        full_text = input_text + target_text
        
        # 分词
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding=False,
            max_length=self.max_length // 2,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': targets['input_ids'].squeeze(),
            'attention_mask': targets['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze().clone(),
            'teacher_logits': torch.tensor(item.get('teacher_logits', []), dtype=torch.float32)
        }

class DistillationTrainer(Trainer):
    """知识蒸馏训练器"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算蒸馏损失"""
        labels = inputs.get("labels")
        teacher_logits = inputs.get("teacher_logits")
        
        # 学生模型前向传播
        outputs = model(**{k: v for k, v in inputs.items() if k not in ['labels', 'teacher_logits']})
        student_logits = outputs.get('logits')
        
        # 计算标准交叉熵损失
        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # 计算知识蒸馏损失
        if teacher_logits is not None and teacher_logits.numel() > 0:
            # 确保维度匹配
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
            
            student_logits_kd = student_logits[:, :min_seq_len, :min_vocab_size]
            teacher_logits_kd = teacher_logits[:, :min_seq_len, :min_vocab_size]
            
            # 软标签损失
            student_probs = F.log_softmax(student_logits_kd / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits_kd / self.temperature, dim=-1)
            
            kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
            
            # 组合损失
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        else:
            total_loss = ce_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

class DeepCodeDistiller:
    """DeepCode-Analyst 知识蒸馏器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
    def load_teacher_model(self):
        """加载教师模型（微调后的7B模型）"""
        logger.info("加载教师模型...")
        
        base_model_id = self.config['teacher']['base_model']
        adapter_path = self.config['teacher']['adapter_path']
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # 加载LoRA适配器
        self.teacher_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.teacher_model.eval()
        
        logger.info("教师模型加载完成")
    
    def create_student_model(self):
        """创建学生模型（更小的模型）"""
        logger.info("创建学生模型...")
        
        student_config = self.config['student']
        
        if student_config['type'] == 'qwen_1.5b':
            model_id = "Qwen/Qwen1.5-1.8B-Chat"
        elif student_config['type'] == 'qwen_0.5b':
            model_id = "Qwen/Qwen1.5-0.5B-Chat"
        else:
            model_id = student_config['model_id']
        
        self.student_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        logger.info(f"学生模型创建完成: {model_id}")
    
    def generate_teacher_outputs(self, dataset_path: str):
        """生成教师模型的输出用于蒸馏"""
        logger.info("生成教师模型输出...")
        
        # 加载训练数据
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        teacher_outputs = []
        device = next(self.teacher_model.parameters()).device
        
        for i, item in enumerate(data):
            if i % 10 == 0:
                logger.info(f"处理进度: {i}/{len(data)}")
            
            # 构建输入
            input_text = f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # 获取教师模型的logits
                outputs = self.teacher_model(**inputs)
                teacher_logits = outputs.logits.cpu()
                
                # 生成教师模型的响应
                generated = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                teacher_output = self.tokenizer.decode(
                    generated[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
            
            teacher_outputs.append({
                'instruction': item['instruction'],
                'input': item['input'],
                'original_output': item['output'],
                'teacher_output': teacher_output,
                'teacher_logits': teacher_logits.tolist()
            })
        
        # 保存教师输出
        output_path = self.config['distillation']['teacher_outputs_path']
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in teacher_outputs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"教师输出已保存到: {output_path}")
        return teacher_outputs
    
    def train_student_model(self, teacher_outputs: List[Dict]):
        """训练学生模型"""
        logger.info("开始蒸馏训练...")
        
        # 创建数据集
        distill_dataset = DistillationDataset(teacher_outputs, self.tokenizer)
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config['distillation']['output_dir'],
            per_device_train_batch_size=self.config['distillation']['batch_size'],
            gradient_accumulation_steps=self.config['distillation']['gradient_accumulation'],
            num_train_epochs=self.config['distillation']['epochs'],
            learning_rate=self.config['distillation']['learning_rate'],
            logging_steps=5,
            save_steps=50,
            warmup_steps=20,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=[],
            save_total_limit=2,
        )
        
        # 创建蒸馏训练器
        trainer = DistillationTrainer(
            model=self.student_model,
            args=training_args,
            train_dataset=distill_dataset,
            data_collator=data_collator,
            temperature=self.config['distillation']['temperature'],
            alpha=self.config['distillation']['alpha']
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['distillation']['output_dir'])
        
        logger.info("蒸馏训练完成")
    
    def run_distillation(self, dataset_path: str):
        """执行完整的蒸馏流程"""
        # 1. 加载教师模型
        self.load_teacher_model()
        
        # 2. 创建学生模型
        self.create_student_model()
        
        # 3. 生成教师输出
        teacher_outputs = self.generate_teacher_outputs(dataset_path)
        
        # 4. 训练学生模型
        self.train_student_model(teacher_outputs)
        
        logger.info("知识蒸馏完成!")

def main():
    """主函数"""
    # 配置参数
    config = {
        'teacher': {
            'base_model': 'Qwen/Qwen1.5-7B-Chat',
            'adapter_path': '/root/autodl-tmp/models/deepcode-analyst-finetuned'
        },
        'student': {
            'type': 'qwen_1.5b',  # 可选: qwen_1.5b, qwen_0.5b
            'model_id': 'Qwen/Qwen1.5-1.8B-Chat'
        },
        'distillation': {
            'teacher_outputs_path': '/root/autodl-tmp/teacher_outputs.jsonl',
            'output_dir': '/root/autodl-tmp/models/deepcode-analyst-distilled',
            'batch_size': 2,
            'gradient_accumulation': 4,
            'epochs': 3,
            'learning_rate': 5e-5,
            'temperature': 4.0,
            'alpha': 0.3  # CE损失权重，KD损失权重为1-alpha
        }
    }
    
    # 创建蒸馏器
    distiller = DeepCodeDistiller(config)
    
    # 执行蒸馏
    dataset_path = './training_data.jsonl'
    distiller.run_distillation(dataset_path)

if __name__ == "__main__":
    main()
