"""
QLoRA Fine-tuning Script for Code Analysis Report Generation
为综合报告生成智能体进行QLoRA微调
"""
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import wandb
from loguru import logger


class QLoRAFineTuner:
    """QLoRA微调器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化微调器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        if self.device.type == "cpu":
            logger.warning("未检测到GPU，将使用CPU训练（速度较慢）")
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        model_id = self.config["model_id"]
        logger.info(f"加载模型: {model_id}")
        
        # BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 准备模型进行训练
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.success("模型和分词器加载完成")
    
    def setup_lora_config(self):
        """设置LoRA配置"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=self.config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.success("LoRA配置完成")
    
    def load_and_prepare_dataset(self):
        """加载和准备数据集"""
        dataset_path = self.config["dataset_path"]
        logger.info(f"加载数据集: {dataset_path}")
        
        try:
            # 尝试加载不同格式的数据集
            if Path(dataset_path).is_dir():
                # Hugging Face 数据集格式
                dataset = load_dataset(dataset_path)
            else:
                # JSONL 文件
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            
            # 如果数据集没有分割，手动分割
            if isinstance(dataset, Dataset):
                dataset = dataset.train_test_split(
                    test_size=self.config.get("test_size", 0.1),
                    seed=42
                )
            
            # 格式化数据
            def format_instruction(example):
                """格式化指令数据"""
                if "instruction" in example and "input" in example and "output" in example:
                    # 标准格式
                    text = f"### 指令:\n{example['instruction']}\n\n### 输入:\n{example['input']}\n\n### 输出:\n{example['output']}"
                else:
                    # 如果格式不标准，尝试其他字段
                    instruction = example.get("instruction", example.get("prompt", "请分析以下代码"))
                    input_text = example.get("input", example.get("input_text", ""))
                    output_text = example.get("output", example.get("response", example.get("target", "")))
                    
                    text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n{output_text}"
                
                return {"text": text}
            
            # 应用格式化
            formatted_dataset = dataset.map(format_instruction, remove_columns=dataset["train"].column_names)
            
            # 分词
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=self.config.get("max_length", 2048),
                    return_tensors=None
                )
            
            tokenized_dataset = formatted_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            
            logger.success(f"数据集准备完成: 训练集 {len(tokenized_dataset['train'])} 样本")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"数据集加载失败: {str(e)}")
            raise
    
    def setup_trainer(self, dataset):
        """设置训练器"""
        output_dir = self.config["output_dir"]
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.get("batch_size", 1),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            num_train_epochs=self.config.get("num_epochs", 3),
            learning_rate=self.config.get("learning_rate", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_steps=self.config.get("eval_steps", 500),
            evaluation_strategy="steps" if "test" in dataset else "no",
            save_total_limit=self.config.get("save_total_limit", 3),
            load_best_model_at_end=True if "test" in dataset else False,
            metric_for_best_model="eval_loss" if "test" in dataset else None,
            greater_is_better=False,
            warmup_steps=self.config.get("warmup_steps", 100),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            optim="adamw_torch",
            dataloader_drop_last=True,
            report_to=["wandb"] if self.config.get("use_wandb", False) else [],
            run_name=self.config.get("run_name", "deepcode-analyst-finetune"),
            seed=42,
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            remove_unused_columns=False
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # SFT Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            max_seq_length=self.config.get("max_length", 2048),
            dataset_text_field="text",
            packing=False
        )
        
        logger.success("训练器设置完成")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        try:
            # 初始化 wandb（如果启用）
            if self.config.get("use_wandb", False):
                wandb.init(
                    project=self.config.get("wandb_project", "deepcode-analyst"),
                    name=self.config.get("run_name", "finetune"),
                    config=self.config
                )
            
            # 开始训练
            train_result = self.trainer.train()
            
            # 保存训练结果
            self.trainer.save_model()
            self.trainer.save_state()
            
            # 保存分词器
            self.tokenizer.save_pretrained(self.config["output_dir"])
            
            # 保存训练配置
            config_path = Path(self.config["output_dir"]) / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.success(f"训练完成！模型已保存到: {self.config['output_dir']}")
            
            return {
                "success": True,
                "output_dir": self.config["output_dir"],
                "train_loss": train_result.training_loss,
                "message": "训练完成"
            }
            
        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "训练失败"
            }
    
    def run_full_pipeline(self):
        """运行完整的微调流程"""
        try:
            # 1. 设置模型和分词器
            self.setup_model_and_tokenizer()
            
            # 2. 设置LoRA配置
            self.setup_lora_config()
            
            # 3. 加载和准备数据集
            dataset = self.load_and_prepare_dataset()
            
            # 4. 设置训练器
            self.setup_trainer(dataset)
            
            # 5. 开始训练
            result = self.train()
            
            return result
            
        except Exception as e:
            error_msg = f"微调流程失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "微调流程失败"
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="QLoRA 微调脚本")
    
    # 模型和数据参数
    parser.add_argument("--model_id", default="Qwen/Qwen1.5-7B-Chat",
                       help="基础模型ID")
    parser.add_argument("--dataset_path", required=True,
                       help="训练数据集路径")
    parser.add_argument("--output_dir", default="./models/deepcode-analyst-finetuned",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="最大序列长度")
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # 其他参数
    parser.add_argument("--use_wandb", action="store_true",
                       help="是否使用 Weights & Biases")
    parser.add_argument("--wandb_project", default="deepcode-analyst",
                       help="W&B 项目名称")
    parser.add_argument("--run_name", help="运行名称")
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = vars(args)
    
    # 创建输出目录
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # 开始微调
    finetuner = QLoRAFineTuner(config)
    result = finetuner.run_full_pipeline()
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"模型保存路径: {result['output_dir']}")
        if "train_loss" in result:
            print(f"最终训练损失: {result['train_loss']:.4f}")
    else:
        print(f"❌ {result['message']}")
        if "error" in result:
            print(f"错误: {result['error']}")


if __name__ == "__main__":
    main()
