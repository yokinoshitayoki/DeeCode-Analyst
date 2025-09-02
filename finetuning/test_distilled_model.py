"""
测试蒸馏后的模型效果
在AutoDL云端环境中运行
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json

# AutoDL环境配置
cache_dir = "/root/autodl-tmp/huggingface_cache"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_teacher_model():
    """加载教师模型"""
    print("🎓 加载教师模型...")
    
    base_model_id = "Qwen/Qwen1.5-7B-Chat"
    adapter_path = "/root/autodl-tmp/models/deepcode-analyst-finetuned"
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    teacher_model = PeftModel.from_pretrained(base_model, adapter_path)
    return teacher_model, tokenizer

def load_distilled_model():
    """加载蒸馏模型"""
    print("🔬 加载蒸馏模型...")
    
    model_path = "/root/autodl-tmp/models/deepcode-analyst-distilled"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"蒸馏模型不存在: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    student_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return student_model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    """生成模型响应"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    generation_time = end_time - start_time
    
    return response.strip(), generation_time

def run_comparison_test():
    """运行对比测试"""
    print("=" * 60)
    print("🧪 DeepCode-Analyst 蒸馏效果测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            "name": "项目架构分析",
            "prompt": """### 指令:
根据提供的代码库结构分析，生成一份关于项目整体架构的技术报告

### 输入:
{"project_name": "E-commerce-API", "components": ["frontend", "backend", "database"], "key_files": ["app.py", "models.py", "api.js"]}

### 输出:
"""
        },
        {
            "name": "函数分析",
            "prompt": """### 指令:
请深入分析指定Python函数的内部实现逻辑和参数

### 输入:
{"function_name": "calculate_score", "parameters": ["data", "weights"], "complexity": "medium"}

### 输出:
"""
        },
        {
            "name": "性能优化建议",
            "prompt": """### 指令:
分析给定的代码片段，识别性能瓶颈并提供优化建议

### 输入:
{"code_snippet": "for item in large_list: result = expensive_operation(item)", "context": "处理大量数据时性能较慢"}

### 输出:
"""
        }
    ]
    
    try:
        # 加载模型
        teacher_model, teacher_tokenizer = load_teacher_model()
        distilled_model, distilled_tokenizer = load_distilled_model()
        
        # 模型参数对比
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        distilled_params = sum(p.numel() for p in distilled_model.parameters())
        compression_ratio = teacher_params / distilled_params
        
        print(f"📊 模型对比:")
        print(f"   教师模型参数: {teacher_params:,}")
        print(f"   蒸馏模型参数: {distilled_params:,}")
        print(f"   压缩比例: {compression_ratio:.1f}x")
        print()
        
        # 运行测试
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"📝 测试案例 {i}: {test_case['name']}")
            print("-" * 50)
            
            prompt = test_case['prompt']
            
            # 教师模型
            print("🎓 教师模型响应:")
            teacher_response, teacher_time = generate_response(teacher_model, teacher_tokenizer, prompt)
            print(f"⏱️  生成时间: {teacher_time:.2f}秒")
            print(f"📄 响应长度: {len(teacher_response)}字符")
            print(f"💬 内容预览: {teacher_response[:100]}...")
            print()
            
            # 蒸馏模型
            print("🔬 蒸馏模型响应:")
            distilled_response, distilled_time = generate_response(distilled_model, distilled_tokenizer, prompt)
            print(f"⏱️  生成时间: {distilled_time:.2f}秒")
            print(f"📄 响应长度: {len(distilled_response)}字符")
            print(f"💬 内容预览: {distilled_response[:100]}...")
            print()
            
            # 性能对比
            speed_improvement = teacher_time / distilled_time if distilled_time > 0 else float('inf')
            length_retention = len(distilled_response) / len(teacher_response) if len(teacher_response) > 0 else 0
            
            print("📈 性能对比:")
            print(f"   速度提升: {speed_improvement:.1f}x")
            print(f"   长度保留: {length_retention:.1%}")
            
            # 质量评估
            quality_score = 0
            if any(keyword in distilled_response for keyword in ['分析', '报告', '建议', '技术', '架构']):
                quality_score += 0.3
            if any(marker in distilled_response for marker in ['###', '**', '1.', '2.', '-', '*']):
                quality_score += 0.3
            if len(distilled_response) > 50:
                quality_score += 0.4
            
            print(f"   质量评分: {quality_score:.1%}")
            print()
            
            results.append({
                'test_case': test_case['name'],
                'teacher_time': teacher_time,
                'distilled_time': distilled_time,
                'speed_improvement': speed_improvement,
                'length_retention': length_retention,
                'quality_score': quality_score,
                'teacher_response': teacher_response,
                'distilled_response': distilled_response
            })
        
        # 总体评估
        print("=" * 60)
        print("📊 总体评估结果")
        print("=" * 60)
        
        avg_speed = sum(r['speed_improvement'] for r in results) / len(results)
        avg_length = sum(r['length_retention'] for r in results) / len(results)
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        
        print(f"平均速度提升: {avg_speed:.1f}x")
        print(f"平均长度保留: {avg_length:.1%}")
        print(f"平均质量评分: {avg_quality:.1%}")
        print()
        
        # 蒸馏成功判定
        success_criteria = {
            'speed_improvement': avg_speed >= 2.0,
            'length_retention': avg_length >= 0.7,
            'quality_score': avg_quality >= 0.6
        }
        
        print("✅ 蒸馏成功标准:")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}: {'通过' if passed else '未通过'}")
        
        overall_success = all(success_criteria.values())
        print(f"\n🎯 整体评估: {'🎉 蒸馏成功!' if overall_success else '⚠️ 需要优化'}")
        
        # 保存测试结果
        test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_comparison': {
                'teacher_params': teacher_params,
                'distilled_params': distilled_params,
                'compression_ratio': compression_ratio
            },
            'performance_metrics': {
                'avg_speed_improvement': avg_speed,
                'avg_length_retention': avg_length,
                'avg_quality_score': avg_quality
            },
            'success_criteria': success_criteria,
            'overall_success': overall_success,
            'detailed_results': results
        }
        
        result_path = "/root/autodl-tmp/models/deepcode-analyst-distilled/test_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 测试结果已保存: {result_path}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def quick_inference_test():
    """快速推理测试"""
    print("\n🚀 快速推理测试")
    print("-" * 30)
    
    try:
        distilled_model, tokenizer = load_distilled_model()
        
        test_prompt = """### 指令:
生成代码分析报告

### 输入:
{"file": "main.py", "functions": 3, "complexity": "low"}

### 输出:
"""
        
        response, inference_time = generate_response(distilled_model, tokenizer, test_prompt, max_new_tokens=100)
        
        print(f"⏱️  推理时间: {inference_time:.3f}秒")
        print(f"📄 生成长度: {len(response)}字符")
        print("\n💬 生成内容:")
        print("-" * 30)
        print(response)
        print("-" * 30)
        print("✅ 快速测试完成")
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")

def main():
    """主函数"""
    print("🔬 DeepCode-Analyst 蒸馏模型测试")
    
    # 检查模型是否存在
    distilled_path = "/root/autodl-tmp/models/deepcode-analyst-distilled"
    teacher_path = "/root/autodl-tmp/models/deepcode-analyst-finetuned"
    
    if not os.path.exists(distilled_path):
        print(f"❌ 蒸馏模型不存在: {distilled_path}")
        print("请先运行知识蒸馏")
        return
    
    if not os.path.exists(teacher_path):
        print(f"⚠️  教师模型不存在: {teacher_path}")
        print("将只运行蒸馏模型的快速测试")
        quick_inference_test()
        return
    
    # 运行完整对比测试
    run_comparison_test()

if __name__ == "__main__":
    main()