"""
DeepCode-Analyst 本地快速启动脚本
"""
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from local_model_loader import LocalModelLoader
import json

def main():
    """主函数"""
    print("🚀 DeepCode-Analyst 本地版启动")
    print("=" * 50)
    
    # 加载配置
    try:
        with open("local_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 配置文件加载成功")
    except:
        print("❌ 配置文件加载失败，使用默认配置")
        config = {"local_config": {"cloud_data_dir": "./cloud_data"}}
    
    # 创建模型加载器
    loader = LocalModelLoader(config["local_config"]["cloud_data_dir"])
    
    # 显示模型状态
    info = loader.get_model_info()
    print("\n📊 模型状态:")
    for model_name, model_info in info["models"].items():
        status = "✅" if model_info["exists"] else "❌"
        print(f"  {status} {model_name}: {model_info['size']}")
    
    # 选择要使用的模型
    print("\n🤖 选择模型:")
    print("1. 0.5B学生模型 (推荐，轻量级)")
    print("2. 7B教师模型 (需要更多GPU内存)")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        print("\n👨‍🎓 加载0.5B学生模型...")
        model, tokenizer = loader.load_student_model()
    elif choice == "2":
        print("\n🎓 加载7B教师模型...")
        model, tokenizer = loader.load_teacher_model()
    else:
        print("❌ 无效选择")
        return
    
    if model is None:
        print("❌ 模型加载失败")
        return
    
    # 交互式问答
    print("\n💬 开始交互 (输入 'quit' 退出):")
    while True:
        user_input = input("\n🙋 您的问题: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        if not user_input:
            continue
        
        # 格式化输入
        prompt = f"""### 指令:
分析代码并提供建议

### 输入:
{user_input}

### 输出:
"""
        
        try:
            # 推理
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print("🤖 思考中...")
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.get("inference_config", {}).get("max_new_tokens", 200),
                    temperature=config.get("inference_config", {}).get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            print(f"🤖 回答:\n{response}")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
    
    print("\n👋 再见!")

if __name__ == "__main__":
    main()
