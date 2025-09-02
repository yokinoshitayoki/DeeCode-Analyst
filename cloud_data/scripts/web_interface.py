"""
DeepCode-Analyst 0.5B学生模型 Web界面
基于Gradio的友好交互界面
"""
import os
import sys
import torch
import warnings
from pathlib import Path
import json
from datetime import datetime

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 忽略警告
warnings.filterwarnings("ignore")

class WebInterface:
    """Web界面类"""
    
    def __init__(self):
        self.cloud_data_dir = Path(__file__).parent.parent
        self.adapter_path = self.cloud_data_dir / "models" / "student_0.5b"
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self):
        """加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            print("🚀 正在加载模型...")
            
            # 检测设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            device_map = "auto" if self.device == "cuda" else "cpu"
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen1.5-0.5B-Chat",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-0.5B-Chat",
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 加载LoRA适配器
            self.model = PeftModel.from_pretrained(
                base_model, 
                str(self.adapter_path)
            )
            
            print("✅ 模型加载完成!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=200, temperature=0.7, top_p=0.9):
        """生成回应"""
        if not self.model or not self.tokenizer:
            return "❌ 模型未加载，请检查模型文件"
        
        try:
            # 格式化输入
            formatted_prompt = f"""### 指令:
作为代码分析专家，请分析以下内容并提供专业建议

### 输入:
{prompt}

### 输出:
"""
            
            # 编码
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"❌ 生成失败: {e}"
    
    def create_gradio_interface(self):
        """创建Gradio界面"""
        try:
            import gradio as gr
        except ImportError:
            print("❌ Gradio未安装，请运行: pip install gradio")
            return None
        
        def chat_function(message, history, max_tokens, temperature, top_p):
            """聊天函数"""
            if not message.strip():
                return history, ""
            
            # 生成回应
            response = self.generate_response(
                message, 
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p
            )
            
            # 更新历史
            history.append([message, response])
            
            return history, ""
        
        def clear_history():
            """清除历史"""
            return [], ""
        
        def load_example(example):
            """加载示例"""
            return example
        
        # 创建界面
        with gr.Blocks(
            title="DeepCode-Analyst 0.5B",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-message {
                padding: 10px;
                margin: 5px;
                border-radius: 10px;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🤖 DeepCode-Analyst 0.5B 学生模型
            
            基于知识蒸馏技术，将7B教师模型的代码分析能力压缩到0.5B参数。
            
            **专长领域**: 代码分析、性能优化、算法解释、代码重构建议
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 聊天界面
                    chatbot = gr.Chatbot(
                        label="💬 对话",
                        height=400,
                        show_label=True,
                        container=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="输入您的问题",
                            placeholder="请输入您的代码分析需求...",
                            lines=3,
                            scale=4
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("清除对话", variant="secondary")
                        
                with gr.Column(scale=1):
                    # 参数设置
                    gr.Markdown("### ⚙️ 生成参数")
                    
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=10,
                        label="最大生成长度"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="创造性 (Temperature)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="多样性 (Top-p)"
                    )
                    
                    # 示例
                    gr.Markdown("### 📝 示例问题")
                    
                    examples = [
                        "分析这段代码的时间复杂度：\ndef find_max(arr):\n    max_val = arr[0]\n    for i in range(1, len(arr)):\n        if arr[i] > max_val:\n            max_val = arr[i]\n    return max_val",
                        "如何优化这个循环：\nresult = []\nfor i in range(len(data)):\n    if data[i] > 0:\n        result.append(data[i] * 2)",
                        "解释这个排序算法的工作原理：\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]"
                    ]
                    
                    for i, example in enumerate(examples, 1):
                        example_btn = gr.Button(
                            f"示例 {i}", 
                            variant="outline",
                            size="sm"
                        )
                        example_btn.click(
                            fn=load_example,
                            inputs=[gr.State(example)],
                            outputs=[msg]
                        )
            
            # 系统信息
            with gr.Accordion("🔧 系统信息", open=False):
                device_info = f"运行设备: {self.device.upper()}"
                if self.device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    device_info += f" ({gpu_name})"
                
                gr.Markdown(f"""
                **模型信息**:
                - 基础模型: Qwen1.5-0.5B-Chat
                - 适配器: LoRA知识蒸馏
                - 教师模型: Qwen1.5-7B-Chat (微调)
                - {device_info}
                - 参数量: ~0.5B
                - 专业领域: 代码分析与优化
                """)
            
            # 绑定事件
            submit_btn.click(
                fn=chat_function,
                inputs=[msg, chatbot, max_tokens, temperature, top_p],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=chat_function,
                inputs=[msg, chatbot, max_tokens, temperature, top_p],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                fn=clear_history,
                outputs=[chatbot, msg]
            )
        
        return interface
    
    def create_streamlit_interface(self):
        """创建Streamlit界面"""
        try:
            import streamlit as st
        except ImportError:
            print("❌ Streamlit未安装，请运行: pip install streamlit")
            return None
        
        st.set_page_config(
            page_title="DeepCode-Analyst 0.5B",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 DeepCode-Analyst 0.5B 学生模型")
        st.markdown("基于知识蒸馏的代码分析AI助手")
        
        # 侧边栏设置
        with st.sidebar:
            st.header("⚙️ 设置")
            
            max_tokens = st.slider("最大生成长度", 50, 500, 200, 10)
            temperature = st.slider("创造性", 0.1, 2.0, 0.7, 0.1)
            top_p = st.slider("多样性", 0.1, 1.0, 0.9, 0.1)
            
            st.header("📝 示例")
            if st.button("代码复杂度分析"):
                st.session_state.example_input = "分析这段代码的时间复杂度：\ndef linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1"
            
            if st.button("性能优化建议"):
                st.session_state.example_input = "如何优化这段代码：\nresult = []\nfor item in data:\n    if item > 0:\n        result.append(item * 2)"
        
        # 主界面
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 输入框
        user_input = st.text_area(
            "输入您的问题:",
            value=st.session_state.get('example_input', ''),
            height=100,
            placeholder="请输入您的代码分析需求..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 分析", type="primary"):
                if user_input.strip():
                    with st.spinner("正在分析..."):
                        response = self.generate_response(
                            user_input,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "assistant": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
        
        with col2:
            if st.button("🗑️ 清除历史"):
                st.session_state.chat_history = []
                st.experimental_rerun()
        
        # 显示对话历史
        if st.session_state.chat_history:
            st.header("💬 对话历史")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # 只显示最近5条
                with st.expander(f"对话 {len(st.session_state.chat_history)-i} ({chat['timestamp']})"):
                    st.markdown(f"**👤 用户**: {chat['user']}")
                    st.markdown(f"**🤖 助手**: {chat['assistant']}")

def main():
    """主函数"""
    print("🌐 启动DeepCode-Analyst Web界面...")
    
    # 创建Web界面实例
    web_interface = WebInterface()
    
    # 检查模型文件
    if not web_interface.adapter_path.exists():
        print(f"❌ 模型文件不存在: {web_interface.adapter_path}")
        print("💡 请先下载学生模型文件")
        return
    
    # 加载模型
    if not web_interface.load_model():
        print("❌ 模型加载失败")
        return
    
    # 选择界面类型
    print("\n🎮 选择Web界面类型:")
    print("1. Gradio (推荐) - 现代化聊天界面")
    print("2. Streamlit - 传统Web应用界面")
    
    try:
        choice = input("请选择 (1/2，默认1): ").strip() or "1"
        
        if choice == "1":
            # Gradio界面
            interface = web_interface.create_gradio_interface()
            if interface:
                print("🚀 启动Gradio界面...")
                interface.launch(
                    server_name="0.0.0.0",
                    server_port=7860,
                    share=False,
                    inbrowser=True
                )
            
        elif choice == "2":
            # Streamlit界面
            print("🚀 启动Streamlit界面...")
            print("请在新终端运行: streamlit run cloud_data/scripts/web_interface.py")
            
            # 创建Streamlit应用
            web_interface.create_streamlit_interface()
            
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n⏹️ Web界面已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()