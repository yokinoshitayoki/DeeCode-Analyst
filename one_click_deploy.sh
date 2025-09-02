#!/bin/bash
# 一键部署脚本 - 在AutoDL服务器上直接运行
# 使用方法: 复制此脚本内容，在AutoDL服务器终端中粘贴执行

set -e

echo "🚀 开始DeepCode-Analyst一键部署..."
echo "服务器信息: $(hostname) - $(whoami)"

# 1. 更新系统和安装基础工具
echo "📦 安装系统依赖..."
apt-get update -qq
apt-get install -y -qq git wget curl vim htop tmux tree build-essential python3-dev python3-pip

# 2. 检查GPU
echo "🎮 检查GPU状态..."
nvidia-smi

# 3. 安装/配置Conda
echo "🐍 设置Conda环境..."
if ! command -v conda &> /dev/null; then
    echo "安装Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# 创建Python环境
conda create -n deepcode python=3.9 -y
source ~/miniconda/etc/profile.d/conda.sh
conda activate deepcode

# 4. 检查项目目录
PROJECT_DIR="/root/DeepCode-Analyst"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 项目目录不存在: $PROJECT_DIR"
    echo "请先上传项目文件到服务器"
    echo ""
    echo "上传方法1 - 本地执行:"
    echo "scp -P 18812 -r ./DeepCode-Analyst root@connect.bjb1.seetacloud.com:/root/"
    echo ""
    echo "上传方法2 - 使用工具:"
    echo "python upload_to_autodl.py"
    echo ""
    read -p "项目文件上传完成后按Enter继续..."
    
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "❌ 项目目录仍不存在，退出部署"
        exit 1
    fi
fi

cd $PROJECT_DIR

# 5. 安装Python依赖
echo "📚 安装Python依赖..."
python -m pip install --upgrade pip

# 安装PyTorch (CUDA版本)
echo "安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
for req_file in "requirements_autodl.txt" "finetuning/requirements.txt" "requirements.txt"; do
    if [ -f "$req_file" ]; then
        echo "安装 $req_file"
        python -m pip install -r "$req_file"
    fi
done

# 6. 验证安装
echo "✅ 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 7. 准备训练数据
echo "📊 准备训练数据..."
if [ ! -f "finetuning/training_data.jsonl" ]; then
    python finetuning/prepare_dataset.py --mode sample --num-samples 50 --output-path ./finetuning/training_data.jsonl
fi

# 检查数据文件
if [ -f "finetuning/training_data.jsonl" ]; then
    SAMPLE_COUNT=$(wc -l < finetuning/training_data.jsonl)
    echo "✅ 训练数据准备完成，包含 $SAMPLE_COUNT 个样本"
else
    echo "❌ 训练数据生成失败"
    exit 1
fi

# 8. 检测GPU显存并创建配置
echo "⚙️ 创建训练配置..."
GPU_MEMORY=8192
if command -v nvidia-smi &> /dev/null; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
fi

echo "检测到GPU显存: ${GPU_MEMORY}MB"

# 根据显存调整配置
if [ "$GPU_MEMORY" -lt 8192 ]; then
    BATCH_SIZE=1
    GRAD_ACCUM=16
    MAX_LEN=1024
    LORA_R=8
    echo "使用小显存配置"
elif [ "$GPU_MEMORY" -lt 16384 ]; then
    BATCH_SIZE=1
    GRAD_ACCUM=8
    MAX_LEN=2048
    LORA_R=16
    echo "使用标准配置"
else
    BATCH_SIZE=2
    GRAD_ACCUM=4
    MAX_LEN=2048
    LORA_R=32
    echo "使用高性能配置"
fi

# 9. 创建启动脚本
echo "📝 创建训练启动脚本..."
cat > start_training.sh << EOF
#!/bin/bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate deepcode
cd $PROJECT_DIR

echo "开始微调训练..."
echo "配置: batch_size=$BATCH_SIZE, max_length=$MAX_LEN, lora_r=$LORA_R"

python finetuning/run_finetune.py \\
    --model_id "Qwen/Qwen1.5-7B-Chat" \\
    --dataset_path "./finetuning/training_data.jsonl" \\
    --output_dir "./models/deepcode-analyst-finetuned" \\
    --batch_size $BATCH_SIZE \\
    --gradient_accumulation_steps $GRAD_ACCUM \\
    --num_epochs 3 \\
    --learning_rate 2e-5 \\
    --max_length $MAX_LEN \\
    --lora_r $LORA_R \\
    --lora_alpha 32 \\
    --lora_dropout 0.05

echo "微调完成！模型保存在: ./models/deepcode-analyst-finetuned/"
EOF

chmod +x start_training.sh

# 10. 创建管理脚本
cat > manage.sh << EOF
#!/bin/bash
case "\$1" in
    start)
        echo "启动训练会话..."
        tmux new-session -d -s training "cd $PROJECT_DIR && source ~/miniconda/etc/profile.d/conda.sh && conda activate deepcode && ./start_training.sh"
        echo "训练已在后台启动，使用 './manage.sh status' 查看状态"
        ;;
    status)
        if tmux has-session -t training 2>/dev/null; then
            echo "训练正在运行..."
            tmux capture-pane -t training -p | tail -n 5
        else
            echo "训练未运行"
        fi
        ;;
    attach)
        tmux attach -t training
        ;;
    stop)
        tmux kill-session -t training 2>/dev/null || echo "训练会话不存在"
        ;;
    gpu)
        watch -n 1 nvidia-smi
        ;;
    *)
        echo "用法: \$0 {start|status|attach|stop|gpu}"
        echo "  start  - 后台启动训练"
        echo "  status - 查看训练状态"  
        echo "  attach - 连接到训练会话"
        echo "  stop   - 停止训练"
        echo "  gpu    - 监控GPU使用"
        ;;
esac
EOF

chmod +x manage.sh

# 11. 创建环境文件
cat > .env << EOF
# 环境变量配置
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=$PROJECT_DIR
EOF

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 使用说明："
echo "1. 直接开始训练:     ./start_training.sh"
echo "2. 后台启动训练:     ./manage.sh start"
echo "3. 查看训练状态:     ./manage.sh status"
echo "4. 连接训练会话:     ./manage.sh attach"
echo "5. 停止训练:         ./manage.sh stop"
echo "6. 监控GPU:          ./manage.sh gpu"
echo ""
echo "📁 重要路径："
echo "  项目目录: $PROJECT_DIR"
echo "  训练数据: $PROJECT_DIR/finetuning/training_data.jsonl"
echo "  模型输出: $PROJECT_DIR/models/deepcode-analyst-finetuned/"
echo ""
echo "现在可以开始训练了！建议先运行: ./manage.sh start"
