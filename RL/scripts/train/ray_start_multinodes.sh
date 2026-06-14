#!/bin/bash
# >>> 先启动一些通用的环境  >>>
set -ex
export PATH="/mnt/shared-storage-user/zengxiangyu/miniconda3/bin:$PATH"
__conda_setup="$('/mnt/shared-storage-user/zengxiangyu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/shared-storage-user/zengxiangyu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/zengxiangyu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/shared-storage-user/zengxiangyu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda deactivate
# conda activate video-o3_rl
conda activate /mnt/shared-storage-user/zengxiangyu/miniconda3/envs/mini-o3-qwen3

RL_ROOT="/your_local_path_to/Video-o3/RL"
cd ${RL_ROOT}
# sh ${RL_ROOT}/scripts/s3mount.sh  # optional: mount S3 if needed
# <<< 先启动一些通用的环境  <<<

# Parse command line arguments
NNODES=${NNODES:-4}
VERTION_NAME=${VERTION_NAME:-"260116v2_suoha_768F_2FPS_fpt_clue3"}
while [ "$#" -gt 0 ]; do
    case $1 in
        --nodes_num) NNODES="$2"; shift ;;
        --version_name) VERTION_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export RUN_NAME=GRPO-${VERTION_NAME}-NNODES${NNODES}

export DATA_MODE="video"
# export SELF_SET_OVERVIEW_FPS=1.0
# export SELF_SET_FPS_MAX_FRAMES=384
export VLLM_USE_V1=1
export WANDB_MODE=offline
# export PRETRAINED_PATH="/root/s3/videogpu/videochat-o3/ckpt/Qwen3VL-4B"
export PRETRAINED_PATH=${PRETRAINED_PATH:-"/root/s3/videogpu/videochat-o3/ckpt/qwen3vl_4b_sft_v2"}
export VISION_PATCH_SIZE=16
export BASE_IMAGE_DIR="/root/s3"
export CKPT_SAVE_DIR="/root/s3/videogpu/videochat-o3/ckpt/${RUN_NAME}"
export LOG_SAVE_DIR="./log/${RUN_NAME}/resume_$(date +"%Y%m%d-%H%M")"
export WANDB_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/wandb_dir
export WANDB_ARTIFACT_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/artifacts_dir
export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

MAX_TOKENS=6144
MIN_TOKENS=256
NFRAMES=128
CROP_MAX_TOKENS_COARSE=1024
CROP_MAX_TOKENS_MEDIUM=2048
CROP_MAX_TOKENS_FINE=3072
export MAX_TOKENS MIN_TOKENS NFRAMES CROP_MAX_TOKENS_COARSE CROP_MAX_TOKENS_MEDIUM CROP_MAX_TOKENS_FINE VISION_PATCH_SIZE

export train_script=${RL_ROOT}/scripts/train/train_RL_multinodes.sh
mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${WANDB_ARTIFACT_DIR}
mkdir -p ${LOG_SAVE_DIR}

echo "NNODES = $NNODES"
echo "NPROC_PER_NODE = $NPROC_PER_NODE"
echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"
echo "NODE_RANK = $NODE_RANK"

if [ $NODE_RANK -eq 0 ]; then
    # Start head node
    ray start --block --head --port=6379 --node-manager-port=33000 --object-manager-port=33001 --runtime-env-agent-port=33002 --dashboard-agent-grpc-port=33003 --dashboard-agent-listen-port=33004 --metrics-export-port=33005 &
    sleep 10
    nnodes=$NNODES bash $train_script
else
    # Wait until head node is ready
    until nc -z ${MASTER_ADDR} 6379; do
        echo "Waiting for Ray head at ${MASTER_ADDR}:6379..."
        sleep 2
    done
    ray start --block --address=${MASTER_ADDR}:6379 --node-manager-port=33000 --object-manager-port=33001 --runtime-env-agent-port=33002 --dashboard-agent-grpc-port=33003 --dashboard-agent-listen-port=33004 --metrics-export-port=33005
fi
