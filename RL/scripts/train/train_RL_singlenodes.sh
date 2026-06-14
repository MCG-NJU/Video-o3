#!/bin/bash
# 兼容旧入口，转发到 train_RL_video_single.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_RL_video_single.sh" "$@"
