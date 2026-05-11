# 
# cd flow_grpo/reward-server
# gunicorn "app_videoalign:create_app()" 

RANK=${RANK:=0}
PORT=${MASTER_PORT:-"6000"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}

# Project root directory (modify according to actual path)
PROJECT_ROOT=""
cd $PROJECT_ROOT
export PYTHONPATH=''

wandb offline
export WANDB_API_KEY=''

# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 12 --num_processes 96 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${PORT} \
    scripts/train_wanx2_1_sample.py \
    --config config/dgx.py:wan2_1_flash
