PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}" # TODO: change this to your own 
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="${REPO_HOME}/src/open-r1-multimodal/data_jsonl/gui_multi-image.jsonl" 
image_folders="/data9/shz/project/vlm-r1/VLM-R1/images/gui_multi-image"
model_path="/data9/shz/ckpt/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=False
reward_methods="all_match"  
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="GUI-multi-image" # TODO: change this to your own experiment name
TASK_TYPE="gui"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
MAX_STEPS=1200 # TODO: change this to your own max steps

# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --reward_method $reward_methods \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --max_steps $MAX_STEPS \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --save_steps 400 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \

echo "Training completed for ${EXP_NAME}"
