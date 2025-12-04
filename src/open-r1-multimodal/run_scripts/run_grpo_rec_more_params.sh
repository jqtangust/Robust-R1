cd /workspace/VLM-R1/src/open-r1-multimodal

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=1,2

RUN_NAME="Qwen2.5-VL-3B-GRPO-REC-SFT"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root /data \
    --max_prompt_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --beta 0.0 \
    --epsilon_high 0.28
