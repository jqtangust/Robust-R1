# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.

# Install torch first (required for flash-attn)
pip install torch>=2.5.1 torchvision

# Install open-r1 package with dev dependencies
pip install -e ".[dev]"

# Additional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]

# Install flash-attn last (requires torch to be already installed)
pip install flash-attn --no-build-isolation