import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from open_r1.vlm_modules.internvl_module import InvernVLModule

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

main_rank = 0
steps = 300
if rank == main_rank:
    print("Steps: ", steps)

RUN_NAME = "InternVL2_5-4B_MPO-rec"

MODEL_PATH=f"/training/shz/project/vlm-r1/VLM-R1/checkpoints/rl/{RUN_NAME}/checkpoint-{steps}" 
OUTPUT_PATH="./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"

BSZ=4
DATA_ROOT = "/training/shz/dataset/vlm-r1/rec_jsons_internvl"

# TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
# IMAGE_ROOT = "/training/shz/dataset/coco"

TEST_DATASETS = ['lisa_test']
IMAGE_ROOT = "/training/shz/dataset/lisa"

random.seed(42)

vlm_module = InvernVLModule()

model = vlm_module.get_model_class(MODEL_PATH, {}).from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
    trust_remote_code=True,
    use_flash_attn=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
vlm_module.post_model_init(model, tokenizer)


def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox
    return [0, 0, 0, 0]

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

from PIL import Image
def process_vision_info(batch_messages):
    images = []
    for msg in batch_messages:
        image_path = msg[0]['content'][0]['image'].replace("file://", "")
        image = Image.open(image_path)
        images.append(image)
    return images


sample_num = 2000
tokenizer.max_anyres_num = 12
for ds in TEST_DATASETS:
    if rank == main_rank:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    data = data[:sample_num]
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []
    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }]
        messages.append(message)
    
    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank):
        batch_messages = messages[i:i + BSZ]
        prompts = vlm_module.prepare_prompt(None, [{"prompt": msg} for msg in batch_messages])

        images = process_vision_info(batch_messages)

        model_inputs = vlm_module.prepare_model_inputs(tokenizer, prompts, images)
        model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
        model_inputs = model_inputs.to(device)

        outputs = model.generate(**{k:v for k,v in model_inputs.items() if k not in vlm_module.get_non_generate_params()}, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        batch_output_text = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rank_outputs.extend(batch_output_text)
    
    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == main_rank:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            model_answer = extract_bbox_answer(original_output)
            
            # Count correct answers
            correct = 0
            if model_answer is not None and iou(model_answer, ground_truth) > 0.5:
                correct = 1
            correct_number += correct
            
            # Create a result dictionary for this example
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=4)

        print(f"Results saved to {output_path}")
        print("-"*100)

    # Synchronize all processes
    dist.barrier()





