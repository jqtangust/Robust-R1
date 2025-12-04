import re
import os
import json
import torch
import random

from tqdm import tqdm
from pprint import pprint
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def extract_bbox_answer(content):
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)
    bbox_json = json_match.group(1).strip() if json_match else None

    if bbox_json:
        try:
            bbox = json.loads(bbox_json)[0]['bbox_2d']
            return bbox, False
        except:
            return [0, 0, 0, 0], False
    else:
        return [0, 0, 0, 0], False


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union


def load_model(model_path, device_map):
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def eval_od_r1(
    model_path, test_datasets, data_root, image_root, question_template, output_dir, batch_size=32, sample_num=500, seed=42, device_map="cuda:0"
):
    random.seed(seed)
    model, processor = load_model(model_path, device_map)

    for ds in test_datasets:
        print(f"Processing {ds}...")

        ds_path = os.path.join(data_root, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.shuffle(data)
        data = data[:sample_num]
        messages = []

        for x in data:
            image_path = os.path.join(image_root, x['image'])
            messages.append(
                [
                    {
                        "role":
                            "user",
                        "content":
                            [
                                {
                                    "type": "image",
                                    "image": f"file://{image_path}"
                                }, {
                                    "type": "text",
                                    "text": question_template.format(Question=x['normal_caption'])
                                }
                            ]
                    }
                ]
            )

        all_outputs = []  # List to store all answers

        # Process data
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i + batch_size]

            # Preparation for inference
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device_map)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_outputs.extend(batch_output_text)

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['solution']
            ground_truth_normalized = input_example['normalized_solution']
            model_answer, normalized = extract_bbox_answer(original_output)

            # Count correct answers
            correct = 0
            if model_answer is not None:
                iou_value = iou(model_answer, ground_truth_normalized if normalized else ground_truth)
                if iou_value > 0.5:
                    correct = 1
            correct_number += correct

            # Create a result dictionary for this example
            result = {
                "question": question_template.format(Question=input_example['normal_caption']),
                "ground_truth": ground_truth if not normalized else ground_truth_normalized,
                "model_output": original_output,
                "extracted_answer": model_answer,
                "correct": correct,
                "iou": iou_value
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        result_path = os.path.join(output_dir, f"{os.path.basename(model_path)}", f"{ds}_od_r1.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)

        print(f"Results saved to {result_path}")
        print('-' * 100)


if __name__ == "__main__":
    model_path = ''  # Add the path to the model
    data_root = ''  # Add the data root
    test_datasets = ['refcoco_val', 'refcocop_val', 'refcocog_val']  # modify the datasets
    image_root = ''  # Add the image root
    output_dir = 'logs'  # Add the output directory, default is logs
    device_map = 'cuda:0'  # select the device, default is cuda:0

    question_template = '{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.'  # modify the question template which must contain {Question}, {Question} will be replaced by the caption

    eval_od_r1(
        model_path=model_path,
        data_root=data_root,
        test_datasets=test_datasets,
        image_root=image_root,
        question_template=question_template,
        output_dir=output_dir,
        device_map=device_map
    )
