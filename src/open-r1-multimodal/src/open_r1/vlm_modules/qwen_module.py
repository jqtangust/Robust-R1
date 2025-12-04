from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re
from transformers import AutoTokenizer
from vlm_modules.vlm_module import VLMBaseModule
import math
import numpy as np

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # print(type(prompts_text))
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "robust":
                return "{Question}First output the types of degradations in image briefly in <TYPE> <TYPE_END> tags, and then output what effects do these degradation have on the image in <INFLUENCE> <INFLUENCE_END> tags, then based on the strength of degradation, output an APPROPRIATE length for the reasoning process in <REASONING> <REASONING_END> tags, and then summarize the content of reasoning and the give the answer in <CONCLUSION> <CONCLUSION_END> tags,provides the user with the answer briefly in <ANSWER> <ANSWER_END>.i.e., <TYPE> degradation type here <TYPE_END>\n<INFLUENCE> influence here<INFLUENCE_END>\n<REASONING> reasoning process here<REASONING_END>\n<CONCLUSION>summary here<CONCLUSION_END>\n<ANSWER>final answer<ANSWER_END>"
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_robust(completions, **kwargs):
        import re
        import os
        from datetime import datetime

        pattern = r"<TYPE>.*?<TYPE_END>\s*<INFLUENCE>.*?<INFLUENCE_END>\s*<REASONING>.*?<REASONING_END>\s*<CONCLUSION>.*?<CONCLUSION_END>\s*<ANSWER>.*?<ANSWER_END>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def type_reward(completions, solution, **kwargs):
        def custom_normalize_reward(score, k_positive=1.0, k_negative=2.0, x0=0.0):
            sigmoid_output = 0.0
            if score >= x0:
                sigmoid_output = 1 / (1 + math.exp(-k_positive * (score - x0)))
            else:
                sigmoid_output = 1 / (1 + math.exp(-k_negative * (score - x0)))
                
            normalized_score = 2 * sigmoid_output - 1
            
            return normalized_score
            
        def extract_image_degradations(text):
            match = re.search(r'<TYPE>(.*?)<TYPE_END>', text, re.DOTALL)
            if not match:
                return []

            types_string = match.group(1)
            degradations = re.findall(r'(\w+(?:\s+\w+)*)\(([\d.]+)\)', types_string)

            result = []
            for degradation, strength in degradations:
                result.append((degradation.strip(), float(strength)))
            
            return result

        def calculate_reward(A, B):
            reward = 0.0
            
            B_dict = dict(B)
            matched_keys = set()
            
            for degradation_A, strength_A in A:
                if degradation_A in B_dict:
                    reward += 1
                    strength_B = B_dict[degradation_A]
                    diff = abs(strength_A - strength_B)
                    reward += (0.5 - diff)
                    matched_keys.add(degradation_A)
                else:
                    reward -= 1
                    
            for degradation_B in B_dict:
                if degradation_B not in matched_keys:
                    reward -= 1
                    
            return reward
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for i in range(len(contents)):
            content_single = extract_image_degradations(contents[i])
            solution_single = extract_image_degradations(solution[i])
            score = calculate_reward(content_single, solution_single)
            rewards.append(score)
        
        return rewards
    
    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
        def extract_answer(text):
            match = re.search(r'<ANSWER>(.*?)<ANSWER_END>', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return None
        
        contents = [completion[0]["content"] for completion in completions]
        
        if len(contents) != len(solution):
            print("Warning: Input list lengths do not match.")
            return []

        rewards = []
        for i in range(len(contents)):
            model_answer = extract_answer(contents[i])
            gt_answer = extract_answer(solution[i])
            if model_answer == gt_answer:
                rewards.append(1)
            else:
                rewards.append(0)
            
        return rewards
    
    @staticmethod
    def length_reward(completions, solution, **kwargs):
        
        processor = AutoProcessor.from_pretrained("your_model_path",user_fast=False)
        tokenizer =processor.tokenizer

        responses = [completion[0]["content"] for completion in completions]
        
        if len(responses) != len(solution):
            print("Warning: Input list lengths do not match.")
            return []
        
        rewards = []
        for resp, sol in zip(responses, solution):
            resp_len = len(tokenizer.encode(resp))
            sol_len = len(tokenizer.encode(sol))
            
            length_diff = abs(resp_len - sol_len)
            
            reward = 1 - (length_diff/sol_len)
            
            rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        import re
        import os
        from datetime import datetime
        import json
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
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        reward = iou(bbox, sol)
            except Exception:
                pass
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "robust":
                    return Qwen2VLModule.accuracy_reward
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        
        elif func == "format":
            match task_type:
                case "robust":
                    return Qwen2VLModule.format_reward_robust
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        
        elif func == "type":
            match task_type:
                case "robust":
                    return Qwen2VLModule.type_reward
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        
        elif func == "length":
            match task_type:
                case "robust":
                    return Qwen2VLModule.length_reward
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


    