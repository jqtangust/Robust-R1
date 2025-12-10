#!/usr/bin/env python3
"""
CLI Demo for Robust-R1: Visual Question Answering with Degradation-Aware Reasoning.
"""

import os
import sys
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default model path - can be overridden by MODEL_PATH environment variable
# Users can set MODEL_PATH to their local model path or HuggingFace model name
DEFAULT_MODEL_PATH = "Jiaqi-hkust/Robust-R1-RL"  # HuggingFace model name
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# Fixed image path for demo
FIXED_IMAGE_PATH = "assets/1.jpg"

SYS_PROMPT = """First output the the types of degradations in image briefly in <TYPE> <TYPE_END> tags, 
and then output what effects do these degradation have on the image in <INFLUENCE> <INFLUENCE_END> tags, 
then based on the strength of degradation, output an APPROPRIATE length for the reasoning process in <REASONING> <REASONING_END> tags, 
and then summarize the content of reasoning and the give the answer in <CONCLUSION> <CONCLUSION_END> tags,
provides the user with the answer briefly in <ANSWER> <ANSWER_END>."""

DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 1024


class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print("Loading model, this may take a few minutes...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise e

    def predict(self, question, image_path, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
        """
        Generate response for the given question and image.
        
        Args:
            question: User question
            image_path: Path to the image
            temperature: Generation temperature
            max_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text response
        """
        sys_prompt_formatted = " ".join(SYS_PROMPT.split())
        full_text = f"{question}\n{sys_prompt_formatted}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_text},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.model.device)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
        )
        
        try:
            print("Generating response...")
            with torch.no_grad():
                generated_ids = self.model.generate(**generation_kwargs)
            
            input_length = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[0][input_length:]
            
            generated_text = self.processor.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            return generated_text
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Generation error: {error_details}")
            raise e


def main():
    parser = argparse.ArgumentParser(
        description="CLI Demo for Robust-R1: Visual Question Answering with Degradation-Aware Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py "What type of vehicles are the people riding?"
    python demo.py "What is in the image?" --temperature 0.7 --max-tokens 2048
    python demo.py "Your question" --image /path/to/image.jpg
        """
    )
    
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask about the image"
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=FIXED_IMAGE_PATH,
        help=f"Path to the input image (default: {FIXED_IMAGE_PATH})"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"Model path or HuggingFace model name (default: {MODEL_PATH}). Can also be set via MODEL_PATH environment variable."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file does not exist: {args.image}")
        sys.exit(1)
    
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image}")
    print(f"Question: {args.question}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print("-" * 80)
    
    model_handler = ModelHandler(args.model_path)
    
    try:
        response = model_handler.predict(
            question=args.question,
            image_path=args.image,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print("\n" + "=" * 80)
        print("Model Response:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
