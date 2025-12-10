import gradio as gr
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import html

sys_prompt = """First output the the types of degradations in image briefly in <TYPE> <TYPE_END> tags, 
        and then output what effects do these degradation have on the image in <INFLUENCE> <INFLUENCE_END> tags, 
        then based on the strength of degradation, output an APPROPRIATE length for the reasoning process in <REASONING> <REASONING_END> tags, 
        and then summarize the content of reasoning and the give the answer in <CONCLUSION> <CONCLUSION_END> tags,
        provides the user with the answer briefly in <ANSWER> <ANSWER_END>."""

project_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(project_dir, ".gradio_temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = temp_dir

MODEL_PATH = os.getenv("MODEL_PATH", "")

if not MODEL_PATH:
    raise ValueError("MODEL_PATH environment variable must be set. Please set it to your model path.")

print(f"==========================================")
print(f"Initializing application...")
print(f"==========================================")

class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"⏳ Loading model weights, this may take a few minutes...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e

    def predict(self, message_dict, history, temperature, max_tokens):
        text = message_dict.get("text", "")
        files = message_dict.get("files", [])

        messages = []
        
        if history:
            print(f"Processing {len(history)} previous messages from history")
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    user_content = []
                    
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                if os.path.exists(item) or any(item.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                                    user_content.append({"type": "image", "image": item})
                                else:
                                    user_content.append({"type": "text", "text": item})
                            elif isinstance(item, dict):
                                user_content.append(item)
                    elif isinstance(content, str):
                        if content:
                            user_content.append({"type": "text", "text": content})
                    
                    if user_content:
                        messages.append({"role": "user", "content": user_content})
                        
                elif role == "assistant":
                    if isinstance(content, str) and content:
                        messages.append({"role": "assistant", "content": content})
        
        current_content = []
        if files:
            for file_path in files:
                current_content.append({"type": "image", "image": file_path})
        
        if text:
            sys_prompt_formatted = " ".join(sys_prompt.split())
            full_text = f"{text}\n{sys_prompt_formatted}"
            current_content.append({"type": "text", "text": full_text})
        
        if current_content:
            messages.append({"role": "user", "content": current_content})
        
        print(f"Total messages for model: {len(messages)}")
        print(f"Message roles: {[m['role'] for m in messages]}")

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
            print("Starting model generation...")
            with torch.no_grad():
                generated_ids = self.model.generate(**generation_kwargs)
            
            input_length = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[0][input_length:]
            
            print(f"Input length: {input_length}, Generated token count: {len(generated_ids)}")
            
            generated_text = self.processor.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            print(f"Generation completed. Output length: {len(generated_text)}, Content preview: {repr(generated_text[:200])}")
            
            if generated_text and generated_text.strip():
                print(f"Yielding generated text: {generated_text[:100]}...")
                yield generated_text
            else:
                warning_msg = "⚠️ No output generated. The model may not have produced any response."
                print(warning_msg)
                yield warning_msg
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in model.generate: {error_details}")
            yield f"❌ Generation error: {str(e)}"
            return

model_handler = ModelHandler(MODEL_PATH)

def create_chat_ui():
    custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    #chatbot { height: 650px !important; overflow-y: auto; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Robust-R1") as demo:
        
        with gr.Row():
            gr.Markdown("# 🤖Robust-R1:Degradation-Aware Reasoning for Robust Visual Understanding")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Chat",
                    type="messages",
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Qwen"),
                    height=650
                )
                
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_types=["image"],
                    placeholder="Enter your question or upload an image...",
                    show_label=False
                )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Config")
                    temperature = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.6, step=0.05, 
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=128, maximum=4096, value=1024, step=128, 
                        label="Max New Tokens"
                    )
                
                clear_btn = gr.Button("🗑️ Clear Context", variant="stop")

        gr.Markdown("---")
        gr.Markdown("### 📚 Examples")
        gr.Markdown("Click the examples below to quickly fill the input box and start a conversation")
        
        example_images_dir = os.path.join(project_dir, "assets")
        
        examples_config = [
            ("What type of vehicles are the people riding?\n0. trucks\n1. wagons\n2. jeeps\n3. cars\n", os.path.join(example_images_dir, "92.jpg")),
            ("What is the giant fish in the air?\n0. blimp\n1. balloon\n2. kite\n3. sculpture\n", os.path.join(example_images_dir, "568.jpg")),
        ]
        
        example_data = []
        for text, img_path in examples_config:
            if os.path.exists(img_path):
                example_data.append({"text": text, "files": [img_path]})
        
        if example_data:
            gr.Examples(
                examples=example_data,
                inputs=chat_input,
                label="",
                examples_per_page=3
            )
        else:
            gr.Markdown("*No example images available, please manually upload images for testing*")
        
        async def respond(user_msg, history, temp, tokens):
            text = user_msg.get("text", "").strip()
            files = user_msg.get("files", [])
            user_content = list(files)
            if text: user_content.append(text)
            
            if not files and text: user_message = {"role": "user", "content": text}
            else: user_message = {"role": "user", "content": user_content}
            
            history.append(user_message)
            yield history, gr.MultimodalTextbox(value=None, interactive=False)

            history.append({"role": "assistant", "content": ""})
            
            try:
                previous_history = history[:-2] if len(history) >= 2 else []
                
                generated_text = ""
                for chunk in model_handler.predict(user_msg, previous_history, temp, tokens):
                    generated_text = chunk
                    
                    safe_text = html.escape(generated_text)
                    safe_text = generated_text.replace("<", "&lt;").replace(">", "&gt;")
                    
                    history[-1]["content"] = safe_text
                    yield history, gr.MultimodalTextbox(interactive=False)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                history[-1]["content"] = f"❌ Inference error: {str(e)}"
                yield history, gr.MultimodalTextbox(interactive=True)
            
            yield history, gr.MultimodalTextbox(value=None, interactive=True)
            
        chat_input.submit(
            respond,
            inputs=[chat_input, chatbot, temperature, max_tokens],
            outputs=[chatbot, chat_input]
        )

        def clear_history(): return [], None
        clear_btn.click(clear_history, outputs=[chatbot, chat_input])

    return demo

if __name__ == "__main__":
    demo = create_chat_ui()
    
    print(f"🚀 Service is starting, please visit: http://localhost:7862")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        allowed_paths=[project_dir]
    )
