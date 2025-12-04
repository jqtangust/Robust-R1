import os
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = os.getenv("MODEL_PATH", "Jiaqi-hkust/Robust-R1")

model = None
processor = None

def load_model():
    global model, processor
    if model is None or processor is None:
        print(f"Loading model: {MODEL_PATH}")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return model, processor

def inference(image, question, max_new_tokens, temperature):
    try:
        model, processor = load_model()
        if image is None: return "⚠️ Error: Please upload an image."
        if not question or question.strip() == "": return "⚠️ Error: Please enter your question."
        
        sys_prompt = """First output the the types of degradations in image briefly in <TYPE> <TYPE_END> tags, 
        and then output what effects do these degradation have on the image in <INFLUENCE> <INFLUENCE_END> tags, 
        then based on the strength of degradation, output an APPROPRIATE length for the reasoning process in <REASONING> <REASONING_END> tags, 
        and then summarize the content of reasoning and the give the answer in <CONCLUSION> <CONCLUSION_END> tags,
        provides the user with the answer briefly in <ANSWER> <ANSWER_END>."""

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{question}\n\n{sys_prompt}"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True if temperature > 0 else False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]
    except Exception as e:
        return f"An error occurred: {str(e)}"

custom_css = """
body, .gradio-container, .prose {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.header-title {
    font-weight: 700; 
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #fff, #ccc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-box textarea, .result-box .wrap {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; 
    font-size: 14px !important;
    line-height: 1.7 !important;
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 16px !important;
    padding: 16px !important;
    box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.05);
}

.input-panel .block, .input-panel textarea, .input-panel .input-image {
     border-radius: 12px !important;
}

.primary-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    border: none !important;
    border-radius: 12px !important; 
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
}
"""

theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="zinc",
    neutral_hue="zinc",
    spacing_size="md",
    radius_size="lg",
    font=("Inter", "sans-serif"),
    font_mono=("JetBrains Mono", "monospace"),
).set(
    body_background_fill="#121212",   
    block_background_fill="#1e1e20",  
    block_border_color="#2e2e32",     
    input_background_fill="#27272a",  
    block_radius="16px",         
    container_radius="16px",     
    input_radius="12px",         
    button_large_radius="12px",  
    block_border_width="1px",
    body_text_color="#e4e4e7",
    block_title_text_color="black",
    input_placeholder_color="#71717a",
)

with gr.Blocks(title="Robust-R1", css=custom_css, theme=theme) as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="text-align: center; padding: 10px 20px 10px;"> 
                <h1 class="header-title" style="font-size: 2.8rem; margin-bottom: 0.5rem;">Robust-R1</h1>
                <p style="color: #a1a1aa; font-size: 1.2em; margin-bottom: 12px;">Degradation-Aware Reasoning for Robust Visual Understanding</p>
                
                <div style="display: flex; justify-content: center; align-items: center; gap: 8px;">
                    <a href="#" target="_blank" style="text-decoration: none;">
                        <img src="https://img.shields.io/badge/cs.CV-Paper-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="Paper">
                    </a>
                    <a href="https://huggingface.co/Jiaqi-hkust/Robust-R1" target="_blank" style="text-decoration: none;">
                        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat" alt="Models">
                    </a>
                    <a href="https://huggingface.co/datasets/Jiaqi-hkust/Robust-R1" target="_blank" style="text-decoration: none;">
                        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-FFD21E?style=flat" alt="Data">
                    </a>
                </div>
            </div>
            """)

    with gr.Row(equal_height=False, variant="default"): 
        
        with gr.Column(scale=2, elem_classes="input-panel"):
            
            image_input = gr.Image(
                type="pil",
                label="Image Input",
                height=320,
                sources=["upload", "clipboard"],
                show_download_button=False,
            )
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about the image...",
                lines=3,
                value="What do you see in this image?"
            )
            
            with gr.Accordion("⚙️ Generation Parameters", open=False):
                max_tokens = gr.Slider(64, 2048, value=1024, step=64, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")

            with gr.Row(): 
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                submit_btn = gr.Button("Run Analysis", variant="primary", size="lg", elem_classes="primary-btn")

        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 🧠 Model Analysis Result", elem_id="output-header")
                output = gr.Textbox(
                    label="", 
                    show_label=False,
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Analysis results will appear here...",
                    elem_classes="result-box"
                )

    submit_btn.click(
        fn=inference,
        inputs=[image_input, question_input, max_tokens, temperature],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: (None, "", 1024, 0.7, ""),
        outputs=[image_input, question_input, max_tokens, temperature, output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
