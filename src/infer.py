#!/usr/bin/env python3
"""
Single-frame inference script.
Loads fine-tuned model and classifies a single image as SAFE or UNSAFE.
"""

import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel

MODEL_DIR = Path("/workspace/outputs")
BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

PROMPT = "You are a workplace safety inspector reviewing CCTV footage. Classify the behavior as SAFE or UNSAFE. Answer with only one word."


def load_model():
    """Load fine-tuned model and processor."""
    print(f"Loading base model: {BASE_MODEL}")
    
    processor = Qwen2VLProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    
    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load LoRA weights if they exist
    if (MODEL_DIR / "adapter_config.json").exists():
        print(f"Loading LoRA adapter from {MODEL_DIR}...")
        model = PeftModel.from_pretrained(model, MODEL_DIR)
        model = model.merge_and_unload()  # Merge for inference
    else:
        print("No LoRA adapter found, using base model")
    
    model.eval()
    return model, processor


def classify_image(image_path, model, processor):
    """Classify a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Extract SAFE or UNSAFE
    response = response.strip().upper()
    if "UNSAFE" in response:
        return "UNSAFE"
    elif "SAFE" in response:
        return "SAFE"
    else:
        return f"UNCLEAR: {response}"


def main():
    """Main inference pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
        print("Example: python infer.py /workspace/data/frames/test/safe/video_001/0001.jpg")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("VLM Safety Classification Inference")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (slow)")
    else:
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, processor = load_model()
    
    # Classify
    print(f"\nClassifying: {image_path}")
    result = classify_image(image_path, model, processor)
    
    print(f"\nResult: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
