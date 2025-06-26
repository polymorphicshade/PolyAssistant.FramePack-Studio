import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# Using a smaller, faster model for this feature.
# This can be moved to a settings file later.
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = """You are a creative assistant for a text-to-video generator. Your task is to take a user's prompt and make it more descriptive, and detailed but still concise. Focus on visual elements. Do not change the core action.

User prompt: "{text_to_enhance}"

Enhanced prompt:"""

# --- Model Loading (cached) ---
model = None
tokenizer = None

def _load_model():
    """Loads the model and tokenizer, caching them globally."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"LLM Enhancer: Loading model '{MODEL_NAME}' to {DEVICE}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        print("LLM Enhancer: Model loaded successfully.")

def _run_inference(text_to_enhance: str) -> str:
    """Runs the LLM inference to enhance a single piece of text."""
    _load_model() # Ensure model is loaded

    formatted_prompt = PROMPT_TEMPLATE.format(text_to_enhance=text_to_enhance)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.95,
        top_k=30
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Clean up the response
    response = response.strip().replace('"', '')
    return response


def enhance_prompt(prompt_text: str) -> str:
    """
    Enhances a prompt, handling both plain text and timestamped formats.
    
    Args:
        prompt_text: The user's input prompt.
        
    Returns:
        The enhanced prompt string.
    """
    if not prompt_text:
        return ""

    # Regex to find timestamp sections like [0s: text] or [1.1s-2.2s: text]
    timestamp_pattern = r'(\[\d+(?:\.\d+)?s(?:-\d+(?:\.\d+)?s)?\s*:\s*)(.*?)(?=\])'
    
    matches = list(re.finditer(timestamp_pattern, prompt_text))

    if not matches:
        # No timestamps found, enhance the whole prompt
        print("LLM Enhancer: Enhancing a simple prompt.")
        return _run_inference(prompt_text)
    else:
        # Timestamps found, enhance each section's text
        print(f"LLM Enhancer: Enhancing {len(matches)} sections in a timestamped prompt.")
        enhanced_parts = []
        last_end = 0
        
        for match in matches:
            # Add the part of the string before the current match (e.g., whitespace)
            enhanced_parts.append(prompt_text[last_end:match.start()])
            
            timestamp_prefix = match.group(1)
            text_to_enhance = match.group(2).strip()
            
            if text_to_enhance:
                enhanced_text = _run_inference(text_to_enhance)
                enhanced_parts.append(f"{timestamp_prefix}{enhanced_text}")
            else:
                # Keep empty sections as they are
                enhanced_parts.append(f"{timestamp_prefix}")

            last_end = match.end()

        # Add the closing bracket for the last match and any trailing text
        enhanced_parts.append(prompt_text[last_end:])

        return "".join(enhanced_parts)