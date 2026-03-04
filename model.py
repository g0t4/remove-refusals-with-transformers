import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.inference_mode()

MODEL_ID = "tiiuae/Falcon3-1B-Instruct"
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
# MODEL_ID = "google/gemma-1.1-2b-it"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

trust_remote_code = False
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=trust_remote_code,
    dtype=torch.float16,
    device_map="cuda",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=trust_remote_code)
