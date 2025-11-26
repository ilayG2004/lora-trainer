import os
import torch
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
load_dotenv()

BASE_MODEL = os.environ.get("MODEL_NAME")           # same base model used in training
LORA_WEIGHTS = os.environ.get("ARTIFACT_DIR")       # directory with adapter_model.safetensors

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)

model.eval()

def generate(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Try your model
print(generate("composer: Chinzer\nmusical_form: concerto\ninstruments: cello, viola, violin\nperiod: Late Baroque\n\nmovement 1:\n  name: allegro\n  key: re\n  scale: major\n  tempo: 2 = 45\n  time: 4/4\n\nmovement 2:\n  name: andante\n  key: sol\n  scale: major\n  tempo: 4 = 55\n  time: 4/4\n\nmovement 3:\n  name: allegro\n  key: re\n  scale: major\n  tempo: 4. = 50\n  time: 3/8"))
