import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


text = "I am very happy because I got a new job."

prompt = f"paraphrase: {text}"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    max_length=128,
    truncation=True
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        do_sample=False
    )

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nOriginal:")
print(text)

print("\nParaphrased:")
print(result)