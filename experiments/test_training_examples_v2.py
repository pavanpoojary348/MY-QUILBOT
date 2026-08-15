from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v2"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

dataset = load_from_disk("../training_data_v2_split")
train = dataset["train"]

for i in range(10):
    example = train[i]
    original = example["sentence1"]
    expected = example["sentence2"]

    input_text = "paraphrase: " + original
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Original: {original}")
    print(f"Expected: {expected}")
    print(f"Model:    {output_text}")
    print("---")