from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk


MODEL_PATH = "../models/paraphraser-v1"
DATASET_PATH = "../training_data"


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

print("Loading dataset...")

dataset = load_from_disk(DATASET_PATH)


for i in range(10):

    original = dataset["train"][i]["sentence1"]
    target = dataset["train"][i]["sentence2"]

    prompt = "paraphrase: " + original

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=8,
        do_sample=False
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\n======================================")
    print("EXAMPLE", i + 1)

    print("\nOriginal:")
    print(original)

    print("\nExpected paraphrase:")
    print(target)

    print("\nOur model:")
    print(result)