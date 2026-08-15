from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v3"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def paraphrase(sentence):
    input_text = "paraphrase: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("=== UNSEEN DECLARATIVE SENTENCES ===\n")
declarative = [
    "I am very happy because I got a new job.",
    "The weather is very nice today.",
    "I love learning new programming languages.",
    "She completed her assignment yesterday.",
    "Artificial intelligence is changing the world.",
]
for s in declarative:
    print(f"Original:   {s}")
    print(f"Paraphrase: {paraphrase(s)}")
    print("---")

print("\n=== UNSEEN QUESTIONS (in-domain) ===\n")
questions = [
    "How do I lose weight quickly?",
    "What is the best way to learn Python?",
    "Why is my internet connection so slow?",
    "How can I improve my English speaking skills?",
    "What are some good books to read this year?",
]
for s in questions:
    print(f"Original:   {s}")
    print(f"Paraphrase: {paraphrase(s)}")
    print("---")

print("\n=== TRAINING EXAMPLES (control) ===\n")
dataset = load_from_disk("../training_data_v2_split")
train = dataset["train"]
for i in range(10):
    original = train[i]["sentence1"]
    expected = train[i]["sentence2"]
    print(f"Original: {original}")
    print(f"Expected: {expected}")
    print(f"Model:    {paraphrase(original)}")
    print("---")