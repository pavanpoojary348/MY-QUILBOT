from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "../models/paraphraser-v1"


print("Loading our model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

print("Our model loaded successfully!")


sentences = [
    "I am very happy because I got a new job.",
    "The weather is very nice today.",
    "I love learning new programming languages.",
    "She completed her assignment yesterday.",
    "Artificial intelligence is changing the world."
]


for sentence in sentences:

    prompt = "paraphrase: " + sentence

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        num_return_sequences=1
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\n--------------------------------")
    print("Original:")
    print(sentence)

    print("\nParaphrase:")
    print(result)