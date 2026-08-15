from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v2"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Question-style sentences, matching the training data's domain
test_sentences = [
    "How do I lose weight quickly?",
    "What is the best way to learn Python?",
    "Why is my internet connection so slow?",
    "How can I improve my English speaking skills?",
    "What are some good books to read this year?",
]

for sentence in test_sentences:
    input_text = "paraphrase: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Original:   {sentence}")
    print(f"Paraphrase: {output_text}")
    print("---")