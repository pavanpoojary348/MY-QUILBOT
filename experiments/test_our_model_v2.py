from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v2"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

test_sentences = [
    "I am very happy because I got a new job.",
    "The weather is very nice today.",
    "I love learning new programming languages.",
    "She completed her assignment yesterday.",
    "Artificial intelligence is changing the world.",
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