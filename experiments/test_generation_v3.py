from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v3"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

test_sentences = [
    "I am very happy because I got a new job.",
    "How do I lose weight quickly?",
    "What is the best way to learn Python?",
    "The weather is very nice today.",
    "Why is my internet connection so slow?",
]

def generate(sentence, **kwargs):
    input_text = "paraphrase: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128, **kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

for s in test_sentences:
    print(f"Original: {s}")
    print(f"  Beam search (baseline):   {generate(s, num_beams=4)}")
    print(f"  Diverse beam search:      {generate(s, num_beams=4, num_beam_groups=4, diversity_penalty=1.0)}")
    print(f"  Sampling (top_p) x3:")
    for i in range(3):
        print(f"    {generate(s, do_sample=True, top_p=0.9, temperature=1.2)}")
    print("---")