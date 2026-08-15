from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "../models/paraphraser-v3"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def paraphrase(sentence, num_candidates=6):
    input_text = "paraphrase: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    candidates = []
    for _ in range(num_candidates):
        output_ids = model.generate(
            input_ids, max_length=128,
            do_sample=True, top_p=0.9, temperature=1.2,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        overlap = word_overlap(sentence, text)
        candidates.append((text, overlap))

    # Prefer candidates that are different but not nonsensically different:
    # sort by overlap ascending, but skip ones with overlap == 0 (likely garbage/unrelated)
    scored = sorted(candidates, key=lambda c: c[1])
    for text, overlap in scored:
        if overlap > 0.05:  # not empty/unrelated
            return text, overlap
    return scored[0]  # fallback

test_sentences = [
    "I am very happy because I got a new job.",
    "How do I lose weight quickly?",
    "What is the best way to learn Python?",
    "The weather is very nice today.",
    "Why is my internet connection so slow?",
]

for s in test_sentences:
    result, overlap = paraphrase(s)
    print(f"Original:   {s}")
    print(f"Paraphrase: {result}  (overlap: {overlap:.2f})")
    print("---")