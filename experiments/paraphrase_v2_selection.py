from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = "../models/paraphraser-v4"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()

def paraphrase(sentence, num_candidates=8):
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
        similarity = semantic_similarity(sentence, text)
        candidates.append((text, overlap, similarity))

    # Require: meaning preserved (high similarity) AND wording changed (low-ish overlap)
    valid = [c for c in candidates if c[2] >= 0.75 and c[1] <= 0.6]
    if valid:
        # Among valid ones, prefer the most reworded (lowest overlap)
        best = min(valid, key=lambda c: c[1])
        return best
    # No candidate passed the meaning-preservation bar
    return None

test_sentences = [
    "I am very happy because I got a new job.",
    "How do I lose weight quickly?",
    "What is the best way to learn Python?",
    "The weather is very nice today.",
    "Why is my internet connection so slow?",
]

for s in test_sentences:
    result = paraphrase(s)
    print(f"Original: {s}")
    if result:
        text, overlap, sim = result
        print(f"Best:     {text}  (overlap: {overlap:.2f}, similarity: {sim:.2f})")
    else:
        print("Best:     [no candidate passed the meaning-preservation threshold]")
    print("---")