from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from eval_set import EVAL_SENTENCES

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()

def load_model(path):
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)
    return tokenizer, model

def paraphrase(sentence, tokenizer, model, num_candidates=8):
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

    valid = [c for c in candidates if c[2] >= 0.75 and c[1] <= 0.6]
    if valid:
        return min(valid, key=lambda c: c[1])
    return sorted(candidates, key=lambda c: c[1])[0]

print("Loading v3...")
tok3, model3 = load_model("../models/paraphraser-v3")
print("Loading v4...")
tok4, model4 = load_model("../models/paraphraser-v4")

for i, sentence in enumerate(EVAL_SENTENCES, 1):
    text3, ov3, sim3 = paraphrase(sentence, tok3, model3)
    text4, ov4, sim4 = paraphrase(sentence, tok4, model4)

    print(f"\n[{i}] Original: {sentence}")
    print(f"    v3: {text3}  (overlap {ov3:.2f}, sim {sim3:.2f})")
    print(f"    v4: {text4}  (overlap {ov4:.2f}, sim {sim4:.2f})")