from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from eval_set import EVAL_SENTENCES
import re


# ============================================================
# 1. LOAD MODELS
# ============================================================

MODEL_PATH = "../models/paraphraser-v6"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Models loaded successfully.")


# ============================================================
# 2. WORD OVERLAP
# ============================================================

def word_overlap(a, b):
    w1 = set(a.lower().split())
    w2 = set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ============================================================
# 3. SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


# ============================================================
# 4. GENERATE PARAPHRASE CANDIDATES
# ============================================================

def generate_candidates(sentence, num_candidates=8):
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
    return candidates


# ============================================================
# 5. ENTITY GROUNDING CHECK
# ============================================================

def introduces_new_entities(original, candidate):
    """
    Returns True if the candidate contains capitalized words
    (proper nouns) or numbers that don't appear in the original.
    """
    def extract_entities(text):
        words = text.split()
        proper_nouns = set()
        for i, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if i > 0 and clean and clean[0].isupper():
                proper_nouns.add(clean.lower())
        numbers = set(re.findall(r'\d+', text))
        return proper_nouns, numbers

    orig_nouns, orig_nums = extract_entities(original)
    cand_nouns, cand_nums = extract_entities(candidate)
    new_nouns = cand_nouns - orig_nouns
    new_nums = cand_nums - orig_nums
    return len(new_nouns) > 0 or len(new_nums) > 0


# ============================================================
# 6. SELECT BEST CANDIDATE
# ============================================================

def pick_best(candidates, original, similarity_min=0.85, overlap_min=0.10, overlap_max=0.55):
    valid = [
        c for c in candidates
        if c[2] >= similarity_min
        and overlap_min <= c[1] <= overlap_max
        and not introduces_new_entities(original, c[0])
    ]
    if valid:
        return min(valid, key=lambda c: c[1])
    return None


# ============================================================
# 7. GENERATE ALL CANDIDATES
# ============================================================

print("\nGenerating candidates for all sentences...")
all_candidates = {s: generate_candidates(s) for s in EVAL_SENTENCES}


# ============================================================
# 8. CONFIDENCE TEST
# ============================================================

for threshold in [0.85]:
    confident = 0
    print(f"\n=== SIMILARITY_MIN = {threshold} + entity grounding ===")
    for sentence in EVAL_SENTENCES:
        result = pick_best(all_candidates[sentence], sentence, threshold)
        if result:
            confident += 1
    print(f"Confident on {confident}/{len(EVAL_SENTENCES)} ({100*confident/len(EVAL_SENTENCES):.0f}%)")


# ============================================================
# 9. FULL RESULTS
# ============================================================

print("\n=== Full results at SIMILARITY_MIN = 0.85 + entity grounding ===\n")

confident = 0
for i, sentence in enumerate(EVAL_SENTENCES, 1):
    result = pick_best(all_candidates[sentence], sentence, 0.85)
    if result:
        confident += 1
        text, overlap, similarity = result
        print(f"[{i}] CONFIDENT")
        print(f"    Original:   {sentence}")
        print(f"    Output:     {text}")
        print(f"    Overlap:    {overlap:.3f}")
        print(f"    Similarity: {similarity:.3f}")
    else:
        print(f"[{i}] FALLBACK")
        print(f"    Original:   {sentence}")
    print()

print(f"Confident on {confident}/{len(EVAL_SENTENCES)} ({100*confident/len(EVAL_SENTENCES):.0f}%)")