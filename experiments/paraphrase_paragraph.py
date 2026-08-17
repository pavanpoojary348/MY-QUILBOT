from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import re

MODEL_PATH = "../models/paraphraser-v6"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# SENTENCE SPLITTING
# ============================================================

ABBREVIATIONS = [
    "Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "Prof",
    "vs", "etc", "e.g", "i.e", "U.S", "U.K", "U.N",
    "Inc", "Ltd", "Co", "St",
]

def split_sentences(text):
    protected = text
    for abbr in ABBREVIATIONS:
        protected = re.sub(rf'\b{re.escape(abbr)}\.', f'{abbr}<PERIOD>', protected)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected.strip())
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s]


# ============================================================
# EXISTING PER-SENTENCE PARAPHRASE LOGIC
# ============================================================

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()

def introduces_new_entities(original, candidate):
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
    return len(cand_nouns - orig_nouns) > 0 or len(cand_nums - orig_nums) > 0

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

def paraphrase_sentence(sentence, num_candidates=8):
    candidates = generate_candidates(sentence, num_candidates)
    result = pick_best(candidates, sentence)
    if result:
        return result[0], True
    return sentence, False


# ============================================================
# NEW: PARAGRAPH-LEVEL PARAPHRASE
# ============================================================

def paraphrase_paragraph(text):
    sentences = split_sentences(text)
    results = []
    for s in sentences:
        paraphrased, confident = paraphrase_sentence(s)
        results.append({
            "original": s,
            "paraphrased": paraphrased,
            "confident": confident
        })
    full_output = " ".join(r["paraphrased"] for r in results)
    return full_output, results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    test_paragraph = (
        "I am very happy because I got a new job. "
        "The weather is very nice today. "
        "What is the best way to learn Python? "
        "Artificial intelligence is changing the world."
    )

    full_output, details = paraphrase_paragraph(test_paragraph)

    print("=== PER-SENTENCE BREAKDOWN ===\n")
    for i, r in enumerate(details, 1):
        status = "CONFIDENT" if r["confident"] else "fallback"
        print(f"[{i}] {status}")
        print(f"    Original:    {r['original']}")
        print(f"    Paraphrased: {r['paraphrased']}")
        print()

    print("=== FULL REJOINED PARAGRAPH ===\n")
    print(full_output)