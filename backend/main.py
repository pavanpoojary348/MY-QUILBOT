from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import re


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="AI Writing Assistant")


# ============================================================
# LOAD MODELS (once, at server startup — not per request)
# ============================================================

MODEL_PATH = "../models/paraphraser-v4"

print("Loading paraphrasing model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

print("Loading semantic similarity model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Models loaded successfully.")


# ============================================================
# HELPER FUNCTIONS (from paraphrase_final.py)
# ============================================================

def word_overlap(a, b):
    w1 = set(a.lower().split())
    w2 = set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


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


def paraphrase(sentence, num_candidates=8):
    """
    Returns (paraphrased_text, confident: bool).
    If confident is False, paraphrased_text is the original sentence —
    we do not return a risky/unverified rewording.
    """
    candidates = generate_candidates(sentence, num_candidates)
    result = pick_best(candidates, sentence)
    if result:
        return result[0], True
    return sentence, False


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "AI Writing Assistant backend is running"}


class ParaphraseRequest(BaseModel):
    text: str


class ParaphraseResponse(BaseModel):
    original: str
    paraphrased: str
    confident: bool


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase_endpoint(request: ParaphraseRequest):
    result, confident = paraphrase(request.text)
    return ParaphraseResponse(
        original=request.text,
        paraphrased=result,
        confident=confident,
    )