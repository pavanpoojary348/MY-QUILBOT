from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2TokenizerFast
)
from sentence_transformers import SentenceTransformer, util
from eval_set import EVAL_SENTENCES
import torch
import re


# ============================================================
# 1. LOAD MODELS
# ============================================================

MODEL_PATH = "../models/paraphraser-v4"

# Paraphrasing model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Semantic similarity model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# GPT-2 fluency model
print("Loading GPT-2 fluency model...")

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

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
    emb = embedder.encode(
        [a, b],
        convert_to_tensor=True
    )

    return util.cos_sim(
        emb[0],
        emb[1]
    ).item()


# ============================================================
# 4. GPT-2 PERPLEXITY / FLUENCY
# ============================================================

def perplexity(text):
    """
    Lower perplexity = generally more fluent/natural.
    Higher perplexity = generally more unusual/awkward.
    """

    input_ids = gpt2_tokenizer.encode(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = gpt2_model(
            input_ids,
            labels=input_ids
        )

    return torch.exp(outputs.loss).item()


# ============================================================
# 5. GENERATE PARAPHRASE CANDIDATES
# ============================================================

def generate_candidates(sentence, num_candidates=8):

    input_text = "paraphrase: " + sentence

    input_ids = tokenizer(
        input_text,
        return_tensors="pt"
    ).input_ids

    candidates = []

    for _ in range(num_candidates):

        output_ids = model.generate(
            input_ids,
            max_length=128,
            do_sample=True,
            top_p=0.9,
            temperature=1.2,
        )

        text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Word overlap
        overlap = word_overlap(
            sentence,
            text
        )

        # Semantic similarity
        similarity = semantic_similarity(
            sentence,
            text
        )

        # Candidate perplexity
        fluency = perplexity(text)

        candidates.append(
            (
                text,
                overlap,
                similarity,
                fluency
            )
        )

    return candidates


# ============================================================
# 6. ENTITY GROUNDING CHECK
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

            clean = re.sub(
                r'[^\w]',
                '',
                w
            )

            # Ignore first word because it is normally
            # capitalized simply because it starts the sentence.
            if (
                i > 0
                and clean
                and clean[0].isupper()
            ):
                proper_nouns.add(
                    clean.lower()
                )

        numbers = set(
            re.findall(
                r'\d+',
                text
            )
        )

        return proper_nouns, numbers

    orig_nouns, orig_nums = extract_entities(
        original
    )

    cand_nouns, cand_nums = extract_entities(
        candidate
    )

    new_nouns = cand_nouns - orig_nouns
    new_nums = cand_nums - orig_nums

    return (
        len(new_nouns) > 0
        or len(new_nums) > 0
    )


# ============================================================
# 7. SELECT BEST CANDIDATE
# ============================================================

def pick_best(
    candidates,
    original,
    original_ppl,
    similarity_min=0.85,
    overlap_min=0.10,
    overlap_max=0.55,
    fluency_ratio_max=1.8
):

    valid = [
        c for c in candidates

        # Semantic similarity
        if c[2] >= similarity_min

        # Word overlap
        and overlap_min <= c[1] <= overlap_max

        # Entity grounding
        and not introduces_new_entities(
            original,
            c[0]
        )

        # Relative fluency
        and (c[3] / original_ppl) <= fluency_ratio_max
    ]

    if valid:

        # Select candidate with lowest
        # word overlap among valid candidates.
        return min(
            valid,
            key=lambda c: c[1]
        )

    return None


# ============================================================
# 8. GENERATE ALL CANDIDATES
# ============================================================

print(
    "\nGenerating candidates for all sentences..."
)

all_candidates = {
    s: generate_candidates(s)
    for s in EVAL_SENTENCES
}


# ============================================================
# 9. CALCULATE ORIGINAL SENTENCE PERPLEXITY
# ============================================================

print(
    "\nCalculating original sentence perplexities..."
)

original_ppl = {
    s: perplexity(s)
    for s in EVAL_SENTENCES
}

print("Original perplexities calculated.")


# ============================================================
# 10. CONFIDENCE TEST
# ============================================================

for threshold in [0.85]:

    confident = 0

    print(
        f"\n=== SIMILARITY_MIN = {threshold} "
        f"+ entity grounding + fluency ratio ==="
    )

    for sentence in EVAL_SENTENCES:

        result = pick_best(
            all_candidates[sentence],
            sentence,
            original_ppl[sentence],
            threshold,
            fluency_ratio_max=1.8
        )

        if result:
            confident += 1

    print(
        f"Confident on "
        f"{confident}/{len(EVAL_SENTENCES)} "
        f"({100 * confident / len(EVAL_SENTENCES):.0f}%)"
    )


# ============================================================
# 11. FULL RESULTS
# ============================================================

print(
    "\n=== Full results at "
    "SIMILARITY_MIN = 0.85 "
    "+ entity grounding "
    "+ fluency ratio <= 1.8 ===\n"
)

confident = 0

for i, sentence in enumerate(
    EVAL_SENTENCES,
    1
):

    result = pick_best(
        all_candidates[sentence],
        sentence,
        original_ppl[sentence],
        0.85,
        fluency_ratio_max=1.8
    )

    if result:

        confident += 1

        text, overlap, similarity, fluency = result

        ratio = fluency / original_ppl[sentence]

        print(
            f"[{i}] CONFIDENT"
        )

        print(
            f"    Original:     {sentence}"
        )

        print(
            f"    Output:       {text}"
        )

        print(
            f"    Overlap:      {overlap:.3f}"
        )

        print(
            f"    Similarity:   {similarity:.3f}"
        )

        print(
            f"    Original PPL: {original_ppl[sentence]:.2f}"
        )

        print(
            f"    Candidate PPL:{fluency:.2f}"
        )

        print(
            f"    PPL Ratio:    {ratio:.2f}x"
        )

    else:

        print(
            f"[{i}] FALLBACK"
        )

        print(
            f"    Original:     {sentence}"
        )

        print(
            f"    Original PPL: {original_ppl[sentence]:.2f}"
        )

    print()


print(
    f"Confident on "
    f"{confident}/{len(EVAL_SENTENCES)} "
    f"({100 * confident / len(EVAL_SENTENCES):.0f}%)"
)


# ============================================================
# 12. PRINT ALL CANDIDATES
# ============================================================

print(
    "\n\n============================================================"
)

print(
    "ALL CANDIDATES WITH FLUENCY / PERPLEXITY RATIOS"
)

print(
    "============================================================\n"
)

for i, sentence in enumerate(
    EVAL_SENTENCES,
    1
):

    print(
        f"\n[{i}] Original: {sentence}"
    )

    print(
        f"Original PPL: {original_ppl[sentence]:.2f}"
    )

    print("-" * 90)

    for (
        text,
        overlap,
        similarity,
        fluency
    ) in all_candidates[sentence]:

        ratio = (
            fluency /
            original_ppl[sentence]
        )

        status = (
            "PASS"
            if ratio <= 1.8
            else "REJECT"
        )

        print(
            f"[ov={overlap:.2f} "
            f"sim={similarity:.2f} "
            f"ppl={fluency:.1f} "
            f"ratio={ratio:.2f}x "
            f"{status}] "
            f"{text}"
        )