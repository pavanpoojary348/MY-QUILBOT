from transformers import pipeline
from nltk.corpus import wordnet

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

def get_wordnet_candidates(word):
    candidates = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower() and ' ' not in name:  # BERT mask needs single tokens
                candidates.add(name.lower())
    return candidates

def get_contextual_synonyms(sentence, target_word, max_results=5):
    masked_sentence = re.sub(
        rf'\b{re.escape(target_word)}\b',
        '[MASK]',
        sentence,
        count=1,
        flags=re.IGNORECASE
    )

    candidates = get_wordnet_candidates(target_word)
    if not candidates:
        return []

    predictions = fill_mask(masked_sentence, top_k=100)
    bert_ranked_words = {p['token_str'].strip().lower(): p['score'] for p in predictions}

    scored = []
    for candidate in candidates:
        score = bert_ranked_words.get(candidate, 0.0)
        scored.append((candidate, score))

    scored.sort(key=lambda x: -x[1])
    # Only keep candidates BERT actually recognized as plausible in this context
    result = [w for w, s in scored[:max_results] if s > 0]
    return result


import re

test_cases = [
    ("I am very happy because I got a new job.", "happy"),
    ("The building is very big.", "big"),
    ("She is a nice person.", "nice"),
]

for sentence, word in test_cases:
    result = get_contextual_synonyms(sentence, word)
    print(f"Sentence: {sentence}")
    print(f"Word: {word} -> {result}")
    print()