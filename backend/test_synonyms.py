from nltk.corpus import wordnet
from collections import defaultdict

def get_synonyms(word, max_results=8):
    scores = defaultdict(int)
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() == word.lower():
                continue
            scores[name] += lemma.count()

    # Sort by frequency, most common first
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [w for w, count in ranked[:max_results]]

test_words = ["happy", "quick", "big", "nice", "learn"]

for word in test_words:
    syns = get_synonyms(word)
    print(f"{word}: {syns}")