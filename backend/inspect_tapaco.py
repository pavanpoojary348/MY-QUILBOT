from datasets import load_dataset
import itertools
import random

dataset = load_dataset("community-datasets/tapaco", "en")["train"]

# Group sentences by paraphrase_set_id
clusters = {}
for row in dataset:
    cid = row["paraphrase_set_id"]
    clusters.setdefault(cid, []).append(row["paraphrase"])

print(f"Total clusters: {len(clusters)}")

def word_overlap(a, b):
    w1 = set(a.lower().split())
    w2 = set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

pairs = []
for cid, sentences in clusters.items():
    if len(sentences) < 2:
        continue
    for a, b in itertools.combinations(sentences, 2):
        overlap = word_overlap(a, b)
        if 0.15 <= overlap <= 0.45:
            pairs.append((a, b, overlap))

print(f"Total filtered pairs: {len(pairs)}")
print()
print("Random sample of 15 pairs:")
for a, b, overlap in random.sample(pairs, 15):
    print(f"  [{overlap:.2f}]  {a!r}  <->  {b!r}")