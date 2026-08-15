from datasets import load_dataset, Dataset

def word_overlap(q1, q2):
    w1, w2 = set(q1.lower().split()), set(q2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

MIN_THRESHOLD = 0.15   # below this: likely unrelated, not a paraphrase
MAX_THRESHOLD = 0.45   # above this: likely a near-copy

def in_band(q1, q2):
    score = word_overlap(q1, q2)
    return MIN_THRESHOLD <= score <= MAX_THRESHOLD

# --- QQP only for now (PAWS structurally unsuited, dropped) ---
qqp = load_dataset("nyu-mll/glue", "qqp")["train"]
qqp_pairs = []
for ex in qqp:
    if ex["label"] == 1 and in_band(ex["question1"], ex["question2"]):
        qqp_pairs.append({"sentence1": ex["question1"], "sentence2": ex["question2"]})
    if len(qqp_pairs) >= 6000:
        break

print(f"QQP filtered pairs (band {MIN_THRESHOLD}-{MAX_THRESHOLD}): {len(qqp_pairs)}")

dataset = Dataset.from_list(qqp_pairs).shuffle(seed=42)

print("\nSample:")
for ex in dataset.select(range(10)):
    print(f"S1: {ex['sentence1']}")
    print(f"S2: {ex['sentence2']}")
    print("---")

dataset.save_to_disk("../training_data_v2")
print("\nSaved to training_data_v2/")