from datasets import load_dataset, Dataset, DatasetDict

def word_overlap(q1, q2):
    w1, w2 = set(q1.lower().split()), set(q2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

MIN_THRESHOLD = 0.15
MAX_THRESHOLD = 0.45

def in_band(q1, q2):
    score = word_overlap(q1, q2)
    return MIN_THRESHOLD <= score <= MAX_THRESHOLD

qqp = load_dataset("nyu-mll/glue", "qqp")["train"]
pairs = []
for ex in qqp:
    if ex["label"] == 1 and in_band(ex["question1"], ex["question2"]):
        pairs.append({"sentence1": ex["question1"], "sentence2": ex["question2"]})
    if len(pairs) >= 12000:  # 10k train + 1k val + 1k test
        break

print(f"Total filtered pairs: {len(pairs)}")

dataset = Dataset.from_list(pairs).shuffle(seed=42)

final = DatasetDict({
    "train": dataset.select(range(10000)),
    "validation": dataset.select(range(10000, 11000)),
    "test": dataset.select(range(11000, 12000)),
})

print(final)
final.save_to_disk("../training_data_v4_split")
print("\nSaved to training_data_v4_split/")