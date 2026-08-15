from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "qqp")
train = dataset["train"]

def word_overlap(q1, q2):
    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

THRESHOLD = 0.5

kept = []
dropped = []

count = 0
for example in train:
    if example["label"] == 1:
        score = word_overlap(example["question1"], example["question2"])
        pair = (example["question1"], example["question2"], score)
        if score <= THRESHOLD:
            kept.append(pair)
        else:
            dropped.append(pair)
        count += 1
    if count >= 2000:  # same sample as before
        break

print(f"Kept: {len(kept)} / {count} ({100*len(kept)/count:.1f}%)")
print(f"Dropped: {len(dropped)} / {count} ({100*len(dropped)/count:.1f}%)")

print("\n--- Sample of KEPT pairs (good, should be varied) ---")
for q1, q2, s in kept[:8]:
    print(f"[{s:.2f}] Q1: {q1}")
    print(f"       Q2: {q2}")
    print("---")

print("\n--- Sample of DROPPED pairs (should look trivial/duplicate) ---")
for q1, q2, s in dropped[:8]:
    print(f"[{s:.2f}] Q1: {q1}")
    print(f"       Q2: {q2}")
    print("---")
