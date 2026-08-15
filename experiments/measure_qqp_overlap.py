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
    return len(intersection) / len(union)  # Jaccard similarity

# Only look at real paraphrase pairs (label == 1)
scores = []
count = 0
for example in train:
    if example["label"] == 1:
        score = word_overlap(example["question1"], example["question2"])
        scores.append(score)
        count += 1
    if count >= 2000:  # sample, not the full 150k+ positive pairs
        break

scores.sort()
print(f"Sampled {len(scores)} positive pairs")
print(f"Min: {scores[0]:.2f}")
print(f"25th percentile: {scores[len(scores)//4]:.2f}")
print(f"Median: {scores[len(scores)//2]:.2f}")
print(f"75th percentile: {scores[3*len(scores)//4]:.2f}")
print(f"Max: {scores[-1]:.2f}")

# Show a few examples near different overlap levels
print("\n--- Low overlap examples (good paraphrases) ---")
for s in scores[:5]:
    pass  # we'll print actual pairs next step once we pick a threshold