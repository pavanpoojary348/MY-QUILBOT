from datasets import load_dataset
from collections import Counter

dataset = load_dataset("humarin/chatgpt-paraphrases")
train = dataset["train"]

categories = Counter(train["category"])
sources = Counter(train["source"])

print("=== Categories ===")
for cat, count in categories.most_common():
    print(f"{cat}: {count}")

print("\n=== Sources ===")
for src, count in sources.most_common():
    print(f"{src}: {count}")

# Pull a few examples from non-question categories, if any exist
print("\n=== Sample non-question examples ===")
count = 0
for ex in train:
    if ex["category"] != "question":
        print(f"[{ex['category']} / {ex['source']}] {ex['text']}")
        count += 1
    if count >= 8:
        break