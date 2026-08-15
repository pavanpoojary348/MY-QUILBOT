from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset(
    "google-research-datasets/paws",
    "labeled_final"
)

train = dataset["train"]

print("\nTotal training examples:", len(train))

# Keep only actual paraphrase pairs
paraphrases = train.filter(lambda example: example["label"] == 1)

print("Actual paraphrase pairs:", len(paraphrases))

print("\nFirst 5 paraphrase examples:\n")

for i in range(5):
    example = paraphrases[i]

    print(f"Example {i + 1}")
    print("Original :", example["sentence1"])
    print("Paraphrase:", example["sentence2"])
    print()