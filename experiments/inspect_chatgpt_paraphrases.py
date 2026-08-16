from datasets import load_dataset

dataset = load_dataset("humarin/chatgpt-paraphrases")
print(dataset)
print()

train = dataset["train"]
print("Columns:", train.column_names)
print()

for i in range(10):
    example = train[i]
    print(example)
    print("---")