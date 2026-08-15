from datasets import load_dataset

# Load the QQP config of the GLUE benchmark (same data, modern format)
dataset = load_dataset("nyu-mll/glue", "qqp")

print(dataset)
print()

train = dataset["train"]

# GLUE QQP columns: question1, question2, label (1 = duplicate/paraphrase)
count = 0
for example in train:
    if example["label"] == 1:
        q1 = example["question1"]
        q2 = example["question2"]
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print("---")
        count += 1
    if count >= 10:
        break