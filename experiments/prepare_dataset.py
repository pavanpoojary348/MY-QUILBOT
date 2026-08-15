from datasets import load_dataset, DatasetDict


print("Loading PAWS dataset...")

dataset = load_dataset(
    "google-research-datasets/paws",
    "labeled_final"
)

# Keep only genuine paraphrase pairs
train = dataset["train"].filter(
    lambda example: example["label"] == 1
)

validation = dataset["validation"].filter(
    lambda example: example["label"] == 1
)

test = dataset["test"].filter(
    lambda example: example["label"] == 1
)

print("\nAvailable paraphrase pairs:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# Select a small subset for our first experiment
train = train.select(range(min(5000, len(train))))
validation = validation.select(range(min(500, len(validation))))
test = test.select(range(min(500, len(test))))


# Keep only the columns we need
train = train.select_columns(["sentence1", "sentence2"])
validation = validation.select_columns(["sentence1", "sentence2"])
test = test.select_columns(["sentence1", "sentence2"])


final_dataset = DatasetDict({
    "train": train,
    "validation": validation,
    "test": test
})


print("\nOur training dataset:")
print(final_dataset)

print("\nExample:")
print(final_dataset["train"][0])


# Save locally
final_dataset.save_to_disk("../training_data")


print("\nDataset saved successfully!")