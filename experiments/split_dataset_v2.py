from datasets import load_from_disk, DatasetDict

dataset = load_from_disk("../training_data_v2")
dataset = dataset.shuffle(seed=42)

train = dataset.select(range(5000))
validation = dataset.select(range(5000, 5500))
test = dataset.select(range(5500, 6000))

final_dataset = DatasetDict({
    "train": train,
    "validation": validation,
    "test": test
})

print(final_dataset)

final_dataset.save_to_disk("../training_data_v2_split")
print("\nSaved to training_data_v2_split/")