from datasets import load_dataset, Dataset, DatasetDict
import ast

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

MIN_THRESHOLD = 0.15
MAX_THRESHOLD = 0.45

def in_band(a, b):
    return MIN_THRESHOLD <= word_overlap(a, b) <= MAX_THRESHOLD

TARGET_PER_SOURCE = 15000

# --- QQP (questions) ---
qqp = load_dataset("nyu-mll/glue", "qqp")["train"]
qqp_pairs = []
for ex in qqp:
    if ex["label"] == 1 and in_band(ex["question1"], ex["question2"]):
        qqp_pairs.append({"sentence1": ex["question1"], "sentence2": ex["question2"]})
    if len(qqp_pairs) >= TARGET_PER_SOURCE:
        break
print(f"QQP filtered pairs: {len(qqp_pairs)}")

# --- Declarative (SQuAD/CNN news) ---
chatgpt_data = load_dataset("humarin/chatgpt-paraphrases")["train"]
declarative = chatgpt_data.filter(lambda ex: ex["category"] == "sentence")

decl_pairs = []
for ex in declarative:
    original = ex["text"]
    paraphrase_list = ast.literal_eval(ex["paraphrases"])
    first_paraphrase = paraphrase_list[0]
    if in_band(original, first_paraphrase):
        decl_pairs.append({"sentence1": original, "sentence2": first_paraphrase})
    if len(decl_pairs) >= TARGET_PER_SOURCE:
        break
print(f"Declarative filtered pairs: {len(decl_pairs)}")

# --- Combine, shuffle, split ---
all_pairs = qqp_pairs + decl_pairs
print(f"Total combined pairs: {len(all_pairs)}")

dataset = Dataset.from_list(all_pairs).shuffle(seed=42)

n = len(dataset)
train_size = int(n * 0.85)
val_size = int(n * 0.075)

final = DatasetDict({
    "train": dataset.select(range(train_size)),
    "validation": dataset.select(range(train_size, train_size + val_size)),
    "test": dataset.select(range(train_size + val_size, n)),
})

print(final)
final.save_to_disk("../training_data_v6_split")
print("\nSaved to training_data_v6_split/")