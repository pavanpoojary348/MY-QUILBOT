from datasets import load_dataset
import ast

def word_overlap(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

dataset = load_dataset("humarin/chatgpt-paraphrases")
train = dataset["train"]
sentence_examples = train.filter(lambda ex: ex["category"] == "sentence")

scores = []
for i in range(2000):
    example = sentence_examples[i]
    original = example["text"]
    paraphrase_list = ast.literal_eval(example["paraphrases"])
    first_paraphrase = paraphrase_list[0]
    scores.append(word_overlap(original, first_paraphrase))

scores.sort()
print(f"Sampled {len(scores)} declarative pairs")
print(f"Min: {scores[0]:.2f}")
print(f"25th percentile: {scores[len(scores)//4]:.2f}")
print(f"Median: {scores[len(scores)//2]:.2f}")
print(f"75th percentile: {scores[3*len(scores)//4]:.2f}")
print(f"Max: {scores[-1]:.2f}")