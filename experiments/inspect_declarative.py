from datasets import load_dataset
import ast

dataset = load_dataset("humarin/chatgpt-paraphrases")
train = dataset["train"]

sentence_examples = train.filter(lambda ex: ex["category"] == "sentence")
print(f"Total 'sentence' category examples: {len(sentence_examples)}")
print()

for i in range(10):
    example = sentence_examples[i]
    original = example["text"]
    paraphrase_list = ast.literal_eval(example["paraphrases"])  # parse the string as a real list
    first_paraphrase = paraphrase_list[0]

    print(f"Source:     {example['source']}")
    print(f"Original:   {original}")
    print(f"Paraphrase: {first_paraphrase}")
    print("---")