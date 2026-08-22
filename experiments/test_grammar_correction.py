from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "vennify/t5-base-grammar-correction"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def correct_grammar(text):
    input_text = "grammar: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

test_sentences = [
    "She dont like to eat vegetables but she like fruits.",
    "I has two cat and they is very cute.",
    "Yesterday I go to the market and buy some apple.",
    "He don't know nothing about it.",
    "This is a correct sentence with no errors.",  # control: should NOT be changed
    "Me and him was going to the store.",
]

for s in test_sentences:
    corrected = correct_grammar(s)
    print(f"Original:  {s}")
    print(f"Corrected: {corrected}")
    print("---")