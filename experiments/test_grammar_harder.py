from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "vennify/t5-base-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def correct_grammar(text):
    input_text = "grammar: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=256, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

test_cases = [
    # Harder grammar patterns
    "If I was you, I would of done it different.",
    "Neither of the students have finished they're homework.",
    "The reason why he is late is because he miss the bus.",
    "Their going to loose the game if they don't try harder.",

    # A full paragraph (multi-sentence, single call)
    "Me and my friend was walking to the store yesterday. We seen a dog who was very hungry. It dont have no owner so we gave it some food.",

    # Edge cases
    "",  # empty string
    "The quick brown fox jumps over the lazy dog while the sun was setting beautifully over the horizon, casting long shadows across the meadow.",  # long, correct
]

for s in test_cases:
    corrected = correct_grammar(s) if s else "(empty input, skipped)"
    print(f"Original:  {s!r}")
    print(f"Corrected: {corrected!r}")
    print("---")