from transformers import pipeline

cola = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")

test_sentences = [
    # Known-good outputs from v6's eval
    "How can I lose weight faster?",
    "Global weather patterns are affected by climate change.",
    "I am tired.",

    # Known-broken outputs that slipped through v6's filter
    "Her passion of writing books at weekends is to read it often.",
    "The effect of photosynthesis on the production of the sunlight becomes chemical energy.",

    # A plain correct sentence for reference
    "The weather is very nice today.",
]

for s in test_sentences:
    result = cola(s)[0]
    print(f"[{result['label']:12s} {result['score']:.3f}] {s}")