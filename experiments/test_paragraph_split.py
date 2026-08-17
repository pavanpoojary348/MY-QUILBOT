import re

ABBREVIATIONS = [
    "Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "Prof",
    "vs", "etc", "e.g", "i.e", "U.S", "U.K", "U.N",
    "Inc", "Ltd", "Co", "St",
]

def split_sentences(text):
    protected = text
    # Temporarily replace "Dr." with "Dr<PERIOD>" etc. so the splitter
    # doesn't mistake the abbreviation's period for a sentence end.
    for abbr in ABBREVIATIONS:
        protected = re.sub(
            rf'\b{re.escape(abbr)}\.',
            f'{abbr}<PERIOD>',
            protected
        )

    # Split on '.', '!', or '?' followed by whitespace and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected.strip())

    # Restore the protected periods
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s]

test_paragraph = """I am very happy because I got a new job. The weather is very nice today. Dr. Smith recommended I take a break. What is the best way to learn Python? Artificial intelligence is changing the world."""

sentences = split_sentences(test_paragraph)
for i, s in enumerate(sentences, 1):
    print(f"[{i}] {s}")