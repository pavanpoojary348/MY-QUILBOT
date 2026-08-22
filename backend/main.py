from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
import re


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="AI Writing Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODELS (once, at server startup — not per request)
# ============================================================

MODEL_PATH = "../models/paraphraser-v6"

print("Loading paraphrasing model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# ============================================================
# LOAD GRAMMAR CORRECTION MODEL
# ============================================================

GRAMMAR_MODEL_PATH = "vennify/t5-base-grammar-correction"

print("Loading grammar correction model...")
grammar_tokenizer = T5Tokenizer.from_pretrained(GRAMMAR_MODEL_PATH)
grammar_model = T5ForConditionalGeneration.from_pretrained(
    GRAMMAR_MODEL_PATH
)

print("Grammar model loaded successfully.")
print("Loading semantic similarity model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading contextual synonym model...")
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

print("Models loaded successfully.")


# ============================================================
# SENTENCE SPLITTING
# ============================================================

ABBREVIATIONS = [
    "Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "Prof",
    "vs", "etc", "e.g", "i.e", "U.S", "U.K", "U.N",
    "Inc", "Ltd", "Co", "St",
]

def split_sentences(text):
    protected = text
    for abbr in ABBREVIATIONS:
        protected = re.sub(rf'\b{re.escape(abbr)}\.', f'{abbr}<PERIOD>', protected)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected.strip())
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def word_overlap(a, b):
    w1 = set(a.lower().split())
    w2 = set(b.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def semantic_similarity(a, b):
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


def introduces_new_entities(original, candidate):
    def extract_entities(text):
        words = text.split()
        proper_nouns = set()
        for i, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if i > 0 and clean and clean[0].isupper():
                proper_nouns.add(clean.lower())
        numbers = set(re.findall(r'\d+', text))
        return proper_nouns, numbers

    orig_nouns, orig_nums = extract_entities(original)
    cand_nouns, cand_nums = extract_entities(candidate)
    new_nouns = cand_nouns - orig_nouns
    new_nums = cand_nums - orig_nums
    return len(new_nouns) > 0 or len(new_nums) > 0


def generate_candidates(sentence, num_candidates=8):
    input_text = "paraphrase: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    candidates = []
    for _ in range(num_candidates):
        output_ids = model.generate(
            input_ids, max_length=128,
            do_sample=True, top_p=0.9, temperature=1.2,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        overlap = word_overlap(sentence, text)
        similarity = semantic_similarity(sentence, text)
        candidates.append((text, overlap, similarity))
    return candidates


def pick_best(candidates, original, similarity_min=0.85, overlap_min=0.10, overlap_max=0.55):
    valid = [
        c for c in candidates
        if c[2] >= similarity_min
        and overlap_min <= c[1] <= overlap_max
        and not introduces_new_entities(original, c[0])
    ]
    if valid:
        return min(valid, key=lambda c: c[1])
    return None


def paraphrase(sentence, num_candidates=8):
    """
    Returns (paraphrased_text, confident: bool).
    If confident is False, paraphrased_text is the original sentence —
    we do not return a risky/unverified rewording.
    """
    candidates = generate_candidates(sentence, num_candidates)
    result = pick_best(candidates, sentence)
    if result:
        return result[0], True
    return sentence, False
# ============================================================
# GRAMMAR CORRECTION
# ============================================================

def correct_grammar(text):
    """
    Corrects grammatical errors while trying to preserve
    the original meaning and wording.
    """

    if not text or not text.strip():
        return "", False

    input_text = "grammar: " + text

    input_ids = grammar_tokenizer(
        input_text,
        return_tensors="pt"
    ).input_ids

    output_ids = grammar_model.generate(
        input_ids,
        max_length=256,
        num_beams=5
    )

    corrected = grammar_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    overlap = word_overlap(text, corrected)

    confident = overlap >= 0.5

    return corrected, confident

# ============================================================
# SYNONYM LOOKUP (WordNet + BERT contextual re-ranking)
# ============================================================

def get_wordnet_candidates(word):
    candidates = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower() and ' ' not in name:
                candidates.add(name.lower())
    return candidates


def get_contextual_synonyms(sentence, target_word, max_results=5):
    """
    Uses WordNet to generate candidate synonyms, then re-ranks them
    using BERT's masked-language-model predictions for the target
    word's position in THIS specific sentence — filters out synonyms
    that are technically valid in isolation but don't fit the context
    (e.g. "felicitous" for "happy" in "I am very happy").
    """
    masked_sentence = re.sub(
        rf'\b{re.escape(target_word)}\b',
        '[MASK]',
        sentence,
        count=1,
        flags=re.IGNORECASE
    )

    candidates = get_wordnet_candidates(target_word)
    if not candidates:
        return []

    predictions = fill_mask(masked_sentence, top_k=100)
    bert_ranked_words = {p['token_str'].strip().lower(): p['score'] for p in predictions}

    scored = [(c, bert_ranked_words.get(c, 0.0)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return [w for w, s in scored[:max_results] if s > 0]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "AI Writing Assistant backend is running"}


# --- Single sentence ---

class ParaphraseRequest(BaseModel):
    text: str


class ParaphraseResponse(BaseModel):
    original: str
    paraphrased: str
    confident: bool


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase_endpoint(request: ParaphraseRequest):
    result, confident = paraphrase(request.text)
    return ParaphraseResponse(
        original=request.text,
        paraphrased=result,
        confident=confident,
    )


# --- Full paragraph ---

class ParagraphRequest(BaseModel):
    text: str


class SentenceResult(BaseModel):
    original: str
    paraphrased: str
    confident: bool


class ParagraphResponse(BaseModel):
    original: str
    paraphrased: str
    sentences: list[SentenceResult]


@app.post("/paraphrase-paragraph", response_model=ParagraphResponse)
def paraphrase_paragraph_endpoint(request: ParagraphRequest):
    sentences = split_sentences(request.text)
    results = []
    for s in sentences:
        paraphrased, confident = paraphrase(s)
        results.append(SentenceResult(original=s, paraphrased=paraphrased, confident=confident))
    full_output = " ".join(r.paraphrased for r in results)
    return ParagraphResponse(
        original=request.text,
        paraphrased=full_output,
        sentences=results,
    )


# --- Synonym lookup (context-aware) ---

class SynonymRequest(BaseModel):
    text: str
    word: str


class SynonymResponse(BaseModel):
    word: str
    synonyms: list[str]


@app.post("/synonyms", response_model=SynonymResponse)
def synonyms_endpoint(request: SynonymRequest):
    return SynonymResponse(
        word=request.word,
        synonyms=get_contextual_synonyms(request.text, request.word),
    )
# ============================================================
# GRAMMAR CORRECTION ENDPOINT
# ============================================================

class GrammarRequest(BaseModel):
    text: str


class GrammarResponse(BaseModel):
    original: str
    corrected: str
    confident: bool


@app.post("/grammar", response_model=GrammarResponse)
def grammar_endpoint(request: GrammarRequest):

    corrected, confident = correct_grammar(
        request.text
    )

    return GrammarResponse(
        original=request.text,
        corrected=corrected if request.text.strip() else request.text,
        confident=confident,
    )