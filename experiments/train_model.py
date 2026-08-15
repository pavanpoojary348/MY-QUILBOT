import os
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "google-t5/t5-small"

DATASET_PATH = "../training_data"
OUTPUT_DIR = "../models/paraphraser-v1"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


# ============================================================
# 2. CHECK DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading dataset...")


# ============================================================
# 3. LOAD OUR DATASET
# ============================================================

dataset = load_from_disk(DATASET_PATH)

print("\nDataset:")
print(dataset)

print("\nTraining examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print("Test examples:", len(dataset["test"]))


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 5. LOAD PRETRAINED T5
# ============================================================

print("Loading T5-small...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


# ============================================================
# 6. PREPROCESS DATA
# ============================================================

def preprocess_function(examples):

    inputs = [
        "paraphrase: " + sentence
        for sentence in examples["sentence1"]
    ]

    targets = examples["sentence2"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print("Tokenization complete!")


# ============================================================
# 7. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# ============================================================
# 8. TRAINING CONFIGURATION
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # CPU-friendly first experiment
    use_cpu=True,

    # Small batch because we are using CPU
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    # First experiment
    num_train_epochs=1,

    learning_rate=5e-5,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save after each epoch
    save_strategy="epoch",
    save_total_limit=1,

    # Generate during evaluation
    predict_with_generate=True,

    # Generation settings
    generation_max_length=128,
    generation_num_beams=4,

    # Logging
    logging_steps=100,
    report_to="none",

    # Reproducibility
    seed=42,
)


# ============================================================
# 9. CREATE TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,
)


# ============================================================
# 10. START TRAINING
# ============================================================

print("\n========================================")
print("STARTING FINE-TUNING")
print("========================================")

trainer.train()


# ============================================================
# 11. SAVE OUR MODEL
# ============================================================

print("\nSaving our fine-tuned model...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n========================================")
print("TRAINING COMPLETE!")
print("========================================")

print("Our model is saved at:")
print(OUTPUT_DIR)