from datasets import load_from_disk
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "google-t5/t5-small"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

dataset = load_from_disk("../training_data_v5_split")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def preprocess(examples):
    inputs = ["paraphrase: " + s for s in examples["sentence1"]]
    targets = examples["sentence2"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="../models/paraphraser-v5-checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    learning_rate=5e-5,
    eval_strategy="epoch",
    predict_with_generate=True,
    generation_num_beams=4,
    logging_steps=200,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("../models/paraphraser-v5")
tokenizer.save_pretrained("../models/paraphraser-v5")
print("\nSaved to models/paraphraser-v5")