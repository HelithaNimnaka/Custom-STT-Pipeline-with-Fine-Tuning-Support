import torch
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# ---------------------- Load Dataset ---------------------- #
token = <YOUR_HUGGINGFACE_TOKEN>

# Stream and take 100 samples
streamed_dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="train",
    token=token,
    trust_remote_code=True,
    streaming=True
)

samples = list(streamed_dataset.take(100))
dataset = Dataset.from_list(samples)

# -------------------- Load Processor -------------------- #
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# ------------------ Preprocessing Function ------------------ #
def preprocess(batch):
    audio = batch["audio"]
    
    # Extract audio features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    
    # Tokenize text labels (with truncation)
    batch["labels"] = processor.tokenizer(
        batch["sentence"], max_length=448, truncation=True
    ).input_ids
    
    return batch

# Apply audio casting & preprocessing
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ----------------- Simple Data Collator ----------------- #
def simple_data_collator(features):
    return {
        "input_features": torch.stack([
            torch.tensor(f["input_features"]) for f in features
        ]),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["labels"]) for f in features],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id
        )
    }

# ------------------ Load Model ------------------ #
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# ----------------- Training Arguments ----------------- #
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    fp16=True,
    save_steps=10
)

# ------------------- Trainer ------------------- #
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=simple_data_collator
)

# -------------------- Train -------------------- #
trainer.train()

# ------------------ Save Model ------------------ #
model.save_pretrained("./whisper-finetuned")
