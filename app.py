import os
import json
from datasets import Dataset, DatasetDict
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments, DataCollatorCTCWithPadding

# Step 1: Load the Dataset
def load_dataset(json_file, audio_dir):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        item['audio'] = os.path.join(audio_dir, item['audio'])
    return Dataset.from_list(data)

json_file = 'output_folder/dataset.json'
audio_dir = 'output_folder'

dataset = load_dataset(json_file, audio_dir)
dataset = DatasetDict({'train': dataset})

# Step 2: Preprocess the Audio Files
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

# Step 3: Tokenize the Transcriptions
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)

def tokenize_transcription(transcription):
    return processor(transcription)

# Step 4: Create a Data Collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Step 5: Define the Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Step 6: Initialize the Model
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# Step 8: Train the Model
trainer.train()
