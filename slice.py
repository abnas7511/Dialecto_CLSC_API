import os
import csv
import librosa
import numpy as np
from scipy.io import wavfile
import re

input_folder = os.path.join(os.getcwd(), 'input_folder')
file_path = os.path.join(input_folder, "kozhikode_example2.wav")
output_folder = os.path.join(os.getcwd(), 'output_folder')
csv_file = os.path.join(os.getcwd(), 'data', 'Kozhikode_dialect_sp2_egs1.csv')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def process_audio_segment(file_path, output_path, start_time=None, end_time=None, target_sample_rate=16000):
    y, sr = librosa.load(file_path, sr=None)

    # Apply timestamp segmentation if provided
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Check if start_sample and end_sample are within bounds
        if start_sample < 0 or end_sample > len(y) or start_sample >= end_sample:
            print(f"Skipping segment: start_time={start_time}, end_time={end_time} out of bounds for {file_path}")
            return

        y = y[start_sample:end_sample]


    if sr != target_sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)

    # Step 2: Normalize audio levels to -3 dB
    if len(y) == 0:  # Check if the audio segment is empty
        print(f"Warning: No audio data after processing for {file_path} with segment {start_time}-{end_time}")
        return  

    y = librosa.util.normalize(y) * 0.707  # -3 dB is approximately 0.707

    # Step 3: Noise reduction (simple high-pass filter)
    y = librosa.effects.preemphasis(y, coef=0.97)

    y, _ = librosa.effects.trim(y, top_db=20)

    # Save processed audio
    wavfile.write(output_path, target_sample_rate, (y * 32767).astype(np.int16))

# Read CSV and process each entry
with open(csv_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row['filename']
        label = sanitize_filename(row['label'])  # Sanitize label
        speaker = sanitize_filename(row.get('speaker', 'unknown'))  # Sanitize speaker
        start_time = float(row.get('start_time', 0))
        end_time = float(row.get('end_time')) if row.get('end_time') else None

        file_path = os.path.join(input_folder, filename)
        output_filename = f"{speaker}{label}{filename}"
        output_path = os.path.join(output_folder, output_filename)

        print(f"Processing {filename} with label {label}...")

        process_audio_segment(file_path, output_path, start_time, end_time)

print("Audio processing complete.")