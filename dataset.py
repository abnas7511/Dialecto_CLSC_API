import pandas as pd
from datasets import Dataset
import soundfile as sf
import os
import json 

def prepare_dataset(csv_path, audio_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Read all CSV files if multiple exist
    if os.path.isfile(csv_path):
        data = pd.read_csv(csv_path, encoding='utf-8')
    else:
        # If csv_path is a directory, combine all CSV files
        csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(csv_path, csv_file), encoding='utf-8')
            data_frames.append(df)
        data = pd.concat(data_frames, ignore_index=True)

    audio_data = []
    
    for _, row in data.iterrows():
        audio_path = os.path.join(audio_folder, row['filename'])
        
        if os.path.exists(audio_path):
            try:
                # Extract the audio segment
                start_sample = int(row['start_time'] * 16000) 
                end_sample = int(row['end_time'] * 16000)
                audio_segment, sample_rate = sf.read(audio_path, start=start_sample, stop=end_sample)
                
                segment_path = os.path.join(output_folder, f"{row['filename']}_{start_sample}_{end_sample}.wav")
                sf.write(segment_path, audio_segment, sample_rate)

                audio_data.append({
                    'audio': segment_path,
                    'text': row['label']
                })
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
    
    dataset = Dataset.from_pandas(pd.DataFrame(audio_data))
    
    # Save dataset as JSON
    json_output = [{
        'audio': item['audio'],
        'text': item['text']
    } for item in dataset]
    
    json_path = os.path.join(output_folder, 'dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)
    
    return dataset

# Example usage
csv_path = 'data'  # folder containing CSV files
audio_folder = 'input_folder'
output_folder = 'output_folder'

train_dataset = prepare_dataset(csv_path, audio_folder, output_folder)

print("Dataset statistics:")
print(train_dataset)

print("\nFirst few entries:")
for i, data in enumerate(train_dataset):
    if i < 5:  # Print first 5 entries
        print(data)
