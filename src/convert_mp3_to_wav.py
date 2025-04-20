import os
from pydub import AudioSegment

input_dir = "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/audio"         # change to actual path of DEAM mp3s
output_dir = "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/audio_wav"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(output_dir, filename.replace(".mp3", ".wav"))
        try:
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")
            print(f"✅ Converted: {filename}")
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")
