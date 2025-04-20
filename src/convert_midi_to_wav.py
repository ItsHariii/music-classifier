import os
import pretty_midi
import numpy as np
import soundfile as sf
import librosa

soundfont_path = "/Users/harimanivannan/Documents/GitHub/music-classifier/assets/GeneralUser GS v1.472.sf2"
input_dir = "/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/audio"
output_dir = "/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/audio_wav"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".mid"):
        midi_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(output_dir, filename.replace(".mid", ".wav"))
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            audio_data = midi.fluidsynth(sf2_path=soundfont_path)
            sf.write(wav_path, audio_data, samplerate=22050)  # <-- fixed line
            print(f"✅ Converted {filename}")
        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")
