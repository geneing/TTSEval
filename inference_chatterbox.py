import glob
import os
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


AUDIO_PROMPT_PATH = "../outputs/kokoro/1.txt.wav"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


model = ChatterboxTTS.from_pretrained(device=device)
print(f"Using device: {device}, {model.sr=}")

# text = "She sells seashells on the seashore. The shells she sells are seashells, I’m sure, so if she sells seashells on the seashore, then I’m sure she sells seashore shells."
for fname in glob.glob("../inputs/*.txt"):
    with open(fname, "r") as f:
        texts = f.read().strip()
        print(texts)
        #split text into sentences using nltk
        text = sent_tokenize(texts)
        all_wavs = []
        for text in sent_tokenize(texts):
            print(f"Processing text: {text}")
            wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
            all_wavs.append(wav)
        print(f"{torch.hstack(all_wavs).shape=}")
        ta.save(f"../outputs/chatterbox/{os.path.basename(fname)}.wav", torch.hstack(all_wavs), model.sr)