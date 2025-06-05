
#uv run python ../scripts/kokoro_test.py

from kokoro import KPipeline
# from IPython.display import display, Audio
import soundfile as sf
import torch
import glob 
import os


pipeline = KPipeline(lang_code='a')
# text = '''
# [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
# '''

fnames = glob.glob("../inputs/*.txt")
for fname in fnames:
    text = []
    with open(fname, 'r') as f:
        for line in f:
            text.append(line.strip())

        generator = pipeline(text, voice='af_heart')
        full_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            full_audio.append(audio)
        audio = torch.cat(full_audio, dim=-1)
        sf.write(f'../outputs/kokoro/{os.path.basename(fname)}.wav', audio, 24000)