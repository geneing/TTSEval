Repository for running comparisons between different TTS libraries. 



# Installing fish-speech

sudo apt install portaudio19-dev
cd fish-speech
uv sync
uv add huggingface_hub
uv run huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
uv add ipykernel

# Installing chatterbox
cd chatterbox
uv sync 
uv add ipykernel

# Installing MegaTTS3
cd MegaTTS3
uv add -r requirements.txt
uv add ipykernel
uv run pip3 install modelscope
uv run modelscope download --model ACoderPassBy/MegaTTS-SFT --local_dir ./checkpoints
PYTHONPATH=.:$PYTHONPATH uv run python tts/infer_cli.py --input_wav "../outputs/kokoro/1.txt.wav"  --input_text "Hello, world!" --output_dir ./tmp
