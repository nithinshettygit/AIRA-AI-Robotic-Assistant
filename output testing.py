from TTS.api import TTS

tts = TTS(model_name="tts_models/en/vctk/vits")

# List available speakers
print(tts.speakers)  # e.g., ['p225', 'p226', ...]

# Specify a speaker_id when synthesizing
tts.tts_to_file(text="Hello, how can I help you?", speaker="p225", file_path="output.wav")
