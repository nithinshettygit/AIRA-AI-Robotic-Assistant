import sounddevice as sd
from vosk import Model, KaldiRecognizer
import queue
import json

model = Model("models/vosk-model-en-us-0.22")  # change path as necessary
q = queue.Queue()

def callback(indata, frames, time, status):
    # Convert buffer to bytes object before putting into queue
    q.put(bytes(indata))

samplerate = 16000  # Default for Vosk
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                      channels=1, callback=callback):
    rec = KaldiRecognizer(model, samplerate)
    print("Speak into your microphone. Press Ctrl+C to exit.")
    while True:
        data = q.get()  # Get bytes from queue
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print("Recognized:", result.get('text', ''))

