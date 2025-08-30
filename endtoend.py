import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
from playsound import playsound
from agent import ask_agent
import threading

MODEL_PATH = r"models/vosk-model-en-us-0.22"  # Update as necessary
WAKE_WORD = "aira"  # Interruption keyword, case-insensitive
SAMPLE_RATE = 16000

# Initialize Vosk model and recognizer
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)
q = queue.Queue()

# Initialize Coqui TTS
tts = TTS(model_name="tts_models/en/vctk/vits")
SPEAKER_ID = "p225"  # adjust speaker for multi-speaker models

# Flags to control listening flow
stop_listening = False
process_query_event = threading.Event()

def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))
    
def play_response(text):
    tts.tts_to_file(text=text, speaker=SPEAKER_ID, file_path="response.wav")
    playsound("response.wav")

def listen_and_recognize():
    global stop_listening
    partial_text = ""
    print("Say something! (Say 'AIRA' to interrupt)")
    
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                          channels=1, callback=audio_callback):
        while not stop_listening:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text:
                    print(f"Recognized: {text}")
                    if WAKE_WORD in text.lower().split():
                        print(f"Interruption detected by keyword '{WAKE_WORD}'")
                        # Signal to process query immediately
                        stop_listening = True
                        return True, text
                    partial_text += " " + text
                    return False, partial_text.strip()
            else:
                # Optionally handle partial results if needed
                pass
    return False, ""

def voice_agent_loop():
    global stop_listening
    stop_listening = False
    
    while True:
        interrupted, user_input = listen_and_recognize()
        
        if interrupted:
            # Extract user query after wake word "aira"
            user_input_words = user_input.lower().split()
            try:
                aira_index = user_input_words.index(WAKE_WORD)
                query = " ".join(user_input_words[aira_index+1:]).strip()
                if not query:
                    query = input("You called AIRA but didn't ask a question, please type it: ")
            except ValueError:
                query = user_input
        else:
            query = user_input
        
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        
        # Query AI agent
        print(f"User Query: {query}")
        ai_response = ask_agent(query)
        print(f"AIRA: {ai_response}")
        
        # Play response as speech
        play_response(ai_response)
        
        # Reset listening flag after response
        stop_listening = False

if __name__ == "__main__":
    voice_agent_loop()
