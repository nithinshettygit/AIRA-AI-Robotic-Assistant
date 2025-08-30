import sounddevice as sd

# Show all devices
print(sd.query_devices())

# Show default input device
default_in = sd.query_devices(sd.default.device[0])
print("Default Input Device:", default_in)
print("Mic default sample rate:", default_in["default_samplerate"])
