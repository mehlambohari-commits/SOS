import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

print("Testing voice...")

engine.say("Hello. This is AISA voice assistant.")
engine.runAndWait()

print("Voice test complete.")