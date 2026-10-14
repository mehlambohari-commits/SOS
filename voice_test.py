import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("API key loaded:", GEMINI_API_KEY is not None)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ============================================================
# TEXT TO SPEECH
# ============================================================

import threading

_engine = None
_engine_lock = threading.Lock()

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
        _engine.setProperty("volume", 1.0)
    return _engine
import pythoncom

def speak(text):
    print("\nAISA:", text)
    try:
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as error:
        print("Voice error:", error)
    finally:
        pythoncom.CoUninitialize()
# ============================================================
# SPEECH TO TEXT
# ============================================================

recognizer = sr.Recognizer()


def listen():

    with sr.Microphone() as source:

        print("\nListening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=0.5
        )

        try:

            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=15
            )

            print("Processing speech...")

            text = recognizer.recognize_google(
                audio
            )

            print("You:", text)

            return text

        except sr.WaitTimeoutError:

            print("No speech detected.")

            return ""

        except sr.UnknownValueError:

            print("Could not understand your voice.")

            return ""

        except sr.RequestError as error:

            print("Speech recognition error:", error)

            return ""


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation = []


# ============================================================
# GEMINI AI FUNCTION
# ============================================================

def ask_ai(user_text):

    conversation.append(
        "User: " + user_text
    )

    prompt = """

You are AISA, an intelligent emergency voice assistant.

You are currently being tested locally.

Your responsibilities:

1. Listen carefully to the user.
2. Understand emergency situations.
3. Ask short and useful questions.
4. Give clear and concise answers.
5. Stay calm and professional.
6. Never invent information.
7. Never claim that emergency services have been contacted unless the program actually did so.
8. If someone is injured, give general safety guidance and recommend contacting appropriate emergency services.
9. Keep your answers suitable for being spoken aloud.

Conversation so far:

""" + "\n".join(conversation)

    response = client.models.generate_content(

        model="gemini-3.5-flash",

        contents=prompt

    )

    answer = response.text

    conversation.append(
        "AISA: " + answer
    )

    return answer


# ============================================================
# START ASSISTANT
# ============================================================

def start_assistant():

    print()
    print("=" * 60)
    print("        AISA EMERGENCY VOICE ASSISTANT")
    print("=" * 60)
    print("Say 'exit' or 'stop assistant' to stop.")
    print("=" * 60)

    speak(
        "AISA emergency voice assistant is ready. "
        "Please describe the situation."
    )

    while True:

        user_text = listen()

        if user_text == "":
            continue

        command = user_text.lower().strip()

        if command in [

            "exit",
            "stop",
            "stop assistant",
            "end conversation",
            "quit"

        ]:

            speak(
                "AISA assistant shutting down."
            )

            break

        try:

            response = ask_ai(user_text)

            speak(response)

        except Exception as error:

            print("\nAI ERROR:")
            print(error)

            speak(
                "I am unable to connect to the AI service right now."
            )

# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    start_assistant()