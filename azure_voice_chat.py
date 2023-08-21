import os

import azure.cognitiveservices.speech as speechsdk

AZURE_SPEECH_KEY = ""
AZURE_SPEECH_REGION = ""

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
speech_config.speech_recognition_language = "en-US"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer (speech_config=speech_config, audio_config=audio_config)
print("You can speak now. I'm listening...")
speech_recognition_result = speech_recognizer.recognize_once_async () -get ()
output = speech_recognition_result.text


import openai
openai.api_type = "azure"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = "hhtps://channel-ins.openai.azure.com"
openai.api_version = "2021-10-21-preview"

audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk. SpeechSynthesizer (speech_config=speech_config, audio_config=audio_config)
speech_synthesis_result = speech_synthesizer.speak_text_async(output).get()