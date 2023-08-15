# Import Chat completion template and set-up variables
# !pip install openai &> /dev/null
import openai
import urllib.parse

openai.api_key = "EMPTY"  # Key is ignored and does not matter
openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"

# Alternate mirrors
# openai.api_base = "http://34.132.127.197:8000/v1"

# Report issues
# def raise_issue(e, model, prompt):
#     issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
#     issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
#     issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
#     print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")


# Query Gorilla server
def get_gorilla_response(
    prompt="I would like to translate from English to French.",
    model="gorilla-7b-hf-v1"):
  try:
    completion = openai.ChatCompletion.create(model=model,
                                              messages=[{
                                                "role": "user",
                                                "content": prompt
                                              }])
    return completion.choices[0].message.content
  except Exception as e:
    print(e)
    # raise_issue(e, model, prompt)


"""## Example 1: Translation ✍ with 🤗"""
# Gorilla `gorilla-mpt-7b-hf-v1` with code snippets
# Translation

prompt = "I would like to translate 'I feel very good today.' from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))
"""
<<<domain>>>: Natural Language Processing Translation
<<<api_call>>>: translator = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. Import the pipeline function from the transformers library provided by Hugging Face.
2. Create a translation pipeline 'translation_en_to_zh' with the model 'Helsinki-NLP/opus-mt-en-zh', which is a pretrained model for English to Chinese translation.
3. Use the created translator to translate the English text 'I feel very good today.' to Chinese.
<<<code>>>:
from transformers import pipeline

def load_model():
    translator = pipeline('translation_en_to_zh', model='Helsinki-NLP/opus-mt-en-zh')
    return translator

def process_data(text, translator):
    response = translator(text)
    translation_text = response[0]['translation_text']
    return translation_text

english_text = 'I feel very good today.'

# Load the model
translator = load_model()

# Process the data
translated_text = process_data(english_text, translator)
print(translated_text)
"""

## Example 2: Object detection 🔷 with 🤗

# Gorilla `gorilla-7b-hf-v1` with code snippets
# Object Detection
prompt = "I want to build a robot that can detecting objects in an image ‘cat.jpeg’. Input: [‘cat.jpeg’]"
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))
"""
<<<domain>>>: Computer Vision Object Detection
<<<api_call>>>: model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. Import the necessary components from the Hugging Face Transformers library, torch, and PIL (Python Imaging Library).
2. Open the image using PIL's Image.open() function with the provided image path.
3. Initialize the pretrained DETR (DEtection TRansformer) model and the image processor.
4. Generate inputs for the model using the image processor.
5. Pass the inputs to the model, which returns object detection results.
<<<code>>>:

from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from PIL import Image
import torch

def load_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = AutoModelForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    return feature_extractor, model

def process_data(image_path, feature_extractor, model):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    results = feature_extractor.post_process(outputs, threshold=0.6)[0]
    response = [model.config.id2label[label.item()] for label in results['labels']]
    return response

image_path = 'cat.jpeg'

# Load the model and feature extractor
feature_extractor, model = load_model()

# Process the data
response = process_data(image_path, feature_extractor, model)
print(response)
"""
"""## Let's try to invoke APIs from Torch Hub instead for the same prompts!"""
# Translation ✍ with Torch Hub
prompt = "I would like to translate from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-th-v0"))
"""
{
  'domain': 'Machine Translation',
  'api_call': "model = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')",
  'api_provider': 'PyTorch',
  'explanation': 'Load the Transformer model from PyTorch Hub, which is specifically trained on the WMT 2014 English-French translation task.',
  'code': 'import torch\nmodel = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')'
}
"""
"""## ⛳️ With Gorilla being fine-tuned on MPT, and Falcon, you can use Gorilla commercially with no obligations! 🟢"""
# Gorilla with `gorilla-mpt-7b-hf-v0`
prompt = "I would like to translate from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-mpt-7b-hf-v0"))
"""
Please provide the English text you would like to translate: \"Translate this text to Chinese:\"\n<<<domain>>>: Natural Language Processing Text2Text Generation\n
<<<api_call>>>: M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>:
1. Import the necessary libraries - M2M100ForConditionalGeneration and M2M100Tokenizer from the transformers library.
2. Load the pretrained model 'facebook/m2m100_1.2B' and its corresponding tokenizer.
3. Set the source language to English (en) and use the tokenizer to tokenize the input text.
4. Use the model to generate the translated text in Chinese by providing the tokenized input to the 'generate' function.
5. Decode the generated tokens back into a readable text string using the tokenizer.

<<<code>>>:
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
src_text = "Translate this text to Chinese:"
src_lang = "en"
model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
tokenized_input = tokenizer(src_text, return_tensors="pt", src_lang=src_lang)
translated_tokens = model.generate(**tokenized_input)
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)" 
"""

##Below are example prompt-responses for `gorilla-7b-hf-v0` Legacy Model for 🤗

prompt = "I would like to translate from English to Chinese."

print(get_gorilla_response(prompt, model="gorilla-7b-hf-v0"))
"""
<<<domain>>>: Natural Language Processing Text2Text Generation
<<<api_call>>>: M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. Import the necessary libraries, which are M2M100ForConditionalGeneration and M2M100Tokenizer from the transformers package.
2. Load the pre-trained model 'facebook/m2m100_1.2B' using the M2M100ForConditionalGeneration.from_pretrained() method. This model is designed for machine-to-machine translation tasks.
3. Load the tokenizer using the M2M100Tokenizer.from_pretrained() method. This tokenizer is used to prepare the input text for the model and convert the translated output back into human-readable text.
4. Define the source text for translation and tokenize it using the tokenizer.
5. Use the model to generate the translated text using the source tokens as input.
6. Decode the translated text using the tokenizer and print the result.
"""

prompt = "I want to build a robot that can detect objects in an image."
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v0"))
"""
<<<domain>>>: Computer Vision Object Detection
<<<api_call>>>: YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
<<<api_provider>>>: Hugging Face Transformers
<<<explanation>>>: 1. We first import the necessary classes from the transformers, PIL, and requests packages. This includes YolosForObjectDetection for the object detection model and Image for processing image data.
2. We then use the from_pretrained method of the YolosForObjectDetection class to load the pre-trained model 'hustvl/yolos-tiny'. This model has been trained for object detection tasks, which is exactly what we need for detecting objects in an image.
3. We load the image data from a file or a URL, and then use the model to analyze the image and identify the objects within it.
"""
