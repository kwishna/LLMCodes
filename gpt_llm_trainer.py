"""
https://colab.research.google.com/drive/1mV9sAY4QBKLmS58dpFGHgwCXQKRASR31

The goal of this notebook is to experiment with a new way to make it very easy to build a task-specific model for your use-case.

First, use the best GPU available (go to Runtime -> change runtime type)

To create your model, just go to the first code cell, and describe the model you want to build in the prompt. Be descriptive and clear.

Select a temperature (high=creative, low=precise), and the number of training examples to generate to train the model. From there, just run all the cells.

You can change the model you want to fine-tune by changing `model_name` in the `Define Hyperparameters` cell.

#Data generation step

Write your prompt here. Make it as descriptive as possible!

Then, choose the temperature (between 0 and 1) to use when generating data. Lower values are great for precise tasks, like writing code, whereas larger values are better for creative tasks, like writing stories.

Finally, choose how many examples you want to generate. The more you generate, a) the longer it takes and b) the more expensive data generation will be. But generally, more examples will lead to a higher-quality model. 100 is usually the minimum to start.
"""

prompt = "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in Spanish."
temperature = .4
number_of_examples = 100

"""Run this to generate the dataset."""

# !pip install openai

import os
import openai
import random

openai.api_key = "YOUR KEY HERE"

def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 10:
            prev_examples = random.sample(prev_examples, 10)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1354,
    )

    return response.choices[0].message['content']

# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)

"""We also need to generate a system message."""

def generate_system_message(prompt):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message['content']

system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')

"""Now let's put our examples into a dataframe and turn them into a final pair of datasets."""

import pandas as pd

# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()

"""Split into train and test sets."""

# Split the data into train and test sets, with 90% in the train set
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save the dataframes to .jsonl files
train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.jsonl', orient='records', lines=True)

"""# Install necessary libraries"""

# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

"""# Define Hyperparameters"""

model_name = "NousResearch/llama-2-7b-chat-hf" # use this if you have access to the official LLaMA 2 model "meta-llama/Llama-2-7b-chat-hf", though keep in mind you'll need to pass a Hugging Face key argument
dataset_name = "/content/train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}

"""#Load Datasets and Train"""

# Load datasets
train_dataset = load_dataset('json', data_files='/content/train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files='/content/test.jsonl', split="train")

# Preprocess datasets
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)

# Cell 4: Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]" # replace the command here with something relevant to your task
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(prompt)
print(result[0]['generated_text'])

"""#Run Inference"""

from transformers import pipeline

prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]" # replace the command here with something relevant to your task
num_new_tokens = 100  # change to the number of new tokens you want to generate

# Count the number of tokens in the prompt
num_prompt_tokens = len(tokenizer(prompt)['input_ids'])

# Calculate the maximum length for the generation
max_length = num_prompt_tokens + num_new_tokens

gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length)
result = gen(prompt)
print(result[0]['generated_text'].replace(prompt, ''))

"""#Merge the model and store in Google Drive"""

# Merge and save the fine-tuned model
from google.colab import drive
drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/llama-2-7b-custom"  # change to your preferred path

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Save the merged model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

"""# Load a fine-tuned model from Drive and run inference"""

from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer

drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/llama-2-7b-custom"  # change to the path where your model is saved

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import pipeline

prompt = "What is 2 + 2?"  # change to your desired prompt
gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = gen(prompt)
print(result[0]['generated_text'])