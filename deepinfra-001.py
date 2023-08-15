import os
import requests

url = "https://api.deepinfra.com/v1/inference/tiiuae/falcon-7b"
input_text = "Generative AI is sometimes used in an unethical way by some bad social elements. It has not done a major harm so far." \
             "Now the government bodies have started looking to bring some law or observation body." \
             "It is highly possible that some company develop an AI based technology that might have potential to impact the society massively" \
             "and before government body understand the impact and can take any action, someone will use it for unethical purpose and it will be too late." \
             "I hope this will not happen. What should we do to ensure that we could monitor any such development and make sure that " \
             "it does not reach in hands of unethical peoples' hand?"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['DEEPINFRA_API_KEY']}"
}

payload = {
    "input": {
        "prompt": input_text
    }
}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    # Process the response here
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Error:", e)


