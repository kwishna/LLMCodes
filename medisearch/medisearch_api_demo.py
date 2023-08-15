"""
    https://colab.research.google.com/drive/1-mc8j6UnIVglOsmE8ue68vGkf_BBxTK0
"""

# !pip install medisearch_client

import json
import uuid

from medisearch_client import MediSearchClient

#@title Set constants

api_key = "gx4XXBhE7Zrga682gmEm"
conversation_id = str(uuid.uuid4())
client = MediSearchClient(api_key=api_key)

#@title Make a search query
query = "Does depression increase the chances of heart attack?"
responses = client.send_user_message(conversation=[query],
                                     conversation_id=conversation_id,
                                     language="English",
                                     should_stream_response=True)

for response in responses:
  if response["event"] == "llm_response":
    text_response = response["text"]
    print(text_response)

#@title Ask a followup question
follow_up_query = "By what percentage does depression increase risk of heart disease?"
responses = client.send_user_message(conversation=[query,
                                                   text_response,
                                                   follow_up_query],
                                     conversation_id=conversation_id,
                                     language="English",
                                     should_stream_response=False)

responses

