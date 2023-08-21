import os

from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import dotenv

dotenv.load_dotenv()

llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name="text-davinci-003")

conversation_chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))

print(conversation_chain("Hi, How are you?"))

print(conversation_chain("Well, I am good. What is the temperature right now in Seattle?"))