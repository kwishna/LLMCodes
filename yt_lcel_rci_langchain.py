# !pip -q install langchain openai tiktoken
# https://www.youtube.com/watch?v=QaKM5s0TnsY
import os

os.environ["OPENAI_API_KEY"] = ""

# !pip show langchain
"""# RCI Chain with ChatModel

## Multi Chain
"""

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initial Prompt
prompt = ChatPromptTemplate.from_template(
  "tell me an intersting fact about {subject}")

# Critique prompt
reverse_prompt = ChatPromptTemplate.from_template(
  "based on this interesting fact which is chunked down from a meta subject:\n\n {interesting_fact}\n\n Recover what the meta subject is\n Subject:"
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"subject": "Elvis"}))

# full chain doesn't work because 'reverse prompt' requires a JSON input instead of string.
full_chain = prompt | model | StrOutputParser(
) | reverse_prompt | model | StrOutputParser()

# -----

import langchain

langchain.debug = True

print(full_chain.invoke({"subject": "Elvis"}))

# Fixing the not working chain by spliting into 2 parts
chain1 = prompt | model | StrOutputParser()

chain2 = {
  "interesting_fact": chain1
} | reverse_prompt | model | StrOutputParser()

print(chain2.invoke({"subject": "elvis"})) # It will first invoke chain1, then use the output to invoke chain2.
"""# As LCEL"""

langchain.debug = False

from langchain import PromptTemplate
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)

template = "You are a helpful assistant that imparts wisdom and guides people with accurate answers."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
  [system_message_prompt, human_message_prompt])

chain1 = chat_prompt | model | StrOutputParser()

initial_question = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

initial_answer = chain1.invoke({"question": initial_question})
print(initial_answer)

fake_initial_ai_answer = """Roger initially has 5 tennis balls. Each can of tennis balls contains 3 tennis balls, and he bought 2 cans, so he has 2 x 3 = 6 additional tennis balls.
Therefore, the total number of tennis balls Roger has now is 5 + 4 = 9."""
"""## Part 2 - Critique  """

# Critique chain
template = "You are a helpful assistant that looks at answers and finds what is wrong with them based on the original question given."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "### Question:\n\n{question}\n\n ###Answer Given:{initial_answer}\n\n Review your previous answer and find problems with your answer"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

rc_prompt = ChatPromptTemplate.from_messages(
  [system_message_prompt, human_message_prompt])

chain2 = rc_prompt | model | StrOutputParser()

constructive_criticism = chain2.invoke({
  "question": initial_question,
  "initial_answer": fake_initial_ai_answer
})
print(constructive_criticism)
"""## Part 3 - The Improvement"""

template = "You are a helpful assistant that reviews answers and critiques based on the original question given and write a new improved final answer."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "### Question:\n\n{question}\n\n ###Answer Given:{initial_answer}\n\n \
###Constructive Criticism:{constructive_criticism}\n\n Based on the problems you found, improve your answer.\n\n### Final Answer:"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

improvement_prompt = ChatPromptTemplate.from_messages(
  [system_message_prompt, human_message_prompt])

chain3 = improvement_prompt | model | StrOutputParser()

final_result = chain3.invoke({
  "question": initial_question,
  "initial_answer": fake_initial_ai_answer,
  "constructive_criticism": constructive_criticism
})

print(final_result)
"""## Combined Chain"""

from operator import itemgetter

chain1 = chat_prompt | model | StrOutputParser()

critque_chain = {
  "question": itemgetter("question"),
  "initial_answer": chain1
} | rc_prompt | model | StrOutputParser()

chain3 = {
  "question": itemgetter("question"),
  "initial_answer": chain1,
  "constructive_criticism": critque_chain
} | improvement_prompt | model | StrOutputParser()

print(chain3.invoke({"question": "Write an sms message to say I am tired"}))

langchain.debug = True

print(chain3.invoke({"question": "Write an sms message to say I am tired"}))
