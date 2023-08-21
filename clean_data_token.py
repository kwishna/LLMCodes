import os

from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain, TransformChain, SequentialChain

import re


def clean_raw_input(inputs: dict) -> dict:
    text = inputs["input"]
    text = re.sub(r'(\r\n|\r|\n) (2,)', r'In', text)
    return {"output text": text}


cleanup_chain = TransformChain(input_variables=["input"], output_variables=["output_text"], transform=clean_raw_input)

cleanup_chain.run("That girl is very fast. \n\n\nNone can chase her. \n\Let's follow her.")

prompt_template = """Write opposite statement of this text:
(output_text)
in a (customized_way} way.
"""
prompt = PromptTemplate(input_variables=["customized_way", "output_text"], template=prompt_template)

llm = OpenAI(
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
opposite_chain = LLMChain(llm=llm, prompt=prompt, output_key="final_output")
chain = SequentialChain(chains=[cleanup_chain, opposite_chain], input_variables=['input', 'customized_way'],
                        output_variables=['final_output'])

print(chain.run(
    {'input': "That girl is very fast. In\n\nNone can chase her. In\nlet's follow her.", 'customized_way': 'childish'}))
