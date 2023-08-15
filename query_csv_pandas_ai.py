from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def chat_with_csv(df, prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result


data = pd.read_csv("./data.csv")
result = chat_with_csv(data, "Who is the highest scorer in the NBA?")
print(result)
