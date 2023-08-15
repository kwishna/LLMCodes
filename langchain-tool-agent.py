from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.8)
tools = load_tools(['serpapi', 'llm-math'])

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

res = agent.run("what is wisper api?")
print(res)
