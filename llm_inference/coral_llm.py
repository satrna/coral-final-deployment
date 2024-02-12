from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate

llm = VertexAI(model_name="code-bison", max_output_tokens=1000, temperature=0.3)
question = "Write a python function that checks if a string is a valid email address"
print(llm.invoke(question))