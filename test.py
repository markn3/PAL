from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

token = open("token.txt", 'r')
HUGGINGFACEHUB_API_TOKEN = token.read()

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# repo_id = "meta-llama/Llama-3.2-1B" # Need Approval


llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
llm_chain = prompt | llm
print(llm_chain.invoke({"question": question}))