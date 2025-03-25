# Required Imports
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load Hugging Face API Token
with open("token.txt", "r") as token_file:
    HUGGINGFACEHUB_API_TOKEN = token_file.read().strip()

# Example documents (Replace with actual project/resume content)
documents = [
    "Project A: Developed an AI chatbot using Python, LangChain, and a RAG approach.",
    "Project B: Created a dynamic portfolio website to showcase projects and technical skills.",
    "Resume: Experienced software developer with expertise in machine learning, web development, and NLP applications.",
]

# Embedding Model (Updated Import)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build Vector Store with FAISS
vectorstore = FAISS.from_texts(documents, embeddings)

# Define Hugging Face Model Endpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# Define the RetrievalQA Chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query the Bot
query = "Can you tell me about your AI chatbot project?"
answer = retrieval_qa.invoke({"query": query})
print("Answer:", answer)


# RAG: https://python.langchain.com/docs/tutorials/rag/