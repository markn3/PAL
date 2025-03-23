from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # FAISS import remains the same for now
from langchain.chains import RetrievalQA
import os

# Ensure your Hugging Face API token is available (you can set it as an environment variable)
token = open("token.txt", 'r')
token = token.read()

# Initialize your Llama 2 model using the new endpoint
llama_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-2-7b",
    huggingfacehub_api_token=token,
    model_kwargs={"temperature": 0.7}
)

# Create embeddings for your documents using the new embeddings class
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example documents representing your resume and project details
documents = [
    "I have 5 years of experience in software engineering, specializing in web development...",
    "During my tenure at XYZ Corp, I led a team to develop a scalable microservices architecture..."
]

# Index your documents using FAISS
vector_store = FAISS.from_texts(documents, embeddings)

# Create a RetrievalQA chain for your chatbot
qa_chain = RetrievalQA.from_chain_type(
    llm=llama_llm,
    chain_type="stuff",  # You can experiment with 'map_reduce', 'refine', etc.
    retriever=vector_store.as_retriever()
)

# Ask a question about your resume/projects
question = "Can you tell me about my work experience?"
answer = qa_chain.run(question)
print("Chatbot Answer:", answer)
