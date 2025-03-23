from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Initialize your Llama 2 model with your Hugging Face API token.
llama_llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b",
    huggingfacehub_api_token="your_actual_token_here",  # Replace with your token
    model_kwargs={"temperature": 0.7}
)

# Create embeddings for your documents
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
    chain_type="stuff",  # You can also try 'map_reduce', 'refine', etc.
    retriever=vector_store.as_retriever()
)

# Ask a question about your resume/projects
question = "Can you tell me about my work experience?"
answer = qa_chain.run(question)
print("Chatbot Answer:", answer)
