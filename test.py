# Import required libraries from LangChain and others
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Read your Hugging Face API token securely from a file
with open("token.txt", "r") as token_file:
    HUGGINGFACEHUB_API_TOKEN = token_file.read().strip()

# Example documents: These could be details from your projects, resume, or any related FAQ content.
documents = [
    "Project A: Developed an AI chatbot using Python, LangChain, and a RAG approach.",
    "Project B: Created a dynamic portfolio website to showcase projects and technical skills.",
    "Resume: Experienced software developer with expertise in machine learning, web development, and NLP applications.",
]

# Create embeddings for your documents using a SentenceTransformer model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build a vector store from your documents using FAISS
vectorstore = FAISS.from_texts(documents, embeddings)

# Set up the Hugging Face endpoint with your API key.
# Here, the API key from token.txt is passed into the HuggingFaceEndpoint.
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Change this to your desired repo/model if needed
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# Optionally, define a prompt template if you need a custom prompt structure
template = """Question: {question}

Answer: Let's think step by step.
"""
prompt = PromptTemplate.from_template(template)

# Create a RetrievalQA chain that uses the vectorstore retriever and the Hugging Face LLM.
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 'stuff' is one way to combine context; other options are available depending on your needs
    retriever=vectorstore.as_retriever()
)

# Now, you can run queries against your RAG system.
query = "Can you tell me about your AI chatbot project?"
answer = retrieval_qa.run(query)
print("Answer:", answer)
