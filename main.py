with open("token.txt", "r") as token_file:
    tokens = token_file.read().strip()
HUGGINGFACEHUB_API_TOKEN = tokens.split()[0]
LANGSMITH_API_KEY = tokens.split()[1]
print(LANGSMITH_API_KEY)

from langchain.chat_models import init_chat_model

llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Load and chunk contents
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter # try both

loader = TextLoader("./data/resume.md")
docs = loader.load()

print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

print(f"Split resume into {len(all_splits)} sub-documents.")



