import torch
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

# Load PDF and split into chunks
file_path = "./p4.pdf"

loader = PyPDFLoader(file_path)
pages = [page.page_content for page in loader.lazy_load()]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = []
for page in pages:
    chunks.extend(text_splitter.split_text(page))

# Prepare documents
documents = [
    Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks
]
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# Initialize with an embedding model
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {"device": "cpu" if torch.mps.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Add documents to vector store
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents=documents)

# Perform similarity search
query = "when did I have pancakes"
retriever = vector_store.as_retriever()
retrieved_docs = retriever.invoke(query)

# Define LLM endpoint
model_id = "google/gemma-1.1-2b-it"
# llm = HuggingFaceEndpoint(
#     repo_id=model_id,
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token="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",
# )
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
    device_map="cpu",
    trust_remote_code=True
)


# Prepare for chat
chat = ChatHuggingFace(llm=llm, verbose=True)
initial_context = "\n".join([doc.page_content for doc in retrieved_docs])

# Chat messages
messages = [
    {
        "role": "user",
        "content": initial_context,
    },
    {"role": "assistant", "content": "process the above text"},
    {"role": "user", "content": "When did I have pancakes?"},
]
response = chat.invoke(messages)

# Output results
print("Chat Response:", response)
