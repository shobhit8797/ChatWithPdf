import logging
import time
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_experimental.text_splitter import SemanticChunker
import faiss
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants and Configurations
CONFIG = {
    "file_path": "./p1.pdf",
    "model_id": "google/gemma-2-2b-it",
    "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
    "questions": [
        "List of Insured Members | Person?",
        "What is the Room Rent amount or room type included in the policy?",
        "What is the Maternity Sum capping or sum insured?",
        "What is the policy start date?",
        "Total Sum Insured for the policy?",
        "Does the policy have copay?",
        "what is the policy inception date",
    ],
    "chunk_size": 800,
    "chunk_overlap": 500,
    "pipeline_kwargs": {
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
}


def log_time(task_name, start_time):
    elapsed_time = time.time() - start_time
    logger.info(f"{task_name} completed in {elapsed_time:.2f} seconds")


def load_pdf(file_path):
    logger.info("Starting PDF loading")
    start_time = time.time()
    loader = PyPDFLoader(file_path)
    pages = [page.page_content for page in loader.lazy_load()]
    log_time("PDF loading", start_time)
    return pages


def split_text_with_semantic_chunker(pages):
    logger.info("Starting semantic chunking")
    start_time = time.time()

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={"device": "cpu"},  # Specify device like "cuda" or "cpu"
        encode_kwargs={"normalize_embeddings": False}
    )

    # Create an instance of SemanticChunker with the embeddings model
    semantic_chunker = SemanticChunker(embeddings=embeddings)

    # Split each page text into semantic chunks
    chunks = []
    for page in pages:
        chunks.extend(semantic_chunker.split_text(page))

    log_time("Semantic chunking", start_time)
    return chunks


def prepare_documents(chunks):
    logger.info("Preparing documents")
    start_time = time.time()
    documents = [
        Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks
    ]
    log_time("Document preparation", start_time)
    return documents


def initialize_embeddings(model_name, device):
    logger.info("Initializing embedding model")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False},
    )
    log_time("Embedding model initialization", start_time)
    return embeddings


def populate_vector_store(embeddings, documents):
    logger.info("Adding documents to vector store")
    start_time = time.time()

    # Create FAISS index (assuming the embedding dimension is 384 for the model "sentence-transformers/all-MiniLM-l6-v2")
    embedding_dim = 384  # This should be the dimension of the embeddings model you're using
    index = faiss.IndexFlatL2(embedding_dim)

    # Create an in-memory document store
    docstore = InMemoryDocstore()

    # Create the FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={},
    )

    # Add documents to the vector store
    vector_store.add_documents(documents)
    log_time("Vector store population", start_time)
    return vector_store


def perform_similarity_search(vector_store, query):
    logger.info("Performing similarity search")
    start_time = time.time()
    # retriever = vector_store.as_retriever()
    # retrieved_docs = retriever.invoke(query)
    retrieved_docs = vector_store.as_retriever(top_k=5).invoke(query)
    log_time("Similarity search", start_time)
    return retrieved_docs

def initialize_llm(model_id, pipeline_kwargs):
    logger.info("Initializing LLM")
    start_time = time.time()
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs=pipeline_kwargs,
    )
    log_time("LLM initialization", start_time)
    return llm


def prepare_chat_context(retrieved_docs):
    logger.info("Preparing chat context")
    start_time = time.time()
    initial_context = "\n".join([doc.page_content for doc in retrieved_docs])
    log_time("Chat context preparation", start_time)
    return initial_context


def invoke_chat(llm, messages):
    # logger.info("Starting chat invocation")
    start_time = time.time()
    chat = ChatHuggingFace(llm=llm, verbose=True)
    response = chat.invoke(messages)
    # log_time("Chat invocation", start_time)
    return response


def parse_response(response):
    # logger.info("Parsing response")
    start_time = time.time()
    question = response.split("<start_of_turn>user")[-1].split("<end_of_turn>")[0]
    final_response = response.split("<start_of_turn>model")[-1]

    log_time("Response parsing", start_time)
    return question, final_response



def main():
    # Load and process the PDF
    pages = load_pdf(CONFIG["file_path"])
    chunks = split_text_with_semantic_chunker(pages)
    documents = prepare_documents(chunks)

    # Initialize embeddings and vector store
    # device = "cpu"  # Update with your device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    embeddings = initialize_embeddings(CONFIG["embedding_model"], device)
    vector_store = populate_vector_store(embeddings, documents)
    llm = initialize_llm(CONFIG["model_id"], CONFIG["pipeline_kwargs"])

    # Perform similarity search and process questions
    for question in CONFIG["questions"]:
        retrieved_docs = perform_similarity_search(vector_store, question)
        initial_context = prepare_chat_context(retrieved_docs)

        messages = [
            {"role": "user", "content": initial_context},
            {"role": "assistant", "content": "process the above text"},
            {"role": "user", "content": question},
        ]
        response = invoke_chat(llm, messages)

        # Parse and output results
        logger.info("Chat completed. Parsing response.")
        question, answer = parse_response(response.content)
        print(f"Question: {question}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

