import logging
import re
import time

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants and Configurations
CONFIG = {
    "file_path": "./p3.pdf",
    "model_id": "google/gemma-2-2b-it",
    "embedding_model": "sentence-transformers/all-MiniLM-l6-v2",
    "questions": [
        "List of Insured Members | Person?",
        "What is the Room Rent amount or room type included in the policy?",
        "What is the Maternity Sum capping or sum insured?",
        "What is the policy start date?",
        "Sum insured for the policy?",
        "Does the policy have copay?",
        # "what is the policy inception date",
        # "what is the waiting period? and list down the categories for waiting periods",
        # "what is the waiting period for specific disease waiting periods",
        "what is the waiting period for maternity package ",
    ],
    "chunk_size": 800,
    "chunk_overlap": 200,
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


def split_text(pages, chunk_size, chunk_overlap):
    logger.info("Starting text splitting")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [chunk for page in pages for chunk in text_splitter.split_text(page)]
    log_time("Text splitting", start_time)
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
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=documents)
    log_time("Vector store population", start_time)
    return vector_store


def perform_similarity_search(vector_store, query):
    logger.info("Performing similarity search")
    start_time = time.time()
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(query)
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
    logger.info("Starting chat invocation")
    start_time = time.time()
    chat = ChatHuggingFace(llm=llm, verbose=True)
    response = chat.invoke(messages)
    log_time("Chat invocation", start_time)
    return response


def parse_response(response):
    logger.info("Parsing response")
    start_time = time.time()
    question = response.split("<start_of_turn>user")[-1].split("<end_of_turn>")[0]
    final_response = response.split("<start_of_turn>model")[-1]

    log_time("Response parsing", start_time)
    return question, final_response


def main():
    # Load and process the PDF
    pages = load_pdf(CONFIG["file_path"])
    chunks = split_text(pages, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
    documents = prepare_documents(chunks)

    # Initialize embeddings and vector store
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    embeddings = initialize_embeddings(CONFIG["embedding_model"], device)
    vector_store = populate_vector_store(embeddings, documents)

    # Perform similarity search and initialize LLM
    llm = initialize_llm(CONFIG["model_id"], CONFIG["pipeline_kwargs"])

    # Process each question and retrieve responses
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
