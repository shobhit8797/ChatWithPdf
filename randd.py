import time
from uuid import uuid4

import faiss
import pdfplumber
import torch
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Global cache for the model and tokenizer
cached_model = None
cached_tokenizer = None

pdf_path = "./p4.pdf"

model_id = "google/gemma-2-2b-it"
device = torch.device("mps" if torch.mps.is_available() else "cpu")
# sentence-transformers/all-mpnet-base-v2
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {"device": "mps" if torch.mps.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": False}

questions = [
    "List of Insured Members?",
    "What is the Room Rent included in the policy?",
    "What is the Maternity Sum capping or sum insured?",
    "What is the policy start date?",
    "Sum insured for the policy?",
    "Does the policy have copay?",
    "what is the policy inception date",
    "what is the waiting period? and list down the categories for waiting periods",
    "what is the waiting period for specific disease waiting periods",
    "what is the waiting period for maternity package ",
]


def extract_text(pdf_path, max_pages=5):
    """Extract text and tables from the PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(layout=True) or ""

    return text


def initialize_model(model_id, device):
    """Initialize or fetch the model and tokenizer."""
    global cached_model, cached_tokenizer
    if not cached_model or not cached_tokenizer:
        cached_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding=True,
            truncation=True,
            max_length=512,
            safe_serialization=False,
        )
        cached_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        ).to(device)
    return cached_tokenizer, cached_model


def ask_question(tokenizer, model, context, question):
    """Ask a question to the model."""
    try:
        print(f"Question: {question}")
        input_text = f"{context}\n\nQuestion: {question}"
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_new_tokens=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_start = answer.find("Answer:") + len("Answer:")
        print(f"Answer: {answer[answer_start:].strip()}")
        return answer[answer_start:].strip()
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "Unable to generate an answer."


def main():
    start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    print(f"Device Being used: {device}")
    extracted_text = extract_text(pdf_path)
    print(f"extracted_text length: {len(extracted_text)}")
    docs = text_splitter.create_documents(extracted_text)
    print("Splitting Done")
    # docs = text_splitter.split_text(extract_text(pdf_path))

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    print("Embeddings created")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("Vector Storage Initialized")
    # db = FAISS.from_documents(docs, embeddings)
    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(documents=docs, ids=uuids)
    print("vector_store updated")

    tokenizer, model = initialize_model(model_id, device)
    print("Model amd Tokenizer Initialized")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_tensors="pt",
        max_length=512,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "temperature": 0.7,
            "max_length": 512,
        },
        device=device,
        truncation=True,
    )
    print("Pipeline created")
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.7, "max_length": 512},
    )
    print("llm initialized")
    retriever = VectorStoreRetriever(vectorstore=vector_store)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("RetrievalQA initialized")
    qa.invoke("Write an educational story for young children.")
    print("qa invoked")


if __name__ == "__main__":
    main()
