import re
import sys

import pdfplumber
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from pytesseract import image_to_string
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from htmlTemplates import bot_template, css, user_template


def pdf_to_images(pdf_path, dpi=150):
    """Converts a PDF file to a list of images."""
    try:
        Image.MAX_IMAGE_PIXELS = None  # Handle large images
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error during PDF to image conversion: {e}")
        sys.exit(1)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the image to RGB format."""
    return image.convert("RGB")


def get_pdf_text(pdf_docs):
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_docs) as pdf:
            for page_number in range(len(pdf.pages)):
                page = pdf.pages[page_number]
                page_text = page.extract_text() or ""
                extracted_text += page_text
        if not extracted_text.strip():  # Check if text is empty
            print("Text not found, using OCR fallback.")
            for image in pdf_to_images(pdf_docs, dpi=150):  # Lower DPI to save memory
                extracted_text += image_to_string(preprocess_image(image))
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")
    return extracted_text.strip()


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="google/gemma-1.1-2b-it",
        device=0,
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(
        repo_id="google/gemma-1.1-2b-it",
        model_kwargs={"temperature": 0.5, "max_length": 256},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
