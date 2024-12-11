import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pdfplumber
import torch
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global cache for the model and tokenizer
cached_model = None
cached_tokenizer = None


def extract_text_and_tables(pdf_path, max_pages=5):
    """Extract text and tables from the PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(layout=True) or ""

    return text

    # extracted_data = {"text": "", "tables": []}
    # try:
    #     with pdfplumber.open(pdf_path) as pdf:
    #         for page in pdf.pages[:max_pages]:
    #             extracted_data["text"] += page.extract_text(layout=True) or ""
    #             tables = page.extract_tables()
    #             if tables:
    #                 extracted_data["tables"].extend(tables)
    # except Exception as e:
    #     print(f"Error extracting text and tables: {e}")
    # return extracted_data


def initialize_model(model_id, device):
    """Initialize or fetch the model and tokenizer."""
    global cached_model, cached_tokenizer
    if not cached_model or not cached_tokenizer:
        cached_tokenizer = AutoTokenizer.from_pretrained(model_id)
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
# def ask_question(tokenizer, model, context, question, max_length=512):
#     """Ask a question to the model with limited context size."""
#     try:
#         print(f"Question: {question}")
#         input_text = f"{context}\n\nQuestion: {question}"
#
#         # Tokenize and truncate input to max_length
#         inputs = tokenizer(
#             input_text, return_tensors="pt", max_length=max_length, truncation=True
#         ).to(model.device)
#
#         outputs = model.generate(inputs.input_ids, max_new_tokens=100)
#         answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("answer:",answer)
#
#         answer_start = answer.find("Answer:") + len("Answer:")
#         print(f"Answer: {answer[answer_start:].strip()}")
#         return answer[answer_start:].strip()
#     except Exception as e:
#         print(f"Error during question answering: {e}")
#         return "Unable to generate an answer."


def main():
    start = time.time()
    pdf_path = "./p4.pdf"
    # model_id = "google/gemma-2-2b-it"
    model_id = "google/gemma-2-2b-it"
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Device Being used: {device}")

    # Extract text and tables
    # Extract text and tables
    extracted = extract_text(pdf_path)
    # print(f"Extracted Text:\n{extracted}")
    print(f"Time elapsed: {time.time() - start}")

    # Answer questions
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

    # Optional: Use the model for more complex queries
    print("\nInitializing model for advanced QA...")
    tokenizer, model = initialize_model(model_id, device)

    print(f"Time elapsed: {time.time() - start}")
    for ques in questions:
        ask_question(tokenizer, model, extracted, ques)
    print(f"Time elapsed: {time.time() - start}")
    # for question, answer in zip(questions, results):
    # print(f"\nQuestion: {question}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
