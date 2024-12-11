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
    with pdfplumber.open(pdf_path) as pdf:
        # Extract the text
        for page in pdf.pages:
            print("Extracting text...")
            text = page.extract_text(layout=True, x_tolerance_ratio=0.5, y_tolerance_ratio=0.75)
            print(text)
            print("*" * 100)

            # Extract the data
            print("Extracting tables...")
            tables = page.extract_table() or []
            for table in tables:
                print(table)
            print("*" * 100)

    return {"text": text, "tables": tables}


def process_tables(tables):
    """Process extracted tables into DataFrames."""
    dataframes = []
    for table in tables:
        try:
            df = pd.DataFrame(table[1:], columns=table[0])  # Use first row as headers
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing table: {e}")
    return dataframes


def format_data(text, tables):
    """Format text and tables into a structured string."""
    formatted = "Document Content:\n\n"
    formatted += "Text Content:\n" + text + "\n\n"
    for i, table in enumerate(tables):
        formatted += f"Table {i + 1}:\n{table.to_string(index=False)}\n\n"
    return formatted


def initialize_model(model_id, device):
    """Initialize or fetch the model and tokenizer."""
    global cached_model, cached_tokenizer
    if not cached_model or not cached_tokenizer:
        cached_tokenizer = AutoTokenizer.from_pretrained(model_id)
        cached_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
    return cached_tokenizer, cached_model


def ask_question(tokenizer, model, context, question):
    """Ask a question to the model."""
    try:
        input_text = f"{context}\n\nQuestion: {question}"
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_new_tokens=500)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_start = answer.find("Answer:") + len("Answer:")
        return answer[answer_start:].strip()
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "Unable to generate an answer."


def main():
    pdf_path = "/Users/shobhitgoyal/Downloads/Insurance Policies/AADHAAR_CARD_FRONTneelam bharathi.pdf"
    model_id = "google/gemma-1.1-2b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract text and tables
    extracted = extract_text_and_tables(pdf_path)
    text = extracted["text"]
    tables = process_tables(extracted["tables"])

    # Format for model or use heuristics
    # formatted_input = format_data(text, tables)

    # # Answer questions
    # questions = [
    #     "List of Insured Members?",
    #     "What is the Room Rent?",
    #     "What is the Maternity Sum capping or sum insured?",
    #     "What is the policy start date?",
    #     "Sum insured for the policy?",
    # ]

    # # Optional: Use the model for more complex queries
    # print("\nInitializing model for advanced QA...")
    # tokenizer, model = initialize_model(model_id, device)
    # with ThreadPoolExecutor() as executor:
    #     results = list(
    #         executor.map(
    #             lambda q: ask_question(tokenizer, model, formatted_input, q), questions
    #         )
    #     )
    # for question, answer in zip(questions, results):
    #     print(f"\nQuestion: {question}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
