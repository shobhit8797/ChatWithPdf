import re
import sys

import pdfplumber
import torch
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import image_to_string
from transformers import AutoModelForCausalLM, AutoTokenizer


def pdf_to_images(pdf_path, dpi=150, max_pages=5):
    """Convert the first few pages of a PDF file to images."""
    try:
        Image.MAX_IMAGE_PIXELS = None  # Handle large images
        return convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        print(f"Error during PDF to image conversion: {e}")
        sys.exit(1)


def extract_text_with_pdfplumber(pdf_path: str, max_pages=5) -> str:
    """Extract text from the first few pages of a PDF using pdfplumber."""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages[:max_pages]):
                extracted_text += page.extract_text() or ""

        # Fallback to OCR if no text is found
        if not extracted_text.strip():
            print("No text found using pdfplumber. Switching to OCR...")
            images = pdf_to_images(pdf_path, dpi=300, max_pages=max_pages)
            extracted_text = "\n".join(image_to_string(image) for image in images)
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")

    return extracted_text.strip()


def map_beneficiaries_to_ids(text: str) -> dict:
    """Extract beneficiary names and Member IDs."""
    pattern = r"Beneficiary name: (.+?)\s.*?Member ID: (\d+)"
    return {
        name.strip(): member_id.strip()
        for name, member_id in re.findall(pattern, text, re.DOTALL)
    }


def initialize_model_pipeline(model_id: str, device):
    """Initialize the tokenizer and model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    return tokenizer, model


def ask_questions(tokenizer, model, text: str, questions: list) -> dict:
    """Ask multiple questions using the text-generation pipeline."""
    answers = {}
    try:
        for question in questions:
            input_text = f"{text}\n\nQuestion: {question}"
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

            # Generate a response
            outputs = model.generate(inputs, max_new_tokens=500)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer part
            answer_start = answer.find("Answer:") + len("Answer:")
            answers[question] = answer[answer_start:].strip()
    except Exception as e:
        print(f"Error during question answering: {e}")
        return {question: "Unable to generate an answer."}

    return answers


def main():
    pdf_path = "./document.pdf"
    print("Extracting text from the document...")
    extracted_text = extract_text_with_pdfplumber(pdf_path, max_pages=5)

    if not extracted_text.strip():
        print("No text found in the document. Exiting.")
        sys.exit(1)

    print("Initializing model pipeline...")
    model_id = "google/gemma-1.1-2b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = initialize_model_pipeline(model_id, device)
    questions = [
        "What are the names of all beneficiaries?",
        "What is the validity date of the document?",
        "Who is the primary insured?",
    ]

    print("Asking questions...")
    answers = ask_questions(tokenizer, model, extracted_text, questions)

    # Print all answers
    for question, answer in answers.items():
        print(f"\nQuestion: {question}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
