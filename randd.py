from transformers import AutoTokenizer, AutoModelForCausalLM
from pdf2image import convert_from_path
from pytesseract import image_to_string
import sys
import re
from PIL import Image
import pdfplumber
import torch

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


def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number in range(len(pdf.pages)):
                page = pdf.pages[page_number]
                page_text = page.extract_text() or ""
                extracted_text += page_text

        if not extracted_text.strip():  # Check if text is empty
            print("Text not found, using OCR fallback.")
            images = pdf_to_images(pdf_path, dpi=300)
            for image in images:
                extracted_text += image_to_string(preprocess_image(image))
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")

    return extracted_text.strip()


def map_beneficiaries_to_ids(text: str) -> dict:
    """Map beneficiary names to their corresponding Member IDs."""
    pattern = r"Beneficiary name: (.+?)\s.*?Member ID: (\d+)"
    matches = re.findall(pattern, text, re.DOTALL)
    return {name.strip(): member_id.strip() for name, member_id in matches}


def ask_question(model, tokenizer, text: str, question: str) -> str:
    """Ask a question about the document text using the loaded model."""
    try:
        # Encode the input text and question as a single prompt
        prompt = f"{text}\n\n{question}"
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        # Generate text using the model
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

        # Decode the output and return the answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print(f"Error during question answering: {e}")
        sys.exit(1)


def find_valid_date(text: str) -> str:
    """Find the valid until date in the text."""
    match = re.search(r"Valid upto: ([^\n]+)", text)
    return match.group(1).strip() if match else "Date not found"


def main():
    pdf_path = "./document.pdf"

    # Extract text using pdfplumber
    extracted_text = extract_text_with_pdfplumber(pdf_path)

    if not extracted_text.strip():
        print("No text found in the document. Exiting.")
        sys.exit(1)

    # Initialize NLP model and tokenizer
    print("Initializing NLP model...")
    model_id = "google/gemma-1.1-2b-it"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        sys.exit(1)

    # Define context and question
    context = extracted_text[:800]  # Adjust the context length as needed
    question = "What are all the names of the beneficiaries?\nWho is the primary insured?"

    # Ask the question and print the answer
    answer = ask_question(model, tokenizer, context, question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()