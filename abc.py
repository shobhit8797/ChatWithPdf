from transformers import AutoTokenizer, AutoModelForCausalLM
from pdf2image import convert_from_path
from pytesseract import image_to_string
import sys
import re
from pdf2image import convert_from_path
from PIL import Image
from transformers import pipeline
import pdfplumber


def pdf_to_images(pdf_path, dpi=150):
    """Converts a PDF file to a list of images."""
    try:
        Image.MAX_IMAGE_PIXELS = None  # Handle large images
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
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
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text
                    else:
                        extracted_text = ""
                        break
                except Exception as page_error:
                    print(
                        f"Error extracting text from page {page_number}: {page_error}"
                    )

        # Check if text is empty, indicating a scanned PDF
        if not extracted_text.strip():
            print("Text not found, using OCR fallback.")
            images = convert_from_path(pdf_path, dpi=300)
            for i, image in enumerate(images):
                extracted_text += image_to_string(image)
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")

    return extracted_text


def map_beneficiaries_to_ids(text: str) -> dict:
    """Map beneficiary names to their corresponding Member IDs."""
    pattern = r"Beneficiary name: (.+?)\s.*?Member ID: (\d+)"
    matches = re.findall(pattern, text, re.DOTALL)
    return {name.strip(): member_id.strip() for name, member_id in matches}


def ask_question(nlp_pipeline, text, question):
    """Ask a question about the document text using a QA pipeline."""
    try:
        print(type(text))
        result = nlp_pipeline(question=question, context=text)
        print("result:",result)
        return result["answer"]
    except Exception as e:
        print(f"Error during question answering: {e}")
        sys.exit(1)


def find_valid_date(text: str) -> str:
    """Find the valid until date in the text."""
    match = re.search(r"Valid upto: ([^\n]+)", text)
    return match.group(1).strip() if match else "Date not found"


def main():
    pdf_path = "/Users/shobhitgoyal/Code/ChatWithPdf/pdf_files/INSURANCE_POLICYba39340d-6fc8-4a39-813e-21a40d690e49.pdf"

    # Extract text using pdfplumber
    print("Extracting text using pdfplumber...")
    extracted_text = extract_text_with_pdfplumber(pdf_path)

    if not extracted_text.strip():
        print("No text found in the document. Exiting.")
        sys.exit(1)

    # Map beneficiary names to Member IDs
    # print("Mapping beneficiary names to Member IDs...")
    # beneficiary_mapping = map_beneficiaries_to_ids(extracted_text)
    # print("Beneficiary Mapping:")
    # for name, member_id in beneficiary_mapping.items():
    #     print(f"{name}: {member_id}")

    # print("Initializing NLP pipeline...")
    # # nlp_pipeline = pipeline("question-answering", model="google/gemma-1.1-2b-it")
    # nlp_pipeline = pipeline("text-generation", model="google/gemma-1.1-2b-it")

    # # Example dynamic questions
    # questions = [
    #     "what are all the names of the beneficiaries?",
    #     "Who is the primary insured?",
    #     "INSURED NAMEs in the document?",
    # ]

    # for question in questions:
    #     print(f"Question: {question}")
    #     answer = ask_question(nlp_pipeline, extracted_text, question)
    #     print(f"Answer: {answer}")
    # Define context and question
    # Import necessary classes for model loading and quantization
    

    # Configure model quantization to 4-bit for memory and computation efficiency
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load the tokenizer for the Gemma 7B Italian model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")

    # Load the Gemma 7B Italian model itself, with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it")

    # pipe = pipeline("text-generation", model="google/gemma-1.1-2b-it")

    # Combine the context and question into a single input
    context =extracted_text[:800]
    question = "what are all the names of the beneficiaries?\n\nWho is the primary insured?\n\nTill when the policy is valid?"
    input_text = f"{context}\n\n{question}"
    
    # Tokenize the input text:
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    # Generate text using the model:
    outputs = model.generate(
        **input_ids,  # Pass tokenized input as keyword argument
        max_length=512,  # Limit output length to 512 tokens
    )
    # Decode the generated text:
    print('*'*100)
    print(tokenizer.decode(outputs[0]))
    print('*'*100)
    print("outputs:",outputs)





    # Generate a response
    # response = pipe(input_text, max_new_tokens=1500)  # You can adjust max_length as needed
    # print("Response:", response)
    # print('*'*100)

    # # Print the response
    # print(response[0]["generated_text"])

    # # Find the valid until date explicitly
    # valid_until_date = find_valid_date(extracted_text)
    # print(f"Valid Until Date: {valid_until_date}")


if __name__ == "__main__":
    main()
