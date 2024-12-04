import os
from typing import List, Tuple
from pdf2image import convert_from_path
from pytesseract import image_to_string
import pdfplumber


# Step 1: Convert PDF pages to images using pdf2image
def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 300) -> List[str]:
    """
    Converts PDF pages to images and saves them to the output folder.
    """
    
    file_name = pdf_path.split("/")[-1].split(".")[0].replace(" ", "_")
    output_folder = f"{output_folder}/{file_name}"
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        print(f"Saved {image_path}")

    return image_paths


# Step 2: Extract text and layout information from the PDF
def extract_text_with_layout(pdf_path: str) -> str:
    """
    Extracts text and layout information from a PDF using pdfplumber and OCR as a fallback.
    """
    text = ""

    # First try to use pdfplumber for text-based extraction
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text(layout=True)
            if extracted_text:
                text += extracted_text
            else:
                print("Empty page detected, OCR fallback will be used.")
    
    print("Text extracted using pdfplumber.", len(text))

    # Check if text is empty, indicating a scanned PDF
    if not text.strip():
        print("Text not found, using OCR fallback.")
        images = convert_from_path(pdf_path, dpi=300)
        for i, image in enumerate(images):
            text += f"\n--- Page {i + 1} ---\n"
            text += image_to_string(image)

    return text


# Step 3: OCR to extract text from a single image
def ocr_from_image(image_path: str) -> str:
    """
    Performs OCR on a given image and returns extracted text.
    """
    print(f"Performing OCR on {image_path}")
    return image_to_string(image_path)


# Step 4: Load the LayoutLM model and tokenizer (Placeholder)
def load_model():
    """
    Loads the LayoutLM model and tokenizer.
    Replace with actual implementation when LayoutLM integration is added.
    """
    print("Loading LayoutLM model and tokenizer (placeholder).")
    model = None  # Replace with actual model loading
    tokenizer = None  # Replace with actual tokenizer loading
    return model, tokenizer


# Step 5: Perform Question-Answering using LayoutLM (Placeholder)
def extract_info_from_layout(text_with_layout: str, model, tokenizer) -> List[dict]:
    """
    Extracts structured information from text using LayoutLM.
    Replace with actual implementation for QA extraction.
    """
    print("Extracting information using LayoutLM (placeholder).")
    # Placeholder for LayoutLM integration
    extracted_data = [{"page_text": text_with_layout}]
    return extracted_data


# Step 6: Combine all steps and process the PDF
def process_pdf(pdf_path: str, output_folder: str) -> Tuple[List[dict], List[str]]:
    """
    Processes a PDF to extract information using LayoutLM and OCR as a fallback.
    """
    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, output_folder)

    # Extract layout and text from the PDF
    text_with_layout = extract_text_with_layout(pdf_path)

    # Load LayoutLM model
    model, tokenizer = load_model()

    # Extract information using LayoutLM model
    extracted_info = extract_info_from_layout(text_with_layout, model, tokenizer)

    # Perform OCR on images for scanned pages
    ocr_texts = [ocr_from_image(image) for image in image_paths]

    return extracted_info, ocr_texts


# Usage
if __name__ == "__main__":
    pdf_path = "pdf_files/36 1182a4999f-42d2-4d27-a1bf-022a6281fa00.pdf"  # Path to the PDF file
    output_folder = "output_images"  # Folder to save images
    
    print(extract_text_with_layout(pdf_path))

    # Process the PDF
    # extracted_info, ocr_texts = process_pdf(pdf_path, output_folder)

    # Save the extracted text to a file
    # output_text_file = "output_text.txt"
    # with open(output_text_file, "w", encoding="utf-8") as f:
    #     for ocr_text in ocr_texts:
    #         f.write(ocr_text)
    #     print(f"Extracted OCR text saved to {output_text_file}")