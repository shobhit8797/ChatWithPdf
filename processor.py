import pytesseract
from pdf2image import convert_from_path
from transformers import LayoutLMForQuestionAnswering, RobertaTokenizer
import torch
from PIL import Image
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


# return text
def extract_text_with_layout(pdf_path: str) -> List[dict]:
    """
    Extracts text and layout information from a PDF using pdfplumber.
    Returns a list of dictionaries, where each dictionary represents a page with text and bounding boxes.
    """
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_data = {"page_number": i + 1, "words": []}
            words = page.extract_words()  # Extract words with their bounding boxes

            for word in words:
                page_data["words"].append(
                    {
                        "text": word["text"],
                        "bbox": [
                            int(word["x0"]),  # Left
                            int(word["top"]),  # Top
                            int(word["x1"]),  # Right
                            int(word["bottom"]),  # Bottom
                        ],
                    }
                )

            pages_data.append(page_data)

    return pages_data


def extract_info_from_layout(pages_data, model, tokenizer):
    """
    Extracts structured information using the LayoutLM model.
    Processes text and bounding box data for each page.
    """
    extracted_data = []

    for page in pages_data:
        print(f"Processing page {page['page_number']}...")

        # Extract text and bounding boxes for the page
        words = page["words"]
        if not words:
            print(f"No words found on page {page['page_number']}. Skipping.")
            continue

        print("word:", words)
        texts = [word["text"] for word in words]
        bboxes = [word["bbox"] for word in words]

        # Tokenize the text and align bounding boxes with tokens
        encoding = tokenizer(
            texts,
            boxes=bboxes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        bbox = encoding["bbox"]  # Bounding box tensor

        # Forward pass through LayoutLM
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)

        # Decode the answer
        answer = tokenizer.decode(input_ids[0][answer_start : answer_end + 1])
        print(f"Answer for page {page['page_number']}: {answer}")
        extracted_data.append({"page_number": page["page_number"], "answer": answer})

    return extracted_data


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
    model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa")
    # tokenizer = LayoutLMTokenizer.from_pretrained("impira/layoutlm-document-qa")
    tokenizer = RobertaTokenizer.from_pretrained("impira/layoutlm-document-qa")

    return model, tokenizer


# Step 5: Perform Question-Answering using LayoutLM (Placeholder)
# def extract_info_from_layout(text_with_layout, model, tokenizer):
#     extracted_data = []

#     for page in text_with_layout:
#         print(page)
#         page_text = page['text']  # Extract the text content from the layout

#         # Tokenize and create the input for the model
#         encoding = tokenizer(page_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         input_ids = encoding["input_ids"]
#         attention_mask = encoding["attention_mask"]

#         # Forward pass through LayoutLM model
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         answer_start = torch.argmax(outputs.start_logits)
#         answer_end = torch.argmax(outputs.end_logits)

#         # Decode the answer
#         answer = tokenizer.decode(input_ids[0][answer_start:answer_end+1])
#         print("answer:", answer)
#         extracted_data.append(answer)

#     return extracted_data


# Step 6: Combine all steps and process the PDF
def process_pdf(pdf_path: str, output_folder: str) -> Tuple[List[dict], List[str]]:
    """
    Processes a PDF to extract information using LayoutLM and OCR as a fallback.
    """
    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, output_folder)

    # Extract text and layout information
    pages_data = extract_text_with_layout(pdf_path)

    # Load LayoutLM model
    model, tokenizer = load_model()

    # Extract information using LayoutLM model
    extracted_info = extract_info_from_layout(pages_data, model, tokenizer)

    # Perform OCR on images for scanned pages
    ocr_texts = [ocr_from_image(image) for image in image_paths]

    return extracted_info, ocr_texts


# Usage
if __name__ == "__main__":
    pdf_path = "pdf_files/51 5220241203015212.pdf"  # Path to the PDF file
    output_folder = "output_images"  # Folder to save images

    # # pdf_to_images(pdf_path, output_folder)

    # print(image_to_string(Image.open("output_images/Santosh_Preveena_2015_care_fi_001/page_27.png")))

    # Process the PDF
    extracted_info, ocr_texts = process_pdf(pdf_path, output_folder)

    print("extracted_info:", extracted_info)

    # Save the extracted text to a file
    output_text_file = "output_text.txt"
    with open(output_text_file, "w", encoding="utf-8") as f:
        for ocr_text in ocr_texts:
            f.write(ocr_text)
        print(f"Extracted OCR text saved to {output_text_file}")
