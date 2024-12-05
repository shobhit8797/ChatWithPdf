import pytesseract
from pdf2image import convert_from_path
from transformers import LayoutLMForQuestionAnswering, LayoutLMTokenizer
import torch
from PIL import Image
import fitz  # PyMuPDF (for extracting raw text and metadata)


# Step 1: Convert PDF pages to images using pdf2image
def pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, 300)  # Convert to images with 300 DPI
    return pages


# Step 2: Extract text and layout information from each page
def extract_text_with_layout(pdf_path):
    doc = fitz.open(pdf_path)
    text_with_layout = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("dict")  # Extracts layout-aware text
        text_with_layout.append(text)

    return text_with_layout


# Step 3: OCR to extract text from images (for scanned PDFs)
def ocr_from_image(image):
    # Use Tesseract OCR to extract text from image
    text = pytesseract.image_to_string(image)
    return text


# Step 4: Load the LayoutLM model and tokenizer
def load_model():
    model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa")
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    return model, tokenizer


# Step 5: Perform Question-Answering using LayoutLM
def extract_info_from_layout(text_with_layout, model, tokenizer):
    extracted_data = []

    for page in text_with_layout:
        page_text = page["text"]  # Extract the text content from the layout

        # Tokenize and create the input for the model
        encoding = tokenizer(
            page_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Forward pass through LayoutLM model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)

        # Decode the answer
        answer = tokenizer.decode(input_ids[0][answer_start : answer_end + 1])
        extracted_data.append(answer)

    return extracted_data


# Step 6: Combine all and process the PDF
def process_pdf(pdf_path):
    # Step 1: Convert PDF to images
    images = pdf_to_images(pdf_path)

    # Step 2: Extract layout and text from the PDF (both scanned and text-based)
    text_with_layout = extract_text_with_layout(pdf_path)

    # Step 3: Load LayoutLM model
    model, tokenizer = load_model()

    # Step 4: Extract information using LayoutLM model
    extracted_info = extract_info_from_layout(text_with_layout, model, tokenizer)

    # Step 5: Perform OCR on images for scanned pages
    ocr_texts = []
    for image in images:
        ocr_text = ocr_from_image(image)
        ocr_texts.append(ocr_text)

    return extracted_info, ocr_texts


if __name__ == "__main__":
    pdf_path = "pdf_files/51 5220241203015212.pdf"
    extracted_info, ocr_texts = process_pdf("pdf_files/51 5220241203015212.pdf")
    print("Extracted Data Using LayoutLM Model:")
    for info in extracted_info:
        print(info)

    print("\nExtracted Text Using OCR (for scanned pages):")
    for ocr_text in ocr_texts:
        print(ocr_text)
    # print("Extracted information using LayoutLM:", extracted_info)
    # print("OCR texts from images:", ocr_texts)
