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
import os
from transformers import pipeline


def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 300) -> list[str]:
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

    return image_paths


def get_answers(nlp, image: str, question: str):
    ans = nlp(image, question)
    return ans


if __name__ == "__main__":
    image_output_folder = "output_images"
    # pdf_path = "/Users/shobhitgoyal/Downloads/INSURANCE_POLICYSwDxyZR6Ji-1733292505023.pdf"
    pdf_path = "/Users/shobhitgoyal/Downloads/INSURANCE_POLICY47eb906f-45bf-47ae-9375-1ee63113edc5.pdf"
    questions = ["What are the names in the document?", "What is the Age?", "what is the insurance or member id?"]
    
    nlp = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
    )
    
    images_path = pdf_to_images(pdf_path, image_output_folder)
    # images_path = ["output_images/51_428f6602fb-4813-45f6-b4b2-239a1122996f/page_1.png"]
    for image_path in images_path:
        for question in questions:
            ans = get_answers(nlp,image_path, question)
            print("ans: ", ans)
            ans = ans[0]
            if ans['score'] > 0.85:
                print(f"Question: {question}")
                print(f"Answer: {ans['answer']}")
                print(f"Answer: {ans['score']}")
        
