import json
import os

import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from unstructured.cleaners.core import clean
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json


def initialize_llm(
    model_id="google/gemma-2-2b-it",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.5,
        "do_sample": True,
    },
    max_batch_size=4,
):
    """
    Initializes the Hugging Face pipeline for text generation.
    Args:
        model_id: The ID of the Hugging Face model to use.
        pipeline_kwargs: Keyword arguments to pass to the `pipeline` function.
        max_batch_size: Maximum batch size for the pipeline.
    Returns:
        A HuggingFacePipeline object.
    """
    torch.mps.empty_cache()  # Clear GPU memory before loading the model
    hf_pipeline = pipeline(task="text-generation", model=model_id, **pipeline_kwargs)
    return HuggingFacePipeline(pipeline=hf_pipeline)


def parse_pdf(file_path):
    """
    Parses the PDF file into a list of elements.
    Args:
        file_path: Path to the PDF file.
    Returns:
        A list of elements extracted from the PDF.
    """
    unstructured_model_name = "yolox"  # yolox | layout_v1.1.0 | detectron2_onnx
    return partition_pdf(
        strategy="hi_res",
        filename=file_path,
        hi_res_model_name=unstructured_model_name,
        infer_table_structure=True,
        extract_image_block_types=["Table"],
        image_output_dir_path="./",
        languages=["eng"],
        dpi=300,
    )


def process_parsed_pdf(elements):
    """
    Processes the parsed PDF elements and extracts relevant information.
    Args:
        elements: A list of elements extracted from the PDF.
    Returns:
        A dictionary containing extracted tables and text.
    """
    json_elements = elements_to_json(elements=elements, indent=4)
    parsed_elements = {"table": [], "text": []}
    for entry in json.loads(json_elements):
        if entry["type"] == "Table":
            parsed_elements["table"].append(
                {"html": entry["metadata"]["text_as_html"], "ocr": entry["text"]}
            )
        elif entry["type"] in ["Text", "NarrativeText"]:
            parsed_elements["text"].append(entry["text"])
    return parsed_elements


def create_prompt():
    """
    Creates a prompt template for summarizing tables.
    Returns:
        A PromptTemplate object.
    """
    template = """
        You are an assistant tasked with summarizing tables and OCR text.
        Provide a concise summary of the table's important details, including key amounts, names, and other critical information.
        Ensure the summary is clear and complete, combining insights from both the table structure and OCR text.
        Do not include any extraneous text or comments. Your response should only contain the summary.
        Table (HTML format): {table}
        OCR Text: {text}
    """
    return PromptTemplate.from_template(template)


def summarize_tables(parsed_elements, llm):
    """
    Summarizes the tables in the parsed elements using the provided LLM.
    Args:
        parsed_elements: A dictionary containing extracted tables and text.
        llm: The LLM to use for summarization.
    Returns:
        A list of table summaries.
    """
    prompt = create_prompt()
    chain = prompt | llm
    table_summaries = []
    for table_info in parsed_elements["table"]:
        try:
            summary = ""
            for chunk in chain.stream(
                {"table": table_info["html"], "text": table_info["ocr"]}
            ):
                summary += clean(chunk, extra_whitespace=True)
            table_summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing table: {e}")
            table_summaries.append(f"Error summarizing table: {e}")
    return table_summaries


if __name__ == "__main__":
    file_path = "p1.pdf"
    llm = initialize_llm()
    chunks = parse_pdf(file_path)
    parsed_elements = process_parsed_pdf(chunks)
    table_summaries = summarize_tables(parsed_elements, llm)
    for summary in table_summaries:
        print(summary)
