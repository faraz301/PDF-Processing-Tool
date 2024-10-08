Overview
This project automates the extraction, processing, and summarization of content from PDF documents using a combination of Apache Airflow for workflow orchestration, Detectron2 for document segmentation, Tesseract OCR for text extraction, and a large language model (LLM) for generating summaries.

The project pipeline starts with a user uploading a PDF document, followed by automated document segmentation, text extraction, and summarization, with the final output being a consolidated summary of the document.

Key Features
Automated Workflow: Orchestrated with Apache Airflow to handle document processing and task scheduling.
Document Segmentation: Utilizes Detectron2 to intelligently divide the PDF into sections.
Text Extraction: Uses Tesseract OCR to extract text from segmented sections of the PDF.
Summarization: Implements LLMs to generate concise summaries of each section.
Combined Summary Output: The segmented and summarized text is returned to the user as a single output.
Technologies Used
Apache Airflow: For workflow management and task orchestration.
Detectron2: For document layout analysis and segmentation.
Tesseract OCR: For text recognition from images.
LLMs: To summarize extracted text into concise, readable summaries.
Python: The primary programming language for the implementation.
Project Workflow
User Uploads PDF: The process begins when the user uploads a PDF document.
Airflow Orchestration: Apache Airflow triggers the workflow, managing each task in the pipeline.
Document Segmentation: Detectron2 segments the PDF into defined sections, such as headers, paragraphs, and tables.
Text Extraction: Tesseract OCR extracts text from the segmented sections.
Summarization: LLMs are used to summarize each extracted section into a concise version.
Combined Summary Output: The final summary is delivered to the user as a consolidated text.
Usage Example
Upload a PDF file via the interface.
Airflow processes the PDF, segmenting it with Detectron2.
Extracted text is processed through Tesseract.
Summaries are generated using LLM.
The user receives a combined text summary as the output.
