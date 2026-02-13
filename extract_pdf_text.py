import pypdf
import os

pdf_path = "Matlab_Assignment_4.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: File not found at {pdf_path}")
    exit(1)

try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text(extraction_mode="layout") + "\n"
    print(text)
except Exception as e:
    print(f"Error extracting text: {e}")
