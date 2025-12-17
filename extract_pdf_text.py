import pypdf
import os

pdf_files = [
    "Assignment_and_Background.pdf",
    "self-assessment-guide.pdf",
    "software_submission_guidelines.pdf"
]

for pdf_file in pdf_files:
    print(f"--- START OF {pdf_file} ---")
    try:
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            print(page.extract_text())
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
    print(f"--- END OF {pdf_file} ---")
