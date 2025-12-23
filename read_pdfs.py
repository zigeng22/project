import pdfplumber
import os

project_dir = r'c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project'

# Read the project requirements PDF
pdf_files = [
    'Project Requirements 2025.pdf',
    'project sample 1.pdf', 
    'project sample 2.pdf'
]

for pdf_file in pdf_files:
    pdf_path = os.path.join(project_dir, pdf_file)
    print(f'\n{"="*80}')
    print(f'FILE: {pdf_file}')
    print(f'{"="*80}\n')
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    print(f'--- Page {i+1} ---')
                    print(text)
                    print()
    except Exception as e:
        print(f'Error reading {pdf_file}: {e}')
