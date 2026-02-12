import pdfplumber
pdf_path= r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"

with pdfplumber.open(pdf_path) as pdf:
    total_pages=len(pdf.pages)
    print("Total number of pages in this pdf: ",total_pages)    

