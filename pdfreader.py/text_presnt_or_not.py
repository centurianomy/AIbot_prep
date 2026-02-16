import pdfplumber
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
with pdfplumber.open(pdf_path) as pdf: #pdf-> variable (name of the file-file handelling)
    page=pdf.pages[0]
    #page=pdf.pages[1]
    #page=pdf.pages[2]
    text=page.extract_text()
    if text:
        print("This page contains text.")
        print(text)
    else:
        print("This page doesn't contains text.")
