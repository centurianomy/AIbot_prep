import pdfplumber
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages): 
        #enunerate is pythons clean replacement for i=0, i<n, i++
        #pdf.pages-> predefined attribute (from pdfplumber)
        #page-> object
        #pdf-> variable
        #page_number-> user defined
        print("Reading page",page_number+1)  #page_number starts from 0, so adding 1 for user-friendly

        #page_number is like i=0,1,2...
        
