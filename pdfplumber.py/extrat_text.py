import pdfplumber
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
with pdfplumber.open(pdf_path) as pdf: 
    first_page=pdf.pages[0] #accessing first page (index 0)
    text=first_page.extract_text() #extracts text from first page
    print(text) #prints text of first page only

    #first_page-> object
    #text-> variable
    #pdf.pages-> predefined attribute (from pdfplumber)