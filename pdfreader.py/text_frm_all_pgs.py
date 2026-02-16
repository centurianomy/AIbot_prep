import pdfplumber
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
all_text=[] #list to store text from all pages
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        print(page) #page is like (itterater i)
        text=page.extract_text()
        if text:
            all_text.append(text)
        else:
            print("No text available on this page.")    
final_text="\n".join(all_text) #final_text = final_text + all_text
print(final_text)            

