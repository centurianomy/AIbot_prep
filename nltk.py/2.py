import pdfplumber 
import nltk
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
nltk.download("punkt")
all_text=[]

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text=page.extract_text()
        if text:
            all_text.append(text)

full_text="\n".join(all_text)
sentences=nltk.sent_tokenize(full_text) # sentence type - string   &  sentences type - list of sentences 

for i, sentence in enumerate(sentences[:10]):  # Print first 10 sentences
    print(i+1, ":", sentence)

