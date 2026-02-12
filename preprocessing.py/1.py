import pdfplumber
import nltk
import re
pdf_path=r"C:\Users\centu\OneDrive\Desktop\Coding\python\AIbot_prep.py\pdfplumber.py\document.pdf"
nltk.download("punkt")
nltk.download('stopwords') #to remove stop words like this, is, are etc.
all_text=[]
#clean_senetences function defined here:
def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text=page.extract_text()
        if text:
            all_text.append(text)
final_text="\n".join(all_text)
sentences=nltk.sent_tokenize(final_text)
print("Before cleaning: ",len(sentences))

cleaned_sentences = clean_sentences(sentences)
print("After cleaning: ", len(cleaned_sentences))

for i, s in enumerate(cleaned_sentences[:10]):
    print(i+1, ":" ,s)

