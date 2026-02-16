import nltk #Natural Lang Tool Kit
nltk.download("punkt") #smart tokenizer (knows . , ! ? etc are sentence boundaries)
text="Hello world. This is a test sentence for tokenization."
sentences=nltk.sent_tokenize(text) #sent_tokenize() -- return a list of sentences
for i, sentence in enumerate(sentences):
    print(i+1,":", sentence)