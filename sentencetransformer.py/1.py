
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
sentence="I love machine learning." #Every sentence gets its own 384-number meaning vector, even when processed together.

embedding=model.encode(sentence)
print(embedding) #prints the 384-number meaning vector for the sentence
print("Embedding length: ",len(embedding)) #prints the length of the embedding, which should be 384 for this model.
