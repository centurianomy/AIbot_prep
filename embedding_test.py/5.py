from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["Hello world", "AI is amazing"]

embeddings = model.encode(sentences)

print(type(sentences))        # list
print(type(sentences[0]))     # str

print(type(embeddings))       # numpy array
print(type(embeddings[0]))    # numpy array
