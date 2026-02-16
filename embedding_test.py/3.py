from sentence_transformers import SentenceTransformer, util #util is needed for similarity computation

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two sentences
sentence1 = "Artificial intelligence is powerful."
sentence2 = "I am from New Zealand."

# Convert both sentences into embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# Compute cosine similarity between the two embeddings
similarity_score = util.cos_sim(embedding1, embedding2)

# Print similarity score
print("Similarity score:", similarity_score)

# o/p: Similarity score: tensor([[0.0202]])