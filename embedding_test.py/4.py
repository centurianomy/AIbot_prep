from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of sentences (like a small document)
sentences = [
    "Machine learning is a subset of AI.",
    "I enjoy playing cricket.",
    "Artificial intelligence is transforming industries.",
    "Pizza is my favorite food."
]

# Sentence we want to compare against
query = "AI is changing the world."

# Convert sentences and query into embeddings
sentence_embeddings = model.encode(sentences)
query_embedding = model.encode(query)

# Compute similarity between query and all sentences
similarities = util.cos_sim(query_embedding, sentence_embeddings)

# Print similarity scores
for i, score in enumerate(similarities[0]):
    print(f"Similarity with sentence {i + 1}: {score}")
