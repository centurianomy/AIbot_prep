#Centroid-Based Summarization
import numpy as np  # For mathematical operations
from sentence_transformers import SentenceTransformer, util

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences (pretend these came from your cleaned PDF)
sentences = [
    "Artificial intelligence is transforming industries.",
    "AI systems are widely used in healthcare.",
    "Football is a popular sport worldwide.",
    "Machine learning is a subset of artificial intelligence.",
    "Hospitals use AI for disease prediction."
]

# Step 1: Convert sentences into embeddings
# Output: One embedding vector per sentence
embeddings = model.encode(sentences)

# Step 2: Compute document embedding (mean of all sentence embeddings)
# This gives us a single vector representing overall document meaning
document_embedding = np.mean(embeddings, axis=0)

# Step 3: Compute similarity between each sentence and the document
similarities = util.cos_sim(document_embedding, embeddings)

# similarities is a matrix, so we take first row
similarity_scores = similarities[0]

# Step 4: Rank sentences by similarity score (descending order)
# argsort() returns indices sorted in ascending order
# [::-1] reverses it to descending order
ranked_indices = np.argsort(similarity_scores)[::-1]

# Step 5: Select top K sentences for summary
top_k = 3  # You can change this number
summary_sentences = []

for i in ranked_indices[:top_k]:
    summary_sentences.append(sentences[i])

# Print summary
print("=== SUMMARY ===")
for sentence in summary_sentences:
    print("-", sentence)
