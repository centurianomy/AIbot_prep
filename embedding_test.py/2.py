from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of sentences
sentences = [
    "Cats are cute animals.",
    "Dogs are loyal animals.",
    "I like programming."
]

# Encode all sentences at once
embeddings = model.encode(sentences)

# Print number of embeddings
print("Number of embeddings:", len(embeddings))

# Print embedding of first sentence
print("First sentence embedding length:", len(embeddings[0]))
