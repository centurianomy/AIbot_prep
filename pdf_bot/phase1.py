# ==============================
# OFFLINE PDF EXTRACTIVE SUMMARIZER (MVP)
# ==============================

import pdfplumber                  # For PDF text extraction
import nltk                        # For sentence splitting
import numpy as np                 # For numerical operations
import re                          # For cleaning text
from sentence_transformers import SentenceTransformer, util  # For embeddings + similarity

# Load embedding model ONCE globally
model = SentenceTransformer("all-MiniLM-L6-v2")

# Download sentence tokenizer (only first time)
nltk.download("punkt")

# ------------------------------
# 1️⃣ Extract text from PDF
# ------------------------------


def extract_text_from_pdf(pdf_path):
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

    return "\n".join(all_text)


# ------------------------------
# 2️⃣ Clean sentences
# ------------------------------

def clean_sentences(sentences):
    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Remove short sentences
        if len(sentence.split()) < 3:
            continue

        # Remove sentences without alphabets
        if not re.search(r"[a-zA-Z]", sentence):
            continue

        cleaned.append(sentence)

    return cleaned

# ------------------------------
# CHUNKING FUNCTION
# ------------------------------

def chunk_sentences(sentences, chunk_size=50, overlap=10):

    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

# ------------------------------
# 3️⃣ MMR-based summarization
# ------------------------------

def summarize(sentences, top_k=5, lambda_param=0.7):

    # Load embedding model
    #model loads inside summarize() every time: This is inefficient. so instead load it once globally and reuse it.
    #model = SentenceTransformer("all-MiniLM-L6-v2")


    # Convert sentences to embeddings
    embeddings = model.encode(sentences)

    # Compute document embedding
    document_embedding = np.mean(embeddings, axis=0)

    # Compute similarity of each sentence with document
    doc_similarities = util.cos_sim(document_embedding, embeddings)[0]

    selected_indices = []

    # Step 1: Select most relevant sentence
    first_index = np.argmax(doc_similarities)
    selected_indices.append(first_index)

    # Step 2: Select remaining sentences using MMR
    for _ in range(top_k - 1):

        mmr_scores = []

        for i in range(len(sentences)):

            if i in selected_indices:
                mmr_scores.append(-1)
                continue
            # 🔥 EXTRA REDUNDANCY CHECK
            # If too similar to any selected sentence, skip it completely
            too_similar = False
            for j in selected_indices:
                similarity = util.cos_sim(embeddings[i], embeddings[j])[0][0]
                if similarity > 0.9:
                    too_similar = True
                    break

            if too_similar:
                mmr_scores.append(-1)
                continue
            
            relevance = doc_similarities[i]

            redundancy = max(
                util.cos_sim(embeddings[i], embeddings[j])[0][0]
                for j in selected_indices
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            mmr_scores.append(mmr_score)

        next_index = np.argmax(mmr_scores)
        selected_indices.append(next_index)

        selected_indices.sort()

    # Return selected sentences
    return [sentences[i] for i in selected_indices]


# ------------------------------
# HIERARCHICAL SUMMARY PIPELINE
# ------------------------------

def summarize_pdf_hierarchical(pdf_path, chunk_size=50):

    # Extract raw text
    text = extract_text_from_pdf(pdf_path)

    # Split into sentences
    sentences = nltk.sent_tokenize(text)

    # Clean sentences
    sentences = clean_sentences(sentences)

    if not sentences:
        return ["No valid content found in PDF."]

    # STEP 1: Create chunks
    chunks = chunk_sentences(sentences, chunk_size)

    chunk_summaries = []

    # STEP 2: Summarize each chunk
    for chunk in chunks:

        # 20% summary per chunk
        top_k = max(1, int(len(chunk) * 0.2))

        summary = summarize(chunk, top_k=top_k)

        chunk_summaries.extend(summary)

    # STEP 3: Final global summarization
    final_top_k = min(15, max(3, int(len(chunk_summaries) * 0.3)))

    final_summary = summarize(chunk_summaries, top_k=final_top_k)

    return final_summary



# ------------------------------
# 5️⃣ Build Knowledge Base (for RAG)
# ------------------------------

def build_knowledge_base(pdf_path, chunk_size=50):

    text = extract_text_from_pdf(pdf_path)
    sentences = nltk.sent_tokenize(text)
    sentences = clean_sentences(sentences)

    chunks = chunk_sentences(sentences, chunk_size)

    # Convert each chunk list into a single string
    chunk_texts = [" ".join(chunk) for chunk in chunks]

    embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    return chunk_texts, embeddings


# ------------------------------
# 6️⃣ Retrieval for Question Answering (RAG)
# ------------------------------

def answer_question(question, chunks, embeddings, top_k=5):

    if len(chunks) == 0:
        return ["No content available in document."]

    question_embedding = model.encode(question)

    similarities = util.cos_sim(question_embedding, embeddings)[0]

    top_indices = np.argsort(similarities.cpu().numpy())[::-1][:top_k]

    return [chunks[i] for i in top_indices]



# ------------------------------
# 5️⃣ Run on your PDF this code is nopt needed.
# ------------------------------

#if __name__ == "__main__":

#    pdf_path = "document.pdf"# Replace with your PDF file

#   summary = summarize_pdf(pdf_path, top_k=5)

#    print("\n=== SUMMARY ===\n")

#    for sentence in summary:
#       print("-", sentence)