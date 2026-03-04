
# OFFLINE PDF EXTRACTIVE SUMMARIZER (MVP)
# ==============================

import pdfplumber                  # For PDF text extraction
import nltk                        # For sentence splitting
import numpy as np                 # For numerical operations
import re                          # For cleaning text, regex operations
from sentence_transformers import SentenceTransformer, util  # For embeddings + similarity

# Load embedding model ONCE globally
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True #Only load from local cache.If not found → fail immediately. Avoids repeated loading and ensures offline functionality.
)

# Download sentence tokenizer (only first time) Only downloads if missing
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# 1️ Extract text from PDF
# ------------------------------
def extract_text_from_pdf(pdf_path):
    all_text = [] #list to store the text extracted from each page of the PDF

    with pdfplumber.open(pdf_path) as pdf: #file handelling with auto close.
        for page in pdf.pages: # page is like itterator and we can loop through it to get text from each page of the PDF
            text = page.extract_text()
            if text:
                all_text.append(text)

    return "\n".join(all_text)


# 2️ Clean sentences
# ------------------------------
def clean_sentences(sentences):
    cleaned = [] # a list to store cleaned sentences

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Remove short sentences
        if len(sentence.split()) < 3:
            continue

        # Remove sentences without alphabets
        if not re.search(r"[a-zA-Z]", sentence): #incvlude numbers in the check
            continue

        cleaned.append(sentence)

    return cleaned #cleaned sentences stored inside cleaned list and returned at the end of function


# 3 CHUNKING FUNCTION
# ------------------------------
def chunk_sentences(sentences, chunk_size=6, overlap=2): #overlap is the number of sentences that will be repeated in consecutive chunks to maintain context. This helps ensure that important information isn't lost between chunks, especially for sentences that may be relevant to multiple chunks.

    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap): # Start:Stop:Skip-> start from 0, go to the end of sentences, and step by chunk_size - overlap to create overlapping chunks
        chunk = sentences[i:i + chunk_size] # Create a chunk of sentences starting from index i to i + chunk_size. This will create chunks of the specified size, with the specified overlap between consecutive chunks.
        chunks.append(chunk)

    return chunks


# 4 MMR-based summarization(lambda_para is used)
# ------------------------------
def summarize(sentences, top_k=5, lambda_param=0.7): #lambda is a parameter that controls the balance between relevance and redundancy in the MMR scoring. A higher lambda gives more weight to relevance, while a lower lambda gives more weight to redundancy. The default value of 0.7 means that relevance is weighted more heavily than redundancy in the scoring process.

    # Load embedding model
    #model loads inside summarize() every time: This is inefficient, instead load it once globally and reuse it.
    #model = SentenceTransformer("all-MiniLM-L6-v2")


    # Convert sentences to embeddings
    embeddings = model.encode(sentences, batch_size=32) #batch_size is the number of sentences that will be processed together in one go. This can speed up the embedding process by leveraging parallel processing capabilities of the hardware. A batch size of 32 is a common choice that balances memory usage and processing speed, but it can be adjusted based on the available resources and the size of the input data.
    # for better memory (CPU) utilisation 
    # Compute document embedding
    document_embedding = np.mean(embeddings, axis=0) #The document embedding is computed by taking the mean of all sentence embeddings. This creates a single vector that represents the overall content of the document, which can then be used to measure the relevance of each individual sentence to the entire document.
    # for e.g. 
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
            # EXTRA REDUNDANCY CHECK
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


# 5 HIERARCHICAL SUMMARY PIPELINE
# ------------------------------
def summarize_pdf_hierarchical(sentences, chunk_size=50):

    if not sentences:
        return ["No valid content found in PDF."]

    # STEP 1: Create chunks
    chunks = chunk_sentences(sentences, chunk_size)

    chunk_summaries = [] # To store summaries of each chunk

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


# 6 Build Knowledge Base (for RAG)
# ------------------------------

def build_knowledge_base(sentences, chunk_size=6): #sentences, chunk_size=6 are

    chunks = chunk_sentences(sentences, chunk_size)

    # Convert each chunk list into a single string
    chunk_texts = [" ".join(chunk) for chunk in chunks]

    embeddings = model.encode(chunk_texts, convert_to_tensor=True, batch_size=32)
  
    print("Number of units stored:", len(chunk_texts))
    
    #for debugging purpose only: shows how many chunks were created before filtering
    print("==========================================")
    print("Total chunks created:", len(chunks)) 
    print("==========================================")
    
    return chunk_texts, embeddings


# 7 Retrieval for Question Answering (RAG)
# ------------------------------

def answer_question(question, chunks, embeddings, top_k=5):

    if len(chunks) == 0:
        return ["No content available in document."]

    question_embedding = model.encode(question, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, embeddings)[0]

    top_indices = np.argsort(similarities.cpu().numpy())[::-1][:top_k]

    return [chunks[i] for i in top_indices]





