from phase1 import build_knowledge_base, answer_question
import requests

def generate_answer_with_llm(question, context):

    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":

    pdf_path = "document.pdf"  # put your PDF in root folder

    print("Building knowledge base...")
    chunks, embeddings = build_knowledge_base(pdf_path)

    print("PDF loaded successfully!\n")

    while True:

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

            # Brain 1- shot or RAG decision based on question type
            # Brain 2- If question is about summary, use the summarization pipeline.
        if "summarize" in question.lower() or "overview" in question.lower():
    
            from phase1 import summarize_pdf_hierarchical
    
            summary = summarize_pdf_hierarchical(pdf_path) #Hierarchical summarization
            context = "\n".join(summary)
    
        else:
    
            top_chunks = answer_question(question, chunks, embeddings, top_k=2)
            context = "\n\n".join(top_chunks)

        answer = generate_answer_with_llm(question, context)

        print("\nAnswer:\n")
        print(answer)
        print("\n")
