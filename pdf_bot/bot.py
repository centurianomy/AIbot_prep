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
    sentences, embeddings = build_knowledge_base(pdf_path)

    print("PDF loaded successfully!\n")

    while True:

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        top_sentences = answer_question(question, sentences, embeddings)

        context = "\n".join(top_sentences)

        answer = generate_answer_with_llm(question, context)

        print("\nAnswer:\n")
        print(answer)
        print("\n")
